"""Deep CFR / SD-CFR training loop (DESIGN.md 5.5 "Loop per CFR iteration").

Per iteration ``t`` and per player ``p`` (player 1 already sees player 0's
network from the same iteration, as in the Deep CFR paper):

1. ``traversals_per_iter`` external-sampling traversals for ``p`` in batches
   of ``roots_per_batch`` root hands, with both players' current
   regret-matching policies; regret samples go into ``p``'s reservoir
   advantage memory with iteration weight ``t``.
2. Re-initialize ``p``'s advantage net (or warm start with ``reinit: false``)
   and train it for ``sgd_steps`` Adam steps on minibatches from the memory
   with the iteration-weighted MSE (linear CFR):
   ``sum_i t_i * sum_a legal_ia * (net(x_i)_a - r_ia)^2 / sum_i t_i``.
3. Save ``checkpoints/p{p}/iter{t}.pt``. The list of all checkpoints is the
   SD-CFR average strategy (:class:`~.policy.SDCFRPolicy`).

With ``memory.holdout > 0`` that fraction of the regret samples goes to a
separate validation reservoir instead; after each fit the net's
iteration-weighted R^2 on it is logged per street (``val_r2_*``: 1 - weighted
MSE / weighted mean square target, so predicting zero scores 0).

Every ``save_every`` iterations the memories and RNG states are saved as a
resume point; every ``eval.every`` iterations the current average strategy
plays duplicate matches against ``EquityThresholdAgent`` and against the
average strategy of the previous evaluation, on the same deals every time
(``eval.seed``), with luck-adjusted results next to the raw ones. Logs:
``log.csv`` and TensorBoard (``tb/``) under the run directory.
"""

from __future__ import annotations

import csv
import json
import queue
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ...config import git_hash
from ...env.obs import SCALAR_NAMES
from .checkpoint import list_checkpoints, load_net, save_net, write_meta
from .config import DeepCFRConfig, spec_to_dict
from .memory import ReservoirMemory, decode
from .networks import AdvantageNet, num_params
from .traversal import FrontierTraverser, NetPolicy

CSV_FIELDS = (
    "iteration",
    "player",
    "loss",
    "loss_first",
    "adv_mem_size",
    "adv_mem_seen",
    "strat_mem_size",
    "nodes",
    "slot_steps",
    "roots",
    "max_frontier",
    "cut_slots",
    "allin_leaves",
    "regret_abs_mean",
    "val_r2_preflop",
    "val_r2_flop",
    "val_r2_turn",
    "val_r2_river",
    "val_r2_all",
    "traversal_s",
    "train_s",
    "slot_steps_per_s",
    "wall_s",
)
EVAL_FIELDS = (
    "iteration",
    "opponent",
    "opponent_iter",
    "mbb_per_hand",
    "ci_low",
    "ci_high",
    "mbb_adj",
    "ci_adj_low",
    "ci_adj_high",
    "hands",
    "eval_s",
)
STREETS = ("preflop", "flop", "turn", "river")
_STREET_COL = SCALAR_NAMES.index("preflop")  # street one-hot in the scalars


class _Prefetcher:
    """Host thread that keeps ``depth`` minibatches ready (overlaps numpy
    gathers with GPU compute)."""

    def __init__(self, mem: ReservoirMemory, batch: int, device: torch.device, depth: int, n: int):
        self.q: queue.Queue = queue.Queue(maxsize=depth)
        self.rng = np.random.default_rng(int(mem.rng.integers(2**63)))
        self.thread = threading.Thread(target=self._run, args=(mem, batch, device, n), daemon=True)
        self.thread.start()

    def _run(self, mem: ReservoirMemory, batch: int, device: torch.device, n: int) -> None:
        try:
            for _ in range(n):
                self.q.put(mem.sample(batch, device, self.rng))
        except Exception as e:  # surfaced in the training thread
            self.q.put(e)

    def get(self) -> dict[str, torch.Tensor]:
        item = self.q.get()
        if isinstance(item, Exception):
            raise item
        return item


class _ChunkSampler:
    """Minibatches sliced on the device from large chunks of the memory.

    A host thread draws ``rows`` rows uniformly (with replacement), gathers them
    in the compact storage dtypes and moves them to ``device``; the training
    loop then takes minibatches in a random order from the device chunk, so the
    host does one gather per ``rows / batch`` steps instead of one per step.
    Every minibatch is still a uniform sample of the memory."""

    def __init__(
        self, mem: ReservoirMemory, batch: int, device: torch.device, rows: int, steps: int
    ) -> None:
        self.batch, self.device = int(batch), device
        self.rows = max(int(rows), self.batch)
        self.per_chunk = self.rows // self.batch
        n_chunks = -(-int(steps) // self.per_chunk)
        seed = int(mem.rng.integers(2**62))
        self.gen = torch.Generator(device=device).manual_seed(seed)
        self.q: queue.Queue = queue.Queue(maxsize=1)
        self.thread = threading.Thread(
            target=self._run, args=(mem, np.random.default_rng(seed), n_chunks), daemon=True
        )
        self.thread.start()
        self.cur: dict[str, torch.Tensor] | None = None
        self.perm: torch.Tensor | None = None
        self.pos = self.per_chunk

    def _run(self, mem: ReservoirMemory, rng: np.random.Generator, n: int) -> None:
        try:
            for _ in range(n):
                idx = np.sort(rng.integers(0, len(mem), size=self.rows))
                host = mem.gather_compact(idx)
                self.q.put({k: v.to(self.device) for k, v in host.items()})
        except Exception as e:  # surfaced in the training thread
            self.q.put(e)

    def get(self) -> dict[str, torch.Tensor]:
        if self.pos >= self.per_chunk:
            item = self.q.get()
            if isinstance(item, Exception):
                raise item
            self.cur = item
            self.perm = torch.randperm(self.rows, device=self.device, generator=self.gen)
            self.pos = 0
        assert self.cur is not None and self.perm is not None
        idx = self.perm[self.pos * self.batch : (self.pos + 1) * self.batch]
        self.pos += 1
        return decode({k: v[idx] for k, v in self.cur.items()})


class DeepCFRTrainer:
    def __init__(
        self, cfg: DeepCFRConfig, out_dir: str | Path | None = None, resume: bool = False
    ) -> None:
        self.cfg = cfg
        self.out = Path(out_dir or cfg.out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.ckpt_root = self.out / "checkpoints"
        self.device = cfg.torch_device()
        if cfg.threads:
            torch.set_num_threads(int(cfg.threads))
        torch.manual_seed(cfg.seed)
        self.game_config = cfg.game_config()
        self.spec = cfg.spec()
        self.net_cfg = cfg.net_config()
        self.tcfg = cfg.traversal_config()
        self.traverser = FrontierTraverser(
            self.game_config, self.spec, self.tcfg, self.device, cfg.seed
        )
        self.amp_dtype = (
            torch.bfloat16 if (cfg.training.bf16 and self.device.type == "cuda") else None
        )
        if self.device.type == "cuda":  # TF32 for the fp32 parts (GRU, loss)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        A, S = self.net_cfg.num_actions, self.net_cfg.num_scalars
        T, V = self.net_cfg.history_len, self.net_cfg.vocab_size
        self.adv_mem = [
            ReservoirMemory(cfg.memory.capacity, A, S, T, V, cfg.seed * 10 + p, f"adv_p{p}")
            for p in (0, 1)
        ]
        self.val_mem: list[ReservoirMemory] | None = None
        if cfg.memory.holdout > 0:
            cap = cfg.memory.holdout_capacity
            self.val_mem = [
                ReservoirMemory(cap, A, S, T, V, cfg.seed * 10 + 7 + p, f"val_p{p}") for p in (0, 1)
            ]
        self._holdout_rng = np.random.default_rng(cfg.seed * 10 + 9)
        self.strat_mem: list[ReservoirMemory] | None = None
        if cfg.traversal.record_strategy:
            cap = cfg.memory.strategy_capacity or cfg.memory.capacity
            self.strat_mem = [
                ReservoirMemory(cap, A, S, T, V, cfg.seed * 10 + 5 + p, f"strat_p{p}")
                for p in (0, 1)
            ]
        self.nets: list[AdvantageNet | None] = [None, None]
        self.checkpoints: list[list[tuple[int, Path]]] = [[], []]
        self.iteration = 0
        self.last_eval_iter = 0
        self.meta = {
            "spec": spec_to_dict(self.spec),
            "features": cfg.features.to_dict(),
            "game": {
                "stacks": list(self.game_config.stacks),
                "small_blind": self.game_config.small_blind,
                "big_blind": self.game_config.big_blind,
                "ante": self.game_config.ante,
            },
            "net_config": self.net_cfg.to_dict(),
            "value_scale": self.traverser.value_scale,
            "fallback": cfg.training.fallback,
            "git": git_hash(),
        }
        write_meta(self.ckpt_root, self.meta)
        (self.out / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2, default=str))
        if resume:
            self._resume()
        self._tb = None
        if cfg.logging.tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self._tb = SummaryWriter(str(self.out / "tb"))
            except ImportError:  # tensorboard not installed: CSV only
                self._tb = None
        self._log(
            f"# deepcfr: device {self.device}, net params {num_params(AdvantageNet(self.net_cfg))}"
            f", memory {self.adv_mem[0].bytes_per_sample} B/sample, start at iter "
            f"{self.iteration + 1}"
        )

    # ---------------------------------------------------------------- policies
    def policy(self, p: int) -> NetPolicy:
        return NetPolicy(
            self.nets[p], self.cfg.training.fallback, self.amp_dtype, self.cfg.traversal.infer_chunk
        )

    # ---------------------------------------------------------------- phases
    def collect(self, p: int, t: int) -> dict[str, float]:
        """Traversals for player ``p`` at iteration ``t`` into its memory."""
        tr = self.cfg.traversal
        policies = [self.policy(0), self.policy(1)]
        agg: dict[str, float] = {
            "nodes": 0,
            "slot_steps": 0,
            "roots": 0,
            "max_frontier": 0,
            "cut_slots": 0,
            "allin_leaves": 0,
        }
        reg_sum = 0.0
        remaining = tr.traversals_per_iter
        while remaining > 0:
            k = min(tr.roots_per_batch, remaining)
            res = self.traverser.traverse(p, policies, t, k)
            self._store(p, res.samples)
            if self.strat_mem is not None and res.strategy is not None:
                self.strat_mem[1 - p].add_batch(res.strategy)
            s = res.stats
            for key in ("nodes", "slot_steps", "roots", "cut_slots", "allin_leaves"):
                agg[key] += s[key]
            agg["max_frontier"] = max(agg["max_frontier"], s["max_frontier"])
            reg_sum += s["regret_abs_mean"] * s["nodes"]
            remaining -= k
        agg["regret_abs_mean"] = reg_sum / max(1, agg["nodes"])
        return agg

    def _store(self, p: int, samples: dict[str, torch.Tensor]) -> None:
        """Regret samples into ``p``'s memory, minus the held-out fraction."""
        if self.val_mem is None:
            self.adv_mem[p].add_batch(samples)
            return
        n = int(samples["target"].shape[0])
        hold = self._holdout_rng.random(n) < self.cfg.memory.holdout
        mask = torch.from_numpy(hold).to(samples["target"].device)
        self.val_mem[p].add_batch({k: v[mask] for k, v in samples.items()})
        self.adv_mem[p].add_batch({k: v[~mask] for k, v in samples.items()})

    @torch.no_grad()
    def validate(self, p: int, net: AdvantageNet, chunk: int = 65536) -> dict[str, float]:
        """Iteration-weighted R^2 of ``net`` on ``p``'s held-out samples, per
        street and overall (NaN without a validation memory)."""
        out = {f"val_r2_{k}": float("nan") for k in (*STREETS, "all")}
        mem = self.val_mem[p] if self.val_mem is not None else None
        if mem is None or len(mem) == 0:
            return out
        res = torch.zeros(4, dtype=torch.float64, device=self.device)
        tot = torch.zeros(4, dtype=torch.float64, device=self.device)
        for lo in range(0, len(mem), chunk):
            b = mem.gather(np.arange(lo, min(len(mem), lo + chunk)), self.device)
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype or torch.float32,
                enabled=self.amp_dtype is not None,
            ):
                adv = net(b)
            legal = b["legal"].float()
            w = b["iteration"].double()
            err = (((adv.float() - b["target"]) ** 2) * legal).sum(1).double() * w
            sq = ((b["target"] ** 2) * legal).sum(1).double() * w
            street = b["scalars"][:, _STREET_COL : _STREET_COL + 4].argmax(1)
            res.index_add_(0, street, err)
            tot.index_add_(0, street, sq)
        r, q = res.cpu().numpy(), tot.cpu().numpy()
        for i, k in enumerate(STREETS):
            if q[i] > 0:
                out[f"val_r2_{k}"] = float(1 - r[i] / q[i])
        if q.sum() > 0:
            out["val_r2_all"] = float(1 - r.sum() / q.sum())
        return out

    def train_net(self, p: int) -> tuple[AdvantageNet, float, float]:
        """Fit ``p``'s advantage net on its memory. Returns (net, final loss, first loss)."""
        tc = self.cfg.training
        mem = self.adv_mem[p]
        if tc.reinit or self.nets[p] is None:
            net = AdvantageNet(self.net_cfg).to(self.device)
        else:
            net = AdvantageNet(self.net_cfg).to(self.device)
            net.load_state_dict(self.nets[p].state_dict())
        net.train()
        params = list(net.parameters())
        opt = torch.optim.Adam(
            params, lr=tc.lr, weight_decay=tc.weight_decay, fused=self.device.type == "cuda"
        )
        # exponential moving average of the weights (the returned net), with the
        # usual warm-up so the random initialization does not linger in it
        ema = [q.detach().clone() for q in params] if tc.ema_decay > 0 else None
        steps = tc.sgd_steps
        pre: _Prefetcher | _ChunkSampler | None = None
        if tc.chunk_rows > 0:
            pre = _ChunkSampler(mem, tc.batch_size, self.device, tc.chunk_rows, steps)
        elif tc.prefetch > 0:
            pre = _Prefetcher(mem, tc.batch_size, self.device, tc.prefetch, steps)
        tail = max(1, steps // 10)
        losses: list[torch.Tensor] = []
        first = float("nan")
        for step in range(steps):
            b = pre.get() if pre else mem.sample(tc.batch_size, self.device)
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype or torch.float32,
                enabled=self.amp_dtype is not None,
            ):
                adv = net(b)
            legal = b["legal"].float()
            err = ((adv.float() - b["target"]) ** 2 * legal).sum(1)
            w = b["iteration"]
            loss = (w * err).sum() / w.sum().clamp(min=1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if tc.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, tc.grad_clip)
            opt.step()
            if ema is not None:
                with torch.no_grad():
                    decay = min(tc.ema_decay, (1 + step) / (10 + step))
                    torch._foreach_lerp_(ema, params, 1.0 - decay)
            if step == 0:
                first = float(loss.detach())
            if step >= steps - tail:
                losses.append(loss.detach())
        final = float(torch.stack(losses).mean()) if losses else float("nan")
        if ema is not None:
            with torch.no_grad():
                torch._foreach_copy_(params, ema)
        return net.eval(), final, first

    def run(self, iterations: int | None = None) -> None:
        target = int(iterations if iterations is not None else self.cfg.iterations)
        for t in range(self.iteration + 1, target + 1):
            self.run_iteration(t)

    def run_iteration(self, t: int) -> list[dict[str, Any]]:
        rows = []
        t_iter = time.time()
        for p in (0, 1):
            t0 = time.time()
            st = self.collect(p, t)
            t1 = time.time()
            net, loss, first = self.train_net(p)
            t2 = time.time()
            val = self.validate(p, net)
            self.nets[p] = net
            path = save_net(
                self.ckpt_root, p, t, net, self.meta, self.cfg.training.checkpoint_dtype
            )
            self.checkpoints[p].append((t, path))
            row = {
                "iteration": t,
                "player": p,
                "loss": loss,
                "loss_first": first,
                "adv_mem_size": len(self.adv_mem[p]),
                "adv_mem_seen": self.adv_mem[p].seen,
                "strat_mem_size": len(self.strat_mem[1 - p]) if self.strat_mem else 0,
                **st,
                **val,
                "traversal_s": t1 - t0,
                "train_s": t2 - t1,
                "slot_steps_per_s": st["slot_steps"] / max(1e-9, t1 - t0),
                "wall_s": time.time() - t_iter,
            }
            rows.append(row)
            self._write_row(row)
        self.iteration = t
        me = self.cfg.memory.save_every
        if me and t % me == 0:
            self.save_state()
        ev = self.cfg.eval.every
        if ev and t % ev == 0:
            self.evaluate(t)
        return rows

    # ---------------------------------------------------------------- evaluation
    def evaluate(self, t: int) -> list[dict[str, Any]]:
        from ...agents import EquityThresholdAgent
        from ...config import game_config
        from ...engine_select import get_engine
        from ...eval.match import run_duplicate_match
        from .agent import NeuralBlueprintAgent

        ec = self.cfg.eval
        engine = get_engine(ec.engine)
        gcfg = game_config(self.meta["game"], engine)
        dev = str(ec.device or self.device)
        cur = NeuralBlueprintAgent.from_dir(
            self.ckpt_root, ec.last_n, t, device=dev, name=f"sdcfr_{t}", sample_net=ec.sample_net
        )
        opponents = []
        if ec.vs_equity:
            opponents.append(("equity", EquityThresholdAgent(**ec.equity)))
        prev_t = self.last_eval_iter if self.last_eval_iter else t - 1
        if ec.vs_previous and prev_t >= 1:
            prev = NeuralBlueprintAgent.from_dir(
                self.ckpt_root,
                ec.last_n,
                prev_t,
                device=dev,
                name=f"sdcfr_{prev_t}",
                sample_net=ec.sample_net,
            )
            opponents.append(("previous", prev))
        rows = []
        seed = t if ec.seed is None else int(ec.seed)  # fixed seed: the same deals every time
        nan = float("nan")
        for label, opp in opponents:
            t0 = time.time()
            res = run_duplicate_match(
                cur,
                opp,
                gcfg,
                ec.deals,
                seed=seed,
                engine=engine,
                luck_adjust=ec.luck_adjust,
                adjust_device=dev,
            )
            adj = res.adjusted_stats
            row = {
                "iteration": t,
                "opponent": label,
                "opponent_iter": prev_t if label == "previous" else "",
                "mbb_per_hand": res.mbb_per_hand,
                "ci_low": res.ci[0],
                "ci_high": res.ci[1],
                "mbb_adj": adj.mbb_per_hand if adj else nan,
                "ci_adj_low": adj.ci_low if adj else nan,
                "ci_adj_high": adj.ci_high if adj else nan,
                "hands": res.hands,
                "eval_s": time.time() - t0,
            }
            rows.append(row)
            self._write_eval(row)
        self.last_eval_iter = t
        return rows

    # ---------------------------------------------------------------- persistence
    def save_state(self) -> None:
        mem_dir = self.out / "memory"
        for p in (0, 1):
            self.adv_mem[p].save(mem_dir / f"adv_p{p}")
            if self.strat_mem is not None:
                self.strat_mem[p].save(mem_dir / f"strat_p{p}")
        if self.val_mem is not None:
            for p in (0, 1):
                self.val_mem[p].save(mem_dir / f"val_p{p}")
        roots = self.traverser._roots
        state = {
            "iteration": self.iteration,
            "last_eval_iter": self.last_eval_iter,
            "holdout_rng": self._holdout_rng.bit_generator.state,
            "traverser_gen": self.traverser.generator.get_state(),
            "roots_gen": roots.generator.get_state() if roots is not None else None,
            "roots_n": roots.n if roots is not None else 0,
            "torch_rng": torch.get_rng_state(),
        }
        tmp = self.out / "trainer_state.tmp"
        torch.save(state, tmp)
        tmp.replace(self.out / "trainer_state.pt")

    def _resume(self) -> None:
        path = self.out / "trainer_state.pt"
        if not path.exists():
            raise FileNotFoundError(f"no resume point at {path}")
        state = torch.load(path, map_location="cpu", weights_only=False)
        t = int(state["iteration"])
        mem_dir = self.out / "memory"
        for p in (0, 1):
            self.adv_mem[p] = ReservoirMemory.load(
                mem_dir / f"adv_p{p}", capacity=self.cfg.memory.capacity
            )
            if self.strat_mem is not None:
                self.strat_mem[p] = ReservoirMemory.load(
                    mem_dir / f"strat_p{p}", capacity=self.strat_mem[p].capacity
                )
            if self.val_mem is not None and (mem_dir / f"val_p{p}").exists():
                self.val_mem[p] = ReservoirMemory.load(
                    mem_dir / f"val_p{p}", capacity=self.val_mem[p].capacity
                )
            cks = list_checkpoints(self.ckpt_root, p)
            # nets saved after the resume point are discarded (their data is gone)
            for it, f in cks:
                if it > t:
                    f.unlink()
            self.checkpoints[p] = [(it, f) for it, f in cks if it <= t]
            if self.checkpoints[p]:
                _, net = load_net(self.checkpoints[p][-1][1], self.device)
                self.nets[p] = net
        self.iteration = t
        self.last_eval_iter = int(state.get("last_eval_iter", 0))
        if state.get("holdout_rng") is not None:
            self._holdout_rng.bit_generator.state = state["holdout_rng"]
        self.traverser.generator.set_state(state["traverser_gen"])
        if state.get("roots_gen") is not None and state["roots_n"]:
            self.traverser.new_roots(int(state["roots_n"]))
            self.traverser._roots.generator.set_state(state["roots_gen"])
        torch.set_rng_state(state["torch_rng"])

    # ---------------------------------------------------------------- logging
    def _log(self, msg: str) -> None:
        if self.cfg.logging.print:
            print(msg, flush=True)

    def _write_row(self, row: dict[str, Any]) -> None:
        if self.cfg.logging.csv:
            _append_csv(self.out / "log.csv", CSV_FIELDS, row)
        if self._tb is not None:
            t, p = row["iteration"], row["player"]
            for k in CSV_FIELDS[2:]:
                if k.startswith("val_r2_") and not np.isfinite(row[k]):
                    continue
                self._tb.add_scalar(f"p{p}/{k}", float(row[k]), t)
            self._tb.flush()
        r2 = ""
        if np.isfinite(row["val_r2_all"]):
            r2 = " val R2 " + " ".join(f"{row[f'val_r2_{k}']:.3f}" for k in (*STREETS, "all"))
        self._log(
            f"iter {row['iteration']} p{row['player']}: loss {row['loss']:.4f} "
            f"(first {row['loss_first']:.4f}) mem {row['adv_mem_size']} nodes {row['nodes']} "
            f"trav {row['traversal_s']:.1f}s ({row['slot_steps_per_s']:.0f} slot-steps/s) "
            f"train {row['train_s']:.1f}s{r2}"
        )

    def _write_eval(self, row: dict[str, Any]) -> None:
        if self.cfg.logging.csv:
            _append_csv(self.out / "eval.csv", EVAL_FIELDS, row)
        adjusted = np.isfinite(row["mbb_adj"])
        if self._tb is not None:
            tag = f"eval/mbb_vs_{row['opponent']}"
            self._tb.add_scalar(tag, row["mbb_per_hand"], row["iteration"])
            if adjusted:
                self._tb.add_scalar(
                    f"eval/mbb_adj_vs_{row['opponent']}", row["mbb_adj"], row["iteration"]
                )
            self._tb.flush()
        adj = (
            f"; luck-adjusted {row['mbb_adj']:+.1f} [{row['ci_adj_low']:+.1f}, "
            f"{row['ci_adj_high']:+.1f}]"
            if adjusted
            else ""
        )
        self._log(
            f"eval iter {row['iteration']} vs {row['opponent']}: {row['mbb_per_hand']:+.1f} mbb/h "
            f"[{row['ci_low']:+.1f}, {row['ci_high']:+.1f}]{adj} over {row['hands']} hands"
        )

    def close(self) -> None:
        if self._tb is not None:
            self._tb.close()


def _append_csv(path: Path, fields: tuple[str, ...], row: dict[str, Any]) -> None:
    new = not path.exists()
    if not new:
        # keep the columns of an existing file (a run resumed with newer code)
        with open(path, newline="") as fh:
            header = next(csv.reader(fh), None)
        if header:
            fields = tuple(header)
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)
