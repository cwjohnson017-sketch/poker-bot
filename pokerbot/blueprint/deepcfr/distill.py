"""Distill an SD-CFR average strategy into one policy network per seat.

The SD-CFR average plays ``sum_t w_t pi_t(I) R_t(I) pi_t(.|I) / sum_t w_t R_t(I)``
over every iteration's net, with the own reach ``R_t`` of the history. Every
query costs one forward pass per net plus the reach bookkeeping; real-time
search makes tens of thousands of them per decision. A distilled net answers
the same query with one forward pass and no reach.

* **Data.** The average strategy plays itself on ``VecNLHE`` with
  ``explore`` uniform mixing at every decision, so off-path lines appear too.
  Every decision is labelled with the exact reach-weighted average given the
  actual history (an exploration action that no net plays leaves the posterior
  at the iteration weights). Rows are the training features of the blueprint.
* **Model.** An :class:`~.networks.AdvantageNet` of the blueprint's config whose
  outputs are logits, trained per seat with the cross-entropy against the
  target distribution over the legal actions.
* **Output.** A run directory with ``checkpoints/p{seat}/iter1.pt`` and a
  ``meta.json`` marking ``policy_head: softmax`` and ``reach_weighted: false``,
  so ``neural:<dir>`` (matches, ABR, ``search:neural:``) loads it like any run.

python scripts/distill_deepcfr.py --run runs/dcfr4 --stride 10 --out runs/dcfr4_distilled
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ...env.vec_env import VecNLHE
from .checkpoint import read_meta, save_net, write_meta
from .config import spec_from_dict
from .features import FeatureConfig, features_from_obs, index_features
from .networks import AdvantageNet, NetConfig
from .policy import SDCFRPolicy
from .strength import add_strength, load_strength

STREET_COL = 8  # street one-hot in the base scalars (env.obs.SCALAR_NAMES)

KEYS = ("cards", "hist", "hist_amt", "scalars", "legal", "target")
_DTYPES = {
    "cards": torch.uint8,
    "hist": torch.uint8,
    "hist_amt": torch.float16,
    "scalars": torch.float16,
    "legal": torch.bool,
    "target": torch.float16,
}


@dataclass
class DistillConfig:
    rows: int = 20_000_000  # total decisions recorded (both seats)
    n_envs: int = 16384
    explore: float = 0.2  # uniform mixing of the behaviour policy
    steps: int = 40_000  # Adam steps per seat
    batch: int = 8192
    lr: float = 1e-3
    lr_final: float = 1e-5  # cosine decay to this
    holdout: float = 0.02
    seed: int = 0
    # share of each training batch per street (preflop, flop, turn, river);
    # empty = sample rows uniformly (preflop decisions dominate the data)
    street_mix: tuple = ()
    width: int = 0  # trunk width of the distilled net (0 = the blueprint's)


def _game(meta: dict[str, Any]) -> Any:
    from ...env.config import GameConfig

    g = meta["game"]
    return GameConfig(
        num_players=2,
        stacks=[int(s) for s in g["stacks"]],
        small_blind=int(g["small_blind"]),
        big_blind=int(g["big_blind"]),
        ante=int(g.get("ante", 0)),
    )


@torch.no_grad()
def generate(
    policies: list[SDCFRPolicy],
    meta: dict[str, Any],
    cfg: DistillConfig,
    device: torch.device,
    log: Any = print,
) -> list[dict[str, torch.Tensor]]:
    """Per seat, the compact rows ``KEYS`` (CPU) of ``cfg.rows`` decisions in all."""
    spec = spec_from_dict(meta.get("spec"))
    features = FeatureConfig.from_dict(meta.get("features"))
    strength = load_strength(features.strength_tables) if features.strength_tables else None
    env = VecNLHE(cfg.n_envs, _game(meta), device, seed=cfg.seed, spec=spec, validate=False)
    gen = torch.Generator(device=device).manual_seed(cfg.seed + 1)
    obs_gen = torch.Generator(device=device).manual_seed(cfg.seed + 2)
    reach = [torch.zeros(len(p), env.n, dtype=torch.float64, device=device) for p in policies]
    parts: list[dict[str, list]] = [{k: [] for k in KEYS} for _ in policies]
    rows, t0 = 0, time.time()
    while rows < cfg.rows:
        live = ~env.done
        obs = env.obs(**features.obs_kwargs(), generator=obs_gen)
        feats = features_from_obs(obs)
        action = env.tab.call_index[env.street.clamp(0, 3)].clone()
        for seat, pol in enumerate(policies):
            idx = (live & (env.actor == seat)).nonzero().squeeze(1)
            if idx.numel() == 0:
                continue
            f = index_features(feats, idx)
            if strength is not None:
                f = add_strength(f, strength)
            lr = reach[seat][:, idx] if pol.reach_weighted else None
            avg, P = pol.average(f, lr)
            legal = f["legal"].float()
            avg = avg.float() * legal
            uni = legal / legal.sum(1, keepdim=True)
            target = torch.where(
                avg.sum(1, keepdim=True) > 0, avg / avg.sum(1, keepdim=True).clamp(min=1e-30), uni
            )
            behave = (1 - cfg.explore) * target + cfg.explore * uni
            a = torch.multinomial(behave, 1, generator=gen).squeeze(1)
            action[idx] = a
            if pol.reach_weighted:
                p_a = P.gather(2, a[None, :, None].expand(P.shape[0], -1, 1)).squeeze(2)
                reach[seat][:, idx] += torch.log(p_a.double().clamp(min=0))
            row = {k: f[k] for k in KEYS[:-1]}
            row["target"] = target
            for k in KEYS:
                parts[seat][k].append(row[k].to("cpu", _DTYPES[k]))
            rows += int(idx.numel())
        _, done = env.step(action)
        if bool(done.any()):
            for r in reach:
                r[:, done] = 0.0
            env.reset(done)
        if log and rows // 1_000_000 != (rows - int(live.sum())) // 1_000_000:
            log(f"# distill data: {rows:,} rows, {time.time() - t0:.0f}s")
    return [{k: torch.cat(v) for k, v in p.items()} for p in parts]


def _batch(data: dict[str, torch.Tensor], idx: torch.Tensor, device: torch.device) -> dict:
    b = {k: data[k][idx].to(device) for k in KEYS}
    cards = b["cards"].long()
    return {
        "cards": cards,
        "card_mask": cards < 52,
        "hist": b["hist"].long(),
        "hist_amt": b["hist_amt"].float(),
        "scalars": b["scalars"].float(),
        "legal": b["legal"],
        "target": b["target"].float(),
    }


def _loss(net: AdvantageNet, b: dict, amp: bool) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.autocast(device_type=b["cards"].device.type, dtype=torch.bfloat16, enabled=amp):
        logits = net(b)
    logits = logits.float().masked_fill(~b["legal"], float("-inf"))
    logp = torch.log_softmax(logits, -1)
    t = b["target"]
    ce = -(t * logp.masked_fill(~b["legal"], 0.0)).sum(1)
    tv = 0.5 * (logp.exp() - t).abs().sum(1)
    return ce.mean(), tv


def train_seat(
    data: dict[str, torch.Tensor],
    net_cfg: NetConfig,
    cfg: DistillConfig,
    device: torch.device,
    log: Any = print,
    label: str = "",
) -> tuple[AdvantageNet, dict[str, float]]:
    data = {k: v.to(device) for k, v in data.items()}  # ~165 B per row
    n = data["target"].shape[0]
    g = torch.Generator(device=device).manual_seed(cfg.seed + 3)
    perm = torch.randperm(n, generator=g, device=device)
    n_val = max(1, int(n * cfg.holdout))
    val, tr = perm[:n_val], perm[n_val:]
    street = data["scalars"][:, STREET_COL : STREET_COL + 4].float().argmax(1)
    by_street = [tr[street[tr] == s] for s in range(4)]
    mix = torch.tensor(cfg.street_mix or [0.0] * 4, dtype=torch.float64)
    stratified = bool(cfg.street_mix) and all(len(b) > 0 for b in by_street)
    counts = (mix / mix.sum() * cfg.batch).round().long().tolist() if stratified else None
    torch.manual_seed(cfg.seed + 4)
    net = AdvantageNet(net_cfg).to(device).train()
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, fused=device.type == "cuda")
    amp = device.type == "cuda"
    t0 = time.time()

    @torch.no_grad()
    def evaluate() -> dict[str, float]:
        net.eval()
        tvs = []
        for lo in range(0, n_val, 65536):
            _, tv = _loss(net, _batch(data, val[lo : lo + 65536], device), amp)
            tvs.append(tv.float().cpu())
        net.train()
        tv = torch.cat(tvs)
        st = street[val].cpu()
        out = {
            "tv_mean": float(tv.mean()),
            "tv_p90": float(tv.quantile(0.9)),
            "tv_p99": float(tv.quantile(0.99)),
        }
        for s, name in enumerate(("preflop", "flop", "turn", "river")):
            m = st == s
            out[f"tv_{name}"] = float(tv[m].mean()) if bool(m.any()) else float("nan")
        return out

    for step in range(cfg.steps):
        frac = step / max(1, cfg.steps - 1)
        lr = cfg.lr_final + 0.5 * (cfg.lr - cfg.lr_final) * (1 + math.cos(math.pi * frac))
        for pg in opt.param_groups:
            pg["lr"] = lr
        if stratified:
            idx = torch.cat(
                [
                    b[torch.randint(0, b.numel(), (k,), generator=g, device=device)]
                    for b, k in zip(by_street, counts, strict=True)
                    if k > 0
                ]
            )
        else:
            idx = tr[torch.randint(0, tr.numel(), (cfg.batch,), generator=g, device=device)]
        loss, _ = _loss(net, _batch(data, idx, device), amp)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if log and (step + 1) % max(1, cfg.steps // 10) == 0:
            ev = evaluate()
            log(
                f"# distill {label} step {step + 1}: loss {float(loss.detach()):.4f}, held-out TV "
                f"mean {ev['tv_mean']:.4f} p90 {ev['tv_p90']:.4f} p99 {ev['tv_p99']:.4f}; streets "
                f"{ev['tv_preflop']:.3f} / {ev['tv_flop']:.3f} / {ev['tv_turn']:.3f} / "
                f"{ev['tv_river']:.3f}, {time.time() - t0:.0f}s"
            )
    return net.eval(), evaluate()


def distill(
    run: str | Path,
    out: str | Path,
    cfg: DistillConfig | None = None,
    stride: int | None = None,
    last_n: int | None = None,
    device: str = "cuda",
    log: Any = print,
    data_path: str | Path | None = None,
) -> dict[str, Any]:
    """Distill ``run`` into ``out``. ``data_path`` caches the generated rows:
    loaded when the file exists, else written after generation."""
    cfg = cfg or DistillConfig()
    log = log or (lambda *_a, **_k: None)
    dev = torch.device(device)
    meta = read_meta(run)
    fallback = meta.get("fallback", "uniform")
    policies = [
        SDCFRPolicy.from_dir(run, p, last_n=last_n, stride=stride, device=dev, fallback=fallback)
        for p in (0, 1)
    ]
    log(f"# distilling {run}: {len(policies[0])} nets per seat (stride {stride}), {cfg}")
    if data_path is not None and Path(data_path).exists():
        data = torch.load(data_path, weights_only=True)
        log(f"# loaded {sum(int(d['target'].shape[0]) for d in data):,} rows from {data_path}")
    else:
        data = generate(policies, meta, cfg, dev, log)
        if data_path is not None:
            Path(data_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(data, data_path)
    del policies
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    net_cfg = NetConfig.from_dict(meta["net_config"])
    if cfg.width:
        net_cfg = NetConfig.from_dict({**net_cfg.to_dict(), "width": int(cfg.width)})
    meta = {**meta, "net_config": net_cfg.to_dict()}
    out = Path(out)
    root = out / "checkpoints"
    new_meta = {k: v for k, v in meta.items() if k != "preflop"}
    new_meta.update(
        policy_head="softmax",
        reach_weighted=False,
        distilled_from=str(run),
        distill={**cfg.__dict__, "stride": stride, "last_n": last_n},
    )
    write_meta(root, new_meta)
    report = {}
    for seat in (0, 1):
        net, ev = train_seat(data[seat], net_cfg, cfg, dev, log, label=f"seat {seat}")
        save_net(root, seat, 1, net, new_meta)
        report[f"seat{seat}"] = {"rows": int(data[seat]["target"].shape[0]), **ev}
    (out / "distill.json").write_text(json.dumps(report, indent=1))
    return report
