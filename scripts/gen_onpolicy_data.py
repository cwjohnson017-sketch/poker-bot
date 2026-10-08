#!/usr/bin/env python
"""On-policy value-net training data: the leaf states a value-net search queries.

1. **Record.** A value-net search agent (``--search-config``, leaf model
   ``--leaf-net``) plays duplicate deals against the blueprint. In its flop
   searches a :class:`~pokerbot.search.leaf_recorder.RecordingLeafProvider`
   stores turn-end leaf states (both current reaches, normalised, (OOP, IP)) on
   every ``--every``-th regret update, ``--per-call`` leaves at a time. The
   turn-end states are saved to ``<out>/turn_states.pt`` (for the turn-end net).
2. **River states.** ``--rivers-per-state`` random river cards per turn-end
   state (:func:`~pokerbot.search.leaf_recorder.river_states`).
3. **Solve.** Exactly, with ``BatchRiverSolver`` (``value_data.solve_states``),
   into river shards of the usual format with ``source`` 3 (on-policy).

python scripts/gen_onpolicy_data.py --blueprint runs/dcfr4_distilled_v2 \
    --leaf-net runs/value_net/turn_v1.pt --deals 150 --out runs/value_net/river_onpolicy
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pokerbot.agents import make_agent  # noqa: E402
from pokerbot.engine_select import get_engine  # noqa: E402
from pokerbot.eval.match import run_duplicate_match  # noqa: E402
from pokerbot.search.agent import SearchAgent  # noqa: E402
from pokerbot.search.blueprint import make_blueprint  # noqa: E402
from pokerbot.search.config import search_config  # noqa: E402
from pokerbot.search.leaf_recorder import (  # noqa: E402
    LeafStateStore,
    RecordingLeafProvider,
    river_states,
)
from pokerbot.search.value_data import shard_name, solve_states, write_atomic  # noqa: E402

SOURCE_ONPOLICY = 3


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


class RecordingSearchAgent(SearchAgent):
    """A search agent whose value-net leaves record their states into ``store``."""

    def __init__(self, *a, store: LeafStateStore, record: dict, **k) -> None:
        super().__init__(*a, **k)
        self.store = store
        self.record = record
        self._n = 0

    def value_leaf_provider(self, tree):
        self._n += 1
        inner = super().value_leaf_provider(tree)
        return RecordingLeafProvider(inner, tree, self.store, seed=self._n, **self.record)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--blueprint", required=True)
    ap.add_argument("--leaf-net", required=True, help="leaf model of the recording searches")
    ap.add_argument("--search-config", default="configs/search_value_net.yaml")
    ap.add_argument("--deals", type=int, default=150, help="duplicate deals (2 hands each)")
    ap.add_argument("--chunk", type=int, default=25)
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--per-call", type=int, default=8)
    ap.add_argument("--min-mass", type=float, default=1e-5)
    ap.add_argument("--rivers-per-state", type=int, default=2)
    ap.add_argument("--iterations", type=int, default=300)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--shard-size", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(args.device)
    engine = get_engine()
    cfg = engine.GameConfig(num_players=2, stacks=[10000, 10000], small_blind=50, big_blind=100)
    bp_spec = f"neural:{args.blueprint}"
    bp = make_blueprint(bp_spec, device=str(dev))
    store = LeafStateStore()
    record = {"every": args.every, "per_call": args.per_call, "min_mass": args.min_mass}
    sc = search_config(args.search_config, leaf={"net": args.leaf_net}, device=str(dev))
    agent = RecordingSearchAgent(bp, sc, name="search_rec", store=store, record=record)
    opp = make_agent(bp_spec, device=str(dev))
    turn_path = out / "turn_states.pt"

    # 1. record
    t0 = time.time()
    n_chunks = (args.deals + args.chunk - 1) // args.chunk
    for k in range(n_chunks):
        deals = min(args.chunk, args.deals - k * args.chunk)
        run_duplicate_match(agent, opp, cfg, deals, seed=args.seed * 1000 + k, engine=engine)
        flop = sum(1 for s in agent.stats if s.get("street") == 1 and not s.get("fallback"))
        write_atomic(turn_path, store.tensors())
        log(
            f"# record chunk {k + 1}/{n_chunks}: {len(store)} turn-end states from {flop} "
            f"flop searches, {time.time() - t0:.0f}s"
        )
    turn = store.tensors()

    # 2. river states
    g = torch.Generator().manual_seed(args.seed + 7)
    states = river_states(turn, args.rivers_per_state, g)
    n = int(states["c"].shape[0])
    perm = torch.randperm(n, generator=g)
    states = {k: v[perm] for k, v in states.items()}
    states["source"] = torch.full((n,), SOURCE_ONPOLICY, dtype=torch.uint8)
    log(f"# {n} river states from {len(turn['c'])} turn-end states")

    # 3. solve, one shard at a time
    t1 = time.time()
    meta = {
        "kind": "river",
        "source": "on-policy: turn-end leaves of value-net flop searches (source 3)",
        "blueprint": args.blueprint,
        "leaf_net": args.leaf_net,
        "search_config": args.search_config,
        "args": vars(args),
        "shards": {},
    }
    for s in range(0, n, args.shard_size):
        k = s // args.shard_size
        path = out / shard_name(k)
        part = {key: v[s : s + args.shard_size] for key, v in states.items()}
        # batches at their median c: on-policy states keep the pots the search had
        rows = solve_states(part, bp.spec, cfg, args.iterations, args.batch, dev)
        write_atomic(path, rows)
        ex = rows["exploit"].float()
        meta["shards"][path.name] = {
            "samples": int(ex.numel()),
            "exploit_mean": float(ex.mean()),
            "exploit_max": float(ex.max()),
        }
        write_atomic(out / "meta.json", meta, as_json=True)
        log(
            f"# {path.name}: {ex.numel()} samples, exploit/pot mean {float(ex.mean()):.4f} "
            f"max {float(ex.max()):.4f}, {time.time() - t1:.0f}s"
        )
    (out / "summary.json").write_text(
        json.dumps({"turn_states": int(len(turn["c"])), "river_states": n}, indent=1)
    )
    log("# done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
