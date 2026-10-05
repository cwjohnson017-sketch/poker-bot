#!/usr/bin/env python
"""Turn-end leaf values against exact targets.

A flop search's value-net leaves sit at the end of turn betting. For fresh
turn-end states (blueprint self-play, perturbed and random ranges, as in the
training data but with other seeds) every one of the 48 river subgames is
solved exactly with ``BatchRiverSolver``, and each player's best-response
values at the river roots are chance-averaged to the turn end (weight 1/44,
the solver's chance identity). That is the target both leaf models
approximate:

* the river net averaged over the river cards (``value_leaf.river_average``,
  what ``ValueLeafEvaluator`` computes in search);
* the turn-end net (one row per leaf, ``TurnEndLeafEvaluator``).

Errors are in pot units per unit of disjoint opponent mass (the nets' output
scale): MAE over combos, range-weighted MAE, and each player's game-value
error, overall, per source and per pot bin.

python scripts/check_turn_leaves.py --river-net runs/value_net/river_v1.pt \
    --turn-net runs/value_net/turn_v1.pt --blueprint runs/dcfr4_distilled_v2 --states 320
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pokerbot.engine_select import get_engine  # noqa: E402
from pokerbot.search.batch_solver import BatchRiverSolver, river_tree  # noqa: E402
from pokerbot.search.blueprint import make_blueprint  # noqa: E402
from pokerbot.search.combos import NUM_COMBOS, avoids_card, blocked_sum, valid_masks  # noqa: E402
from pokerbot.search.turn_data import make_turn_states  # noqa: E402
from pokerbot.search.turn_net import load_leaf_predictor  # noqa: E402
from pokerbot.search.value_leaf import river_average  # noqa: E402
from pokerbot.search.value_ranges import reachable_c  # noqa: E402

C = NUM_COMBOS
SOURCES = {0: "self-play", 1: "perturbed", 2: "random"}
POT_BINS = (100, 250, 500, 1000, 2000, 4000, 10001)


def exact_turn_values(states: dict, spec, cfg, iterations: int, per_batch: int, dev, log):
    """``[n, 2, C]`` exact turn-end values (chips, (OOP, IP)) and the ``c`` each
    state was solved at (states are batched by pot; ranges are inputs only)."""
    n = states["c"].shape[0]
    order = torch.argsort(states["c"], stable=True)
    out = torch.zeros(n, 2, C, device=dev)
    c_used = states["c"].clone()
    avoid = avoids_card(dev).float()
    t0 = time.time()
    for lo in range(0, n, per_batch):
        idx = order[lo : lo + per_batch]
        med = int(states["c"][idx].float().median().round())
        c = int(reachable_c(torch.tensor(med)))
        c = min(c, int(cfg.stacks[0]) - 1)
        c_used[idx] = c
        rows, boards, ranges = [], [], []
        for i in idx.tolist():
            b4 = states["boards"][i].tolist()
            r = states["ranges"][i].to(dev)
            for x in range(52):
                if x in b4:
                    continue
                rows.append(i)
                boards.append(b4 + [x])
                ranges.append(r * avoid[x])
        rt = river_tree(cfg, c, spec, button=1, device=dev)  # seat 0 = OOP
        bs = BatchRiverSolver(rt, torch.tensor(boards, device=dev), torch.stack(ranges), device=dev)
        bs.solve(iterations)
        sig = bs.average_strategy()
        ri = torch.tensor(rows, device=dev)
        for p in (0, 1):
            v = bs.root_values(p, sig, best_response=True)
            out[:, p].index_add_(0, ri, v / 44.0)
        del bs, sig
        log(f"# exact: {min(lo + per_batch, n)}/{n} states, c={c}, {time.time() - t0:.0f}s")
    return out, c_used


def to_ev(values: torch.Tensor, ranges: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Chips ``[n, 2, C]`` -> pot units per unit of disjoint opponent mass."""
    m = torch.stack([blocked_sum(ranges[:, 1]), blocked_sum(ranges[:, 0])], 1)
    pot = (2 * c).float()[:, None, None]
    return torch.where(m > 1e-9, values / (m.clamp(min=1e-30) * pot), 0.0)


def metrics(pred: torch.Tensor, exact: torch.Tensor, ranges: torch.Tensor, mask: torch.Tensor):
    """Per-sample sums for MAE, range-weighted MAE and game-value error."""
    err = (pred - exact) * mask
    m = torch.stack([blocked_sum(ranges[:, 1]), blocked_sum(ranges[:, 0])], 1)
    w = ranges * m * mask
    z = w.sum(-1).clamp(min=1e-12)  # [n, 2]
    return {
        "abs": err.abs().sum((1, 2)),
        "cnt": mask.sum((1, 2)),
        "wabs": (w * err.abs()).sum(-1) / z,  # [n, 2]
        "gv": (w * err).sum(-1).abs() / z,  # [n, 2]
        "zero": (exact * mask).abs().sum((1, 2)),
    }


def summarise(st: dict, sel: torch.Tensor) -> dict:
    if not bool(sel.any()):
        return {"samples": 0}
    cnt = float(st["cnt"][sel].sum())
    return {
        "samples": int(sel.sum()),
        "mae": float(st["abs"][sel].sum()) / cnt,
        "wmae": float(st["wabs"][sel].mean()),
        "gv_mae": float(st["gv"][sel].mean()),
        "zero_mae": float(st["zero"][sel].sum()) / cnt,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--river-net", required=True)
    ap.add_argument("--turn-net")
    ap.add_argument("--blueprint", required=True)
    ap.add_argument("--states", type=int, default=320)
    ap.add_argument("--mix", default="0.5,0.25,0.25")
    ap.add_argument("--iterations", type=int, default=400)
    ap.add_argument("--per-batch", type=int, default=10, help="states per river batch (x48)")
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="runs/value_net/turn_leaf_check.json")
    args = ap.parse_args(argv)

    def log(msg: str) -> None:
        print(msg, flush=True)

    dev = torch.device(args.device)
    engine = get_engine()
    cfg = engine.GameConfig(num_players=2, stacks=[10000, 10000], small_blind=50, big_blind=100)
    bp = make_blueprint(f"neural:{args.blueprint}", device=str(dev))
    mix = tuple(float(x) for x in args.mix.split(","))
    seeds = [args.seed, args.seed + 1, args.seed + 2]
    states = make_turn_states(bp, cfg, args.states, mix, seeds, dev)
    states = {k: v.to(dev) for k, v in states.items()}
    valid = valid_masks(states["boards"], dev).float()
    r = states["ranges"] * valid[:, None, :]
    states["ranges"] = r / r.sum(-1, keepdim=True).clamp(min=1e-30)
    exact_v, c_used = exact_turn_values(
        states, bp.spec, cfg, args.iterations, args.per_batch, dev, log
    )
    states["c"] = c_used
    states["stack"] = int(cfg.stacks[0]) - c_used
    ranges = states["ranges"]
    exact = to_ev(exact_v, ranges, c_used)
    m = torch.stack([blocked_sum(ranges[:, 1]), blocked_sum(ranges[:, 0])], 1)
    mask = (valid[:, None, :] > 0) & (m > 1e-4)
    mask = mask.float()

    preds = {}
    river = load_leaf_predictor(args.river_net, dev)
    rv = river_average(river, states["boards"], c_used, states["stack"], ranges.transpose(0, 1))
    preds["river net x 44 rivers"] = to_ev(rv.transpose(0, 1), ranges, c_used)
    if args.turn_net:
        turn = load_leaf_predictor(args.turn_net, dev)
        preds["turn-end net"] = turn.predict(states["boards"], ranges, c_used, states["stack"])
    src = states["source"].cpu()
    cb = torch.bucketize(c_used.cpu(), torch.tensor(POT_BINS[1:]), right=True)
    report: dict = {"states": args.states, "iterations": args.iterations, "models": {}}
    for name, pred in preds.items():
        st = {k: v.cpu() for k, v in metrics(pred.float(), exact, ranges, mask).items()}
        rep = {"overall": summarise(st, torch.ones(len(src), dtype=torch.bool))}
        rep["by_source"] = {SOURCES[s]: summarise(st, src == s) for s in SOURCES}
        rep["by_pot"] = {
            f"[{POT_BINS[i]},{POT_BINS[i + 1]})": summarise(st, cb == i)
            for i in range(len(POT_BINS) - 1)
        }
        report["models"][name] = rep
        o = rep["overall"]
        log(
            f"{name}: MAE {o['mae']:.4f} wMAE {o['wmae']:.4f} game value {o['gv_mae']:.4f} "
            f"(zero {o['zero_mae']:.4f}) pot units, {o['samples']} states"
        )
        for k, s in rep["by_source"].items():
            if s["samples"]:
                log(f"   {k:>10}: MAE {s['mae']:.4f} wMAE {s['wmae']:.4f} gv {s['gv_mae']:.4f}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))
    log(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
