#!/usr/bin/env python
r"""Solve the preflop against the postflop the search plays (``pokerbot.search.preflop_solve``).

Round r (``--out-dir/round_R``), starting from the blueprint's preflop (symmetrized
over suits) in round 0:

1. **Flop solves.** At each of the 31 flop starts and ``--flops`` flops (seed
   ``--seed + r``: new flops every round), solve the flop root from the current
   preflop's ranges (production's flop search settings, ``--iterations``) and
   linearise it: root values at the current ranges, and class values ``W``.
   Saved after every solve; a rerun resumes.
2. **Evaluation** of the current preflop on these flops
   (``preflop_eval.evaluate``: best responses in-sample and cross-fitted). From
   round 1 on, the preflop was solved on the previous round's flops, so this is
   out of sample.
3. **Preflop CFR** (``--cfr-iterations``) on the linearised game. The next
   preflop is ``(1 - mix) * current + mix * solution`` (``--mix``), written as
   ``round_R/next.pt`` and as a production table ``round_R/table.pt``
   (``search:preflop:<table.pt>``).

python scripts/solve_preflop.py --blueprint runs/dcfr4_distilled_v2 --rounds 4 --flops 50 \
    --out-dir runs/preflop/solve1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from pokerbot.engine_select import get_engine  # noqa: E402
from pokerbot.search.agent import SearchAgent  # noqa: E402
from pokerbot.search.blueprint import BLUEPRINTS, make_blueprint  # noqa: E402
from pokerbot.search.config import search_config  # noqa: E402
from pokerbot.search.preflop_eval import (  # noqa: E402
    FlopValues,
    evaluate,
    preflop_allin_matrix,
    preflop_tree,
    sample_flops,
    suit_perms,
)
from pokerbot.search.preflop_solve import (  # noqa: E402
    ClassValues,
    PreflopGame,
    exploitability,
    node_policies,
    save_table,
    solve_flop_linear,
    solve_preflop,
    table_from_strategy,
    with_strategy,
)


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def free_gpu(agent) -> None:
    """Drop the leaf nets' board-feature caches (up to 32k boards each) and cached
    blocks: a 16-round run ran out of GPU memory after about 2,500 flop solves."""
    for pred in (agent.value_predictor, agent.turn_value_predictor):
        cache = getattr(pred, "cache", None)
        if cache is not None:
            cache.clear()
    torch.cuda.empty_cache()


def flop_round(args, rdir: Path, nodes, flops, agent, gcfg, engine, perms) -> tuple[dict, dict]:
    """Solve and linearise every flop start; ``({flop_id: FlopValues}, {flop_id: W mean})``."""
    starts = [n for n in nodes if n.kind == "flop"]
    fv: dict[int, FlopValues] = {}
    W: dict[int, torch.Tensor] = {}
    for n in starts:
        path = rdir / f"flop_{n.flop_id:02d}.pt"
        d = torch.load(path) if path.exists() else {"line": n.line(), "values": {}, "W": None}
        cv = ClassValues(perms)
        if d["W"] is not None:
            cv.load(d["W"])
        todo = [f for f in flops if f not in d["values"]]
        if todo:
            log(f"flop start {n.flop_id} `{n.line()}`: {len(todo)} flops")
        for f in todo:
            r = solve_flop_linear(agent, gcfg, engine, n.history, f, n.reach, args.iterations)
            cv.add(f, r["W"])
            d["values"][f] = r["values"]
            d["W"] = cv.state_dict()
            torch.save(d, path)
        if todo:
            log(f"  done, {r['seconds']:.1f} + {r['linear_seconds']:.1f} s per flop (last)")
            free_gpu(agent)
        v = FlopValues(perms)
        for f in flops:
            v.add(flops.index(f) % 2, f, d["values"][f])
        fv[n.flop_id] = v
        W[n.flop_id] = cv.mean()
    return fv, W


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--blueprint", required=True)
    ap.add_argument("--config", default="configs/search_value_net.yaml")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--flops", type=int, default=50)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--iterations", type=int, default=100)
    ap.add_argument("--cfr-iterations", type=int, default=2000)
    ap.add_argument("--mix", type=float, default=0.5)
    ap.add_argument("--allin-runouts", type=int, default=2000)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=1))
    spec = (
        args.blueprint
        if args.blueprint.partition(":")[0] in BLUEPRINTS
        else f"neural:{args.blueprint}"
    )
    dev = torch.device(args.device)
    engine = get_engine()
    gcfg = engine.GameConfig(num_players=2, stacks=[10000, 10000], small_blind=50, big_blind=100)
    bp = make_blueprint(spec, device=str(dev))
    base_nodes = preflop_tree(bp, gcfg, engine, dev)
    agent = SearchAgent(bp, search_config(args.config, device=str(dev), fallback_on_error=False))
    agent._check_leaf_nets()
    perms = suit_perms()
    E = preflop_allin_matrix(args.allin_runouts)
    sigma = node_policies(base_nodes)
    summary = []
    for r in range(args.rounds):
        rdir = out / f"round_{r}"
        rdir.mkdir(exist_ok=True)
        if r > 0:
            sigma = torch.load(out / f"round_{r - 1}" / "next.pt")
        nodes = with_strategy(base_nodes, sigma)
        flops = sample_flops(args.flops, args.seed + r)
        log(f"round {r}: {args.flops} flops, seed {args.seed + r}")
        fv, W = flop_round(args, rdir, nodes, flops, agent, gcfg, engine, perms)
        res = evaluate(nodes, fv, E, int(gcfg.big_blind))
        game = PreflopGame(nodes, W, E)
        lin = exploitability(game, sigma)
        log(
            f"round {r}: exploitability {res['exploitability_cross_fit']:.1f} (cross-fit) - "
            f"{res['exploitability_in_sample']:.1f} (in-sample) mbb/hand; linearised "
            f"{lin['exploitability']:.1f}"
        )
        sol, trace = solve_preflop(
            game, args.cfr_iterations, log_every=args.cfr_iterations // 10, log=log
        )
        nxt = {i: (1 - args.mix) * sigma[i] + args.mix * sol[i].cpu() for i in sigma}
        lin_next = exploitability(game, nxt)
        torch.save(nxt, rdir / "next.pt")
        torch.save(sol, rdir / "solution.pt")
        save_table(
            rdir / "table.pt",
            spec,
            table_from_strategy(base_nodes, nxt, bp.spec.num_actions),
            {"round": r, "args": vars(args)},
        )
        row = {
            "round": r,
            "seed": args.seed + r,
            "exploitability_in_sample": res["exploitability_in_sample"],
            "exploitability_cross_fit": res["exploitability_cross_fit"],
            "zero_sum_check": res["zero_sum_check"],
            "seat_values": [s["blueprint"] for s in res["seats"]],
            "linearised": lin,
            "cfr_trace": trace,
            "linearised_next": lin_next,
        }
        res["args"] = vars(args)
        (rdir / "eval.json").write_text(json.dumps(res, indent=1))
        summary.append(row)
        (out / "summary.json").write_text(json.dumps(summary, indent=1))
        log(
            f"round {r}: CFR solution {trace[-1]['exploitability']:.2f}, next preflop "
            f"{lin_next['exploitability']:.1f} mbb/hand in the linearised game"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
