#!/usr/bin/env python
r"""Flop decisions of a depth-0 search (turn-start net leaves) against the production
depth-1 search, scored exactly on the production search's own tree.

Per spot:

1. the production search (``--config``, default configs/search_value_net.yaml: 20,000-node
   flop trees, turn-end leaves, 4 s) decides; its tree is the scoring trunk (the
   blueprint's whole flop abstraction, turn to the end of betting);
2. the depth-0 search (``--depth0-config``, default configs/search_turn_start.yaml:
   flop betting only, leaves valued by the turn-start net over the 49 turn cards)
   decides on the same spot;
3. **hybrid** = the depth-0 search's strategy at every flop decision of the trunk
   (matched exactly by history: both trees have the full flop abstraction) and the
   production search's at every turn decision. In play, turn decisions are
   re-solved anyway; this isolates the flop decisions;
4. both profiles are scored with the exact river evaluation
   (``spot_eval.score_profile``): one-sided (opponent's BR against the searcher)
   and two-sided, in mbb/hand.

python scripts/eval_depth0.py --blueprint runs/dcfr4_distilled_v2 \
    --spots 0:bb_first 0:btn_vs_check 1:bb_first --out-json runs/tree_size/depth0.json
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
from pokerbot.search.blueprint import BLUEPRINTS, make_blueprint  # noqa: E402
from pokerbot.search.config import search_config  # noqa: E402
from pokerbot.search.size_eval import SizeSettings, select_spots  # noqa: E402
from pokerbot.search.spot_eval import (  # noqa: E402
    _free,
    _score_with_retry,
    _self_exploitability,
    run_search,
    scoring_solver,
)
from pokerbot.search.tree import DECISION  # noqa: E402
from pokerbot.search.tree_map import StrategySnapshot, match_nodes, translate_sigma  # noqa: E402


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--blueprint", required=True)
    ap.add_argument("--config", default=str(ROOT / "configs" / "search_value_net.yaml"))
    ap.add_argument("--depth0-config", default=str(ROOT / "configs" / "search_turn_start.yaml"))
    ap.add_argument("--spots", nargs="+", default=["0:bb_first", "0:btn_vs_check", "1:bb_first"])
    ap.add_argument("--river-iters", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-json", default="runs/tree_size/depth0.json")
    args = ap.parse_args(argv)

    spec = args.blueprint
    if spec.partition(":")[0] not in BLUEPRINTS:
        spec = f"neural:{spec}"
    dev = torch.device(args.device)
    engine = get_engine()
    gcfg = engine.GameConfig(num_players=2, stacks=[10000, 10000], small_blind=50, big_blind=100)
    bp = make_blueprint(spec, device=str(dev))
    settings = SizeSettings(device=str(dev), river_iters=args.river_iters)
    over = {"device": str(dev), "fallback_on_error": False}
    prod_cfg = lambda: search_config(args.config, **over)  # noqa: E731
    d0_cfg = lambda: search_config(args.depth0_config, **over)  # noqa: E731
    boards = 1 + max(int(s.partition(":")[0]) for s in args.spots)
    spots = select_spots(engine, gcfg, boards, 5, picks=args.spots)
    warm = select_spots(engine, gcfg, 1, 1005, ["bb_first"])[0]
    run_search("warmup", bp, warm.state, gcfg, prod_cfg())
    run_search("warmup0", bp, warm.state, gcfg, d0_cfg())
    _free()
    out = {"args": vars(args), "spots": []}
    mbb = 1000.0 / float(gcfg.big_blind)
    for spot in spots:
        t_spot = time.perf_counter()
        state = spot.state
        seat = int(state.current_player)
        log(f"{spot.label}: seat {seat} to act")
        runs = {}
        for name, cfg in (("depth1", prod_cfg()), ("depth0", d0_cfg())):
            run = run_search(name, bp, state, gcfg, cfg)
            st = run.stats
            runs[name] = {
                "snap": StrategySnapshot.of(run.solver, "cpu"),
                "stats": {
                    k: st.get(k)
                    for k in ("iterations", "nodes", "value_leaves", "total_seconds", "gadget")
                },
                "self_exploitability": _self_exploitability(run, gcfg),
            }
            log(f"  {name}: {runs[name]['stats']}, own game {runs[name]['self_exploitability']}")
            if name == "depth1":
                eval_solver = scoring_solver(run.solver, settings.eval_max_runouts)
            del run
            _free()
        trunk = runs["depth1"]["snap"]
        tree = eval_solver.tree
        base = trunk.sigma.to(eval_solver.device)
        flop = [
            int(n)
            for n in eval_solver.dec_nodes.tolist()
            if int(tree.kind[n]) == DECISION and int(tree.street[n]) == 1
        ]
        d0 = runs["depth0"]["snap"]
        match = match_nodes(d0.tree, tree)
        hybrid, rep = translate_sigma(
            d0, eval_solver, seat, strict=True, fallback=base, match=match, nodes=flop
        )
        log(
            f"  hybrid: {len(flop)} flop decision nodes, translated {rep['translated_nodes']}, "
            f"unmatched {rep['unmatched_nodes']}, lost {rep['lost_mass']}"
        )
        if sum(rep["unmatched_nodes"].values()) or sum(rep["translated_nodes"].values()):
            raise RuntimeError("the depth-0 tree's flop does not match the trunk's exactly")
        profiles = {}
        batch = [settings.river_batch]
        for name, sigma in (("depth1", base), ("hybrid_depth0_flop", hybrid)):
            res = _score_with_retry(batch, log, eval_solver, sigma, bp.spec, gcfg, settings)
            res["one_sided_mbb"] = res["br"][1 - seat] * mbb
            profiles[name] = res
            log(f"  {name}: one-sided {res['one_sided_mbb']:.0f}, two-sided {res['mbb']:.0f}")
        out["spots"].append(
            {
                "label": spot.label,
                "seat": seat,
                "flop_decision_nodes": len(flop),
                "searches": {
                    k: {kk: vv for kk, vv in v.items() if kk != "snap"} for k, v in runs.items()
                },
                "hybrid_report": rep,
                "profiles": profiles,
                "seconds": time.perf_counter() - t_spot,
            }
        )
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(out, indent=1, default=float))
        del eval_solver, base, hybrid
        _free()
    print(
        "\n| spot | depth-1 one-sided | depth-0 flop one-sided "
        "| depth-1 two-sided | depth-0 flop two-sided |"
    )
    print("|---|---:|---:|---:|---:|")
    for s in out["spots"]:
        p = s["profiles"]
        a, b = p["depth1"], p["hybrid_depth0_flop"]
        print(
            f"| {s['label']} | {a['one_sided_mbb']:.0f} | {b['one_sided_mbb']:.0f} | "
            f"{a['mbb']:.0f} | {b['mbb']:.0f} |"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
