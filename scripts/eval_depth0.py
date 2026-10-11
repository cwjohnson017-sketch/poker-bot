#!/usr/bin/env python
r"""Flop decisions of a depth-0 search (turn-start net leaves) against the production
depth-1 search, scored exactly on the production search's own tree.

Per spot:

1. the production search (``--config``, default configs/search_value_net.yaml: 20,000-node
   flop trees, turn-end leaves, 4 s) decides; its tree is the scoring trunk (the
   blueprint's whole flop abstraction, turn to the end of betting);
2. each depth-0 search (``--depth0-config [NAME=]PATH``, repeatable; default
   configs/search_turn_start.yaml as ``depth0``: flop betting only, leaves valued by
   the turn-start net over the 49 turn cards) decides on the same spot;
3. **hybrid** = a depth-0 search's strategy at every flop decision of the trunk
   (matched exactly by history: both trees have the full flop abstraction) and the
   production search's at every turn decision (profile ``hybrid_<name>_flop``);
4. with ``--resolve-iters N``, **re-solved** profiles (``<profile>_resolved``): every flop
   decision of both players locked to the profile's, every turn decision solved again
   on the trunk with the production search's own leaf net for N iterations
   (``size_eval.resolve_below_root``; ``--resolve-mix`` uniform mixed into the flop
   locks while solving). The hybrid plays a turn strategy solved for the production
   search's flop ranges, which charges a depth-0 flop strategy for a mismatch; the
   re-solved profiles differ only in their flop strategies. ``--no-plain`` scores only
   the re-solved ones;
5. every profile is scored with the exact river evaluation
   (``spot_eval.score_profile``): one-sided (opponent's BR against the searcher)
   and two-sided, in mbb/hand.

python scripts/eval_depth0.py --blueprint runs/dcfr4_distilled_v2 \
    --spots 0:bb_first 0:btn_vs_check 1:bb_first --out-json runs/tree_size/depth0.json
python scripts/eval_depth0.py --blueprint runs/dcfr4_distilled_v2 \
    --depth0-config v1=runs/r4/search_turn_start_v1.yaml \
    --depth0-config v2=runs/r4/search_turn_start_v2.yaml \
    --resolve-iters 200 --no-plain --out-json runs/r4/depth0_resolved.json
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
from pokerbot.search.size_eval import (  # noqa: E402
    SizeSettings,
    resolve_below_root,
    resolved_name,
    select_spots,
)
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


def depth0_configs(items: list[str] | None) -> dict[str, str]:
    """``[NAME=]PATH`` items -> ``{name: path}``; one unnamed item (or none: the
    default config) is named ``depth0``."""
    if not items:
        return {"depth0": str(ROOT / "configs" / "search_turn_start.yaml")}
    out: dict[str, str] = {}
    for item in items:
        name, eq, path = item.partition("=")
        if not eq:
            name, path = ("depth0" if len(items) == 1 else Path(item).stem), item
        if not name or name in out or name == "depth1":
            raise SystemExit(f"--depth0-config {item!r}: names must be new and not depth1")
        out[name] = path
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--blueprint", required=True)
    ap.add_argument("--config", default=str(ROOT / "configs" / "search_value_net.yaml"))
    ap.add_argument(
        "--depth0-config",
        action="append",
        default=None,
        metavar="[NAME=]PATH",
        help="a depth-0 search config (repeatable; default configs/search_turn_start.yaml)",
    )
    ap.add_argument("--spots", nargs="+", default=["0:bb_first", "0:btn_vs_check", "1:bb_first"])
    ap.add_argument("--river-iters", type=int, default=200)
    ap.add_argument(
        "--resolve-iters", type=int, default=0, help="also re-solved profiles (0: none)"
    )
    ap.add_argument("--resolve-mix", type=float, default=0.01)
    ap.add_argument("--no-plain", action="store_true", help="score the re-solved profiles only")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-json", default="runs/tree_size/depth0.json")
    args = ap.parse_args(argv)
    if args.no_plain and not args.resolve_iters:
        ap.error("--no-plain needs --resolve-iters")
    d0_paths = depth0_configs(args.depth0_config)

    spec = args.blueprint
    if spec.partition(":")[0] not in BLUEPRINTS:
        spec = f"neural:{spec}"
    dev = torch.device(args.device)
    engine = get_engine()
    gcfg = engine.GameConfig(num_players=2, stacks=[10000, 10000], small_blind=50, big_blind=100)
    bp = make_blueprint(spec, device=str(dev))
    settings = SizeSettings(device=str(dev), river_iters=args.river_iters)
    over = {"device": str(dev), "fallback_on_error": False}
    configs = {"depth1": lambda: search_config(args.config, **over)}
    for name, path in d0_paths.items():
        configs[name] = lambda path=path: search_config(path, **over)
    boards = 1 + max(int(s.partition(":")[0]) for s in args.spots)
    spots = select_spots(engine, gcfg, boards, 5, picks=args.spots)
    warm = select_spots(engine, gcfg, 1, 1005, ["bb_first"])[0]
    for name, cfg in configs.items():
        run_search(f"warmup-{name}", bp, warm.state, gcfg, cfg())
    _free()
    out = {"args": vars(args), "depth0_configs": d0_paths, "spots": []}
    mbb = 1000.0 / float(gcfg.big_blind)
    order: list[str] = []
    for spot in spots:
        t_spot = time.perf_counter()
        state = spot.state
        seat = int(state.current_player)
        log(f"{spot.label}: seat {seat} to act")
        runs = {}
        trunk_agent = trunk_cfg = None
        for name, cfg in configs.items():
            run = run_search(name, bp, state, gcfg, cfg())
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
                trunk_agent, trunk_cfg = run.agent, run.solver.cfg
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
        sigmas = {"depth1": base}
        reports = {}
        for name in d0_paths:
            d0 = runs[name]["snap"]
            match = match_nodes(d0.tree, tree)
            hybrid, rep = translate_sigma(
                d0, eval_solver, seat, strict=True, fallback=base, match=match, nodes=flop
            )
            log(
                f"  hybrid {name}: {len(flop)} flop decision nodes, translated "
                f"{rep['translated_nodes']}, unmatched {rep['unmatched_nodes']}, "
                f"lost {rep['lost_mass']}"
            )
            if sum(rep["unmatched_nodes"].values()) or sum(rep["translated_nodes"].values()):
                raise RuntimeError(f"{name}'s flop does not match the trunk's exactly")
            sigmas[f"hybrid_{name}_flop"] = hybrid
            reports[name] = rep
        names = [] if args.no_plain else list(sigmas)
        resolve_info = {}
        if args.resolve_iters:
            for name in list(sigmas):
                sig, info = resolve_below_root(
                    eval_solver,
                    sigmas[name],
                    trunk_agent.value_leaf_provider(tree),
                    args.resolve_iters,
                    gcfg,
                    trunk_cfg,
                    args.resolve_mix,
                )
                rname = resolved_name(name)
                sigmas[rname] = sig
                resolve_info[rname] = info
                names.append(rname)
                log(
                    f"  {rname}: {info['iterations']} iterations in {info['seconds']:.1f}s, "
                    f"{info['locked_nodes']} flop decisions locked, own game "
                    f"{info['self_exploitability']['mbb']:.0f} mbb/hand"
                )
                _free()
        trunk_agent = None
        profiles = {}
        batch = [settings.river_batch]
        for name in names:
            sigma = sigmas[name].to(eval_solver.device)
            res = _score_with_retry(batch, log, eval_solver, sigma, bp.spec, gcfg, settings)
            res["one_sided_mbb"] = res["br"][1 - seat] * mbb
            profiles[name] = res
            log(f"  {name}: one-sided {res['one_sided_mbb']:.0f}, two-sided {res['mbb']:.0f}")
            del sigma
        order = names
        out["spots"].append(
            {
                "label": spot.label,
                "seat": seat,
                "flop_decision_nodes": len(flop),
                "searches": {
                    k: {kk: vv for kk, vv in v.items() if kk != "snap"} for k, v in runs.items()
                },
                "hybrid_reports": reports,
                "resolve": resolve_info,
                "profiles": profiles,
                "seconds": time.perf_counter() - t_spot,
            }
        )
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(out, indent=1, default=float))
        del eval_solver, base, sigmas
        _free()
    for key, title in (("one_sided_mbb", "one-sided"), ("mbb", "two-sided")):
        print(f"\n{title} (mbb/hand)\n\n| spot | " + " | ".join(order) + " |")
        print("|---|" + "---:|" * len(order))
        rows = [[s["profiles"][n][key] for n in order] for s in out["spots"]]
        for s, row in zip(out["spots"], rows, strict=True):
            print(f"| {s['label']} | " + " | ".join(f"{x:.0f}" for x in row) + " |")
        means = [sum(col) / len(col) for col in zip(*rows, strict=True)]
        print("| **mean** | " + " | ".join(f"{x:.0f}" for x in means) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
