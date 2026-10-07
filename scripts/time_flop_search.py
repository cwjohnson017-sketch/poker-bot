#!/usr/bin/env python
r"""Time production flop search decisions on the evaluation spots.

Runs ``SearchAgent.act`` with a search config (default
configs/search_value_net.yaml) and per-variant overrides at the flop spots of
scripts/eval_search_exploit.py, and reports the agent's own per-decision
stats: total seconds, gadget terminate seconds, setup, solve, DCFR
iterations, tree nodes and value leaves.

python scripts/time_flop_search.py --blueprint runs/dcfr4_distilled_v2 \
    --variant rollouts='{"gadget": {"terminate": "rollouts"}}' \
    --variant br='{"gadget": {"terminate": "blueprint_br"}}' --repeats 2

A warm-up decision on another board runs first, so CUDA start-up and the
one-time preflop tables stay out of the numbers. Each (variant, spot) runs
``--repeats`` times with a fresh agent (the first run of a spot is cold for
the blueprint's caches, as in play).
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
from pokerbot.search.spot_eval import SPOT_TYPES, exploit_spots, run_search  # noqa: E402

KEYS = (
    "total_seconds",
    "terminate_seconds",
    "setup_seconds",
    "solve_seconds",
    "iterations",
    "nodes",
    "value_leaves",
)


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--blueprint", required=True, help="Deep CFR run dir or blueprint spec")
    ap.add_argument("--config", default=str(ROOT / "configs" / "search_value_net.yaml"))
    ap.add_argument("--value-net", help="leaf.net override (default: the config's)")
    ap.add_argument(
        "--variant",
        action="append",
        default=[],
        metavar="NAME=JSON",
        help="search config overrides of one variant (repeat); none: the config as is",
    )
    ap.add_argument("--search", type=json.loads, default={}, help="overrides for every variant")
    ap.add_argument("--boards", type=int, default=2)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--spot-types", nargs="+", choices=SPOT_TYPES, default=list(SPOT_TYPES))
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-json", help="write every decision's stats here")
    args = ap.parse_args(argv)

    variants: dict[str, dict] = {}
    for item in args.variant or ["config={}"]:
        name, sep, js = item.partition("=")
        if not sep or not name or name in variants:
            ap.error(f"--variant {item!r}: expected a new NAME=JSON")
        variants[name] = json.loads(js)
    spec = args.blueprint
    if spec.partition(":")[0] not in BLUEPRINTS:
        spec = f"neural:{spec}"
    dev = torch.device(args.device)
    engine = get_engine()
    cfg = engine.GameConfig(num_players=2, stacks=[10000, 10000], small_blind=50, big_blind=100)
    bp_kwargs = {"device": str(dev)} if spec.startswith("neural:") else {}
    bp = make_blueprint(spec, **bp_kwargs)
    base = {"device": str(dev), "fallback_on_error": False, **args.search}
    if args.value_net:
        base["leaf"] = {**base.get("leaf", {}), "net": str(Path(args.value_net).resolve())}

    def config(over: dict):
        merged = dict(base)
        for k, v in over.items():
            merged[k] = {**merged[k], **v} if isinstance(v, dict) and k in merged else v
        return search_config(args.config, **merged)

    warm = exploit_spots(engine, cfg, 1, args.seed + 1000, ["bb_first"])[0]
    t0 = time.perf_counter()
    run_search("warmup", bp, warm.state, cfg, config(next(iter(variants.values()))))
    log(f"warm-up {time.perf_counter() - t0:.1f}s")
    spots = exploit_spots(engine, cfg, args.boards, args.seed, args.spot_types)
    rows = []
    for name, over in variants.items():
        c = config(over)
        for spot in spots:
            for r in range(args.repeats):
                run = run_search(name, bp, spot.state, cfg, c)
                st = {k: run.stats.get(k) for k in KEYS}
                st.update(variant=name, spot=spot.label, repeat=r, gadget=run.stats.get("gadget"))
                rows.append(st)
                log(
                    f"{name:>12} {spot.label:<24} {st['total_seconds']:.2f}s "
                    f"(T {st['terminate_seconds']:.2f}s, setup {st['setup_seconds']:.2f}s, "
                    f"solve {st['solve_seconds']:.2f}s) {st['iterations']} it, "
                    f"{st['nodes']} nodes, {st['value_leaves']} leaves, gadget {st['gadget']}"
                )
                del run
                if dev.type == "cuda":
                    torch.cuda.empty_cache()
    print("\n| variant | decisions | total s | terminate s | setup s | solve s | iterations |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for name in variants:
        xs = [x for x in rows if x["variant"] == name]

        def mean(k: str, xs: list = xs) -> float:
            return sum(float(x[k] or 0) for x in xs) / len(xs)

        print(
            f"| {name} | {len(xs)} | {mean('total_seconds'):.2f} | "
            f"{mean('terminate_seconds'):.2f} | {mean('setup_seconds'):.2f} | "
            f"{mean('solve_seconds'):.2f} | {mean('iterations'):.0f} |"
        )
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
