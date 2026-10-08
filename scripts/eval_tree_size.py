#!/usr/bin/env python
r"""Flop tree size under a fixed time budget, scored with an exact river.

For every flop spot: production value-net searches (configs/search_value_net.yaml)
at several tree.max_nodes under the same flop time budget, so bigger trees get
fewer DCFR iterations. Each search's strategy is translated onto the largest
search's tree and scored there with every (leaf, river card) river subgame
solved exactly, next to the blueprint (see pokerbot/search/size_eval.py).

python scripts/eval_tree_size.py --blueprint runs/dcfr4_distilled_v2 \
    --value-net runs/value_net/turn_v2w.pt --spots 0:bb_first 0:btn_vs_check 1:bb_first \
    --sizes 6000 10000 20000 --budget 4 --river-iters 200 \
    --out-json runs/tree_size/size.json --out-md runs/tree_size/size.md

python scripts/eval_tree_size.py --blueprint runs/dcfr4_distilled_v2 --oracle showdown \
    --spots 0:bb_first --sizes 1000 2000 --river-iters 20 --eval-runouts 0 --device cpu \
    --out-json runs/tree_size/smoke.json --out-md runs/tree_size/smoke.md     # smoke test

Spots: --spots <board>:<type> ... picks spots by board index and type
(bb_first, btn_vs_check, btn_vs_lead); without it, --boards x --spot-types.
--iters N adds a fixed-N-iteration search per size as a reference.
Each smaller budget search is also scored re-searched, as the agent plays: the same
agent searches again wherever the opponent takes a flop size its tree lacks
(--no-research: translated only).
--search '{"gadget": {"safe": false}}' overrides the config for every search.

--oracle showdown values every leaf as a checked-down river; --oracle untrained
is an untrained river value net (each call costs what a trained net's does).

The JSON (and the markdown) are rewritten after every spot; --resume skips the
spots an existing JSON with the same settings already has. A failing spot is
recorded and skipped (exit code 1 at the end).
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from pokerbot.engine_select import get_engine  # noqa: E402
from pokerbot.search.blueprint import BLUEPRINTS, make_blueprint  # noqa: E402
from pokerbot.search.size_eval import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_SIZES,
    SizeSettings,
    config_path,
    render_markdown,
    run_size_evaluation,
    select_spots,
)
from pokerbot.search.spot_eval import EXACT_FLOP_RUNOUTS, SPOT_TYPES, exploit_spots  # noqa: E402
from pokerbot.search.value_leaf import ShowdownOracle  # noqa: E402


def _untrained(device: torch.device):
    from pokerbot.search.value_net import RiverValueNet, ValueNetConfig, ValueNetPredictor

    torch.manual_seed(0)
    return ValueNetPredictor(RiverValueNet(ValueNetConfig()), device)


ORACLES = {"showdown": lambda device: ShowdownOracle(), "untrained": _untrained}


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def _blueprint_spec(arg: str) -> str:
    """``neural:<dir>``, ``blueprint:<file>``, ``uniform``, or a bare Deep CFR run dir."""
    prefix = arg.partition(":")[0]
    return arg if prefix in BLUEPRINTS else f"neural:{arg}"


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True
        )
        return out.stdout.strip() or None
    except OSError:
        return None


def main(argv: list[str] | None = None) -> int:
    d = SizeSettings()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument(
        "--blueprint",
        required=True,
        help="Deep CFR run dir (e.g. runs/dcfr4_distilled_v2) or a search blueprint spec",
    )
    leaf = ap.add_mutually_exclusive_group(required=True)
    leaf.add_argument("--value-net", help="leaf net checkpoint (loaded through leaf.net)")
    leaf.add_argument("--oracle", choices=sorted(ORACLES), help="a test predictor instead")
    ap.add_argument("--config", default=DEFAULT_CONFIG, help="base search config")
    ap.add_argument("--boards", type=int, default=1, help="boards (without --spots)")
    ap.add_argument("--seed", type=int, default=5, help="board generator seed")
    ap.add_argument("--spot-types", nargs="+", choices=SPOT_TYPES, default=["bb_first"])
    ap.add_argument("--spots", nargs="+", help="<board>:<type> picks, e.g. 0:bb_first 1:bb_first")
    ap.add_argument("--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES))
    ap.add_argument("--budget", type=float, default=d.budget, help="flop seconds per decision")
    ap.add_argument("--iters", type=int, help="also a fixed-iteration search per size")
    ap.add_argument("--river-iters", type=int, default=d.river_iters)
    ap.add_argument("--mix", type=float, default=d.mix, help="uniform mixed into river ranges")
    ap.add_argument("--min-mass", type=float, default=d.min_mass)
    ap.add_argument("--river-batch", type=int, default=d.river_batch)
    ap.add_argument(
        "--eval-runouts",
        type=int,
        default=EXACT_FLOP_RUNOUTS,
        help=f"all-in run-outs per board when scoring (default {EXACT_FLOP_RUNOUTS}: every "
        "flop run-out; 0: the search's own solver.max_runouts)",
    )
    ap.add_argument(
        "--search",
        type=json.loads,
        default={},
        help='extra search config overrides as JSON, e.g. \'{"gadget": {"safe": false}}\'',
    )
    ap.add_argument("--strict", action="store_true", help="fail on mass lost in translation")
    ap.add_argument(
        "--no-research",
        action="store_true",
        help="score translated strategies only (no re-search at off-tree opponent actions)",
    )
    ap.add_argument("--no-spot-warmup", action="store_true", help="no per-spot cache warm-up")
    ap.add_argument("--device", default="auto", help="auto | cuda | cpu")
    ap.add_argument("--out-md", default="runs/tree_size/size.md")
    ap.add_argument("--out-json", default="runs/tree_size/size.json")
    ap.add_argument("--resume", action="store_true", help="keep the spots of --out-json")
    ap.add_argument("--no-warmup", action="store_true", help="skip the warm-up search")
    ap.add_argument(
        "--render-only", action="store_true", help="only rewrite --out-md from --out-json"
    )
    args = ap.parse_args(argv)

    if args.render_only:
        data = json.loads(Path(args.out_json).read_text())
        Path(args.out_md).write_text(render_markdown(data), encoding="utf-8")
        log(f"wrote {args.out_md}")
        return 0
    settings = SizeSettings(
        sizes=tuple(args.sizes),
        budget=args.budget,
        iters=args.iters,
        config=args.config,
        river_iters=args.river_iters,
        mix=args.mix,
        min_mass=args.min_mass,
        river_batch=args.river_batch,
        eval_max_runouts=args.eval_runouts or None,
        strict=args.strict,
        research=not args.no_research,
        spot_warmup=not args.no_spot_warmup,
        device=args.device,
        search=args.search,
    )
    config_path(settings.config)  # fail fast
    dev = settings.torch_device()
    engine = get_engine()
    cfg = engine.GameConfig(num_players=2, stacks=[10000, 10000], small_blind=50, big_blind=100)
    spec = _blueprint_spec(args.blueprint)
    if args.oracle:
        net_path = None
        factory = lambda: ORACLES[args.oracle](dev)  # noqa: E731 - a fresh one per search
        leaf_model = f"oracle:{args.oracle}"
    else:
        if not Path(args.value_net).exists():
            ap.error(f"--value-net {args.value_net} does not exist")
        net_path = str(Path(args.value_net).resolve())
        factory = lambda: None  # noqa: E731 - every agent loads leaf.net itself
        leaf_model = net_path
    log(f"# eval_tree_size: {settings.comparable()}")
    log(f"# blueprint {spec}, leaf model {leaf_model}, {dev}")
    bp_kwargs = {"device": str(dev)} if spec.startswith("neural:") else {}
    bp = make_blueprint(spec, **bp_kwargs)
    spots = select_spots(engine, cfg, args.boards, args.seed, args.spot_types, args.spots)
    warm = None
    if not args.no_warmup:  # a board of its own, so no spot's blueprint queries are cached
        warm = exploit_spots(engine, cfg, 1, args.seed + 1000, ["bb_first"])[0]
    meta = {
        "blueprint": spec,
        "leaf_model": leaf_model,
        "device": str(dev),
        "gpu": torch.cuda.get_device_name(dev) if dev.type == "cuda" else None,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "commit": _git_commit(),
        "command": " ".join(sys.argv),
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "board_seed": args.seed,
    }
    log(f"# {len(spots)} spots: {', '.join(s.label for s in spots)}")
    t0 = time.perf_counter()
    data = run_size_evaluation(
        spots,
        bp,
        cfg,
        settings,
        meta,
        args.out_json,
        args.out_md,
        predictor_factory=factory,
        net_path=net_path,
        resume=args.resume,
        warmup=warm,
        log=log,
    )
    log(f"# done in {(time.perf_counter() - t0) / 60:.1f} min: {args.out_md}, {args.out_json}")
    print(render_markdown(data), flush=True)
    return 1 if data["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
