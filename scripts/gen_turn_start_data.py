#!/usr/bin/env python
"""Generate turn-start value net training data: blueprint turn-root states
solved with BatchTurnSolver, whose turn-end leaves are valued by a turn-end
value net (see pokerbot/search/turn_start_data.py and docs/turn_start_net.md).

python scripts/gen_turn_start_data.py --turn-net runs/value_net/turn_v2w.pt \
    --blueprint runs/dcfr4_distilled_v2 --out data/value_turn_start --samples 200000 \
    --iterations 300 --batch 256 [--mix 0.5,0.25,0.25] [--resume]

--turn-net checkdown values the turn-end leaves by the exact check-down river
(ShowdownOracle averaged over the river cards) instead of a net (tests, smoke
runs); a river-net checkpoint is averaged over the river cards (48 rows per
leaf). The turn tree uses the blueprint's action spec (DEFAULT_SPEC without
--blueprint, which requires a mix without self-play). Writes one shard per
--shard-size samples plus meta.json (kind turn_start); --resume skips the
shards that exist (each shard is seeded from (seed, index)).
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pokerbot.search.turn_start_data import (  # noqa: E402
    TurnStartGenConfig,
    generate_turn_start_data,
)


def _floats(text: str) -> tuple[float, ...]:
    return tuple(float(x) for x in text.split(","))


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def main(argv: list[str] | None = None) -> int:
    d = TurnStartGenConfig()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--turn-net", required=True, help="turn-end value-net checkpoint, or checkdown")
    ap.add_argument("--blueprint", default=None, help="distilled Deep CFR run directory")
    ap.add_argument("--out", required=True, help="output shard directory")
    ap.add_argument("--samples", type=int, required=True)
    ap.add_argument("--iterations", type=int, default=d.iterations, help="DCFR iterations")
    ap.add_argument("--batch", type=int, default=d.batch, help="instances per solve")
    ap.add_argument("--mix", type=_floats, default=d.mix, help="self-play,perturbed,random")
    ap.add_argument("--seed", type=int, default=d.seed)
    ap.add_argument("--shard-size", type=int, default=d.shard_size)
    ap.add_argument("--explore", type=float, default=d.explore, help="self-play uniform mixing")
    ap.add_argument("--n-envs", type=int, default=d.n_envs, help="self-play lockstep slots")
    ap.add_argument("--c-range", type=_floats, default=d.c_range, help="random states' c: lo,hi")
    ap.add_argument(
        "--leaf-every", type=int, default=d.leaf_every, help="run the turn-end net every n updates"
    )
    ap.add_argument("--solver", type=json.loads, default={}, help="SolverConfig overrides (JSON)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resume", action="store_true", help="continue an existing output dir")
    args = ap.parse_args(argv)
    cfg = TurnStartGenConfig(
        samples=args.samples,
        mix=args.mix,
        iterations=args.iterations,
        batch=args.batch,
        seed=args.seed,
        shard_size=args.shard_size,
        explore=args.explore,
        n_envs=args.n_envs,
        c_range=tuple(int(x) for x in args.c_range),
        leaf_every=args.leaf_every,
        solver=args.solver,
    )
    if cfg.mix[0] + cfg.mix[1] > 0 and not args.blueprint:
        ap.error("--blueprint is required when --mix has a self-play share")
    log(f"# gen_turn_start_data: {cfg.to_dict()}, turn net {args.turn_net}, out {args.out}")
    if args.turn_net == "checkdown":
        from pokerbot.search.turn_data import RiverAveragePredictor
        from pokerbot.search.value_leaf import ShowdownOracle

        turn, turn_path = RiverAveragePredictor(ShowdownOracle(max_boards=1 << 15)), "checkdown"
    else:
        from pokerbot.search.turn_net import load_leaf_predictor

        turn = load_leaf_predictor(args.turn_net, args.device)
        if turn.kind not in ("turn_end", "river"):
            ap.error(f"--turn-net must be a turn-end (or river) net, got kind {turn.kind}")
        turn_path = str(Path(args.turn_net).resolve())
    bp, bp_path, spec = None, None, None
    if args.blueprint:
        from pokerbot.search.blueprint import make_blueprint

        bp = make_blueprint(f"neural:{args.blueprint}", device=args.device)
        bp_path = str(Path(args.blueprint).resolve())
    else:
        from pokerbot.env.actions import DEFAULT_SPEC

        spec = DEFAULT_SPEC
        log("# no blueprint: the turn tree uses DEFAULT_SPEC")
    meta = generate_turn_start_data(
        turn,
        bp,
        args.out,
        cfg,
        device=args.device,
        resume=args.resume,
        turn_net=turn_path,
        blueprint_path=bp_path,
        spec=spec,
        log=log,
    )
    log(f"# done: {meta.get('timing', {})}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
