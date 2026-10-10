#!/usr/bin/env python
"""Generate turn-end value net training data: blueprint turn-end states with
targets bootstrapped from a river value net (no solving; see
pokerbot/search/turn_data.py and docs/value_net.md).

python scripts/gen_turn_data.py --river-net runs/vn/river.pt --blueprint runs/dcfr4_distilled_v2 \
    --out data/value_turn --samples 200000 [--mix 0.5,0.25,0.25] [--resume]

--river-net checkdown uses the exact check-down river (ShowdownOracle) instead
of a net (tests, smoke runs). --blueprint may be omitted when --mix has no
self-play share (random ranges only). Each sample costs 48 river-net rows.
Writes one shard per --shard-size samples plus meta.json (kind turn_end);
--resume skips the shards that exist (each shard is seeded from (seed, index)).
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pokerbot.search.turn_data import TurnGenConfig, generate_turn_data  # noqa: E402


def _floats(text: str) -> tuple[float, ...]:
    return tuple(float(x) for x in text.split(","))


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def main(argv: list[str] | None = None) -> int:
    d = TurnGenConfig()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--river-net", required=True, help="river value-net checkpoint, or checkdown")
    ap.add_argument("--blueprint", default=None, help="distilled Deep CFR run directory")
    ap.add_argument("--out", required=True, help="output shard directory")
    ap.add_argument("--samples", type=int, required=True)
    ap.add_argument("--mix", type=_floats, default=d.mix, help="self-play,perturbed,random")
    ap.add_argument("--seed", type=int, default=d.seed)
    ap.add_argument("--shard-size", type=int, default=d.shard_size)
    ap.add_argument("--explore", type=float, default=d.explore, help="self-play uniform mixing")
    ap.add_argument("--n-envs", type=int, default=d.n_envs, help="self-play lockstep slots")
    ap.add_argument("--c-range", type=_floats, default=d.c_range, help="random states' c: lo,hi")
    ap.add_argument("--chunk", type=int, default=d.chunk, help="river-net rows per call")
    ap.add_argument("--batch", type=int, default=d.batch, help="states per target batch")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resume", action="store_true", help="continue an existing output dir")
    args = ap.parse_args(argv)
    cfg = TurnGenConfig(
        samples=args.samples,
        mix=args.mix,
        seed=args.seed,
        shard_size=args.shard_size,
        explore=args.explore,
        n_envs=args.n_envs,
        c_range=tuple(int(x) for x in args.c_range),
        chunk=args.chunk,
        batch=args.batch,
    )
    if cfg.mix[0] + cfg.mix[1] > 0 and not args.blueprint:
        ap.error("--blueprint is required when --mix has a self-play share")
    log(f"# gen_turn_data: {cfg.to_dict()}, river net {args.river_net}, out {args.out}")
    if args.river_net == "checkdown":
        from pokerbot.search.value_leaf import ShowdownOracle

        river, river_path = ShowdownOracle(max_boards=1 << 14), "checkdown"
    else:
        from pokerbot.search.value_net import ValueNetPredictor

        river = ValueNetPredictor.from_path(args.river_net, args.device)
        river_path = str(Path(args.river_net).resolve())
    bp, bp_path = None, None
    if args.blueprint:
        from pokerbot.search.blueprint import make_blueprint

        bp = make_blueprint(f"neural:{args.blueprint}", device=args.device)
        bp_path = str(Path(args.blueprint).resolve())
    meta = generate_turn_data(
        river,
        bp,
        args.out,
        cfg,
        device=args.device,
        resume=args.resume,
        river_net=river_path,
        blueprint_path=bp_path,
        log=log,
    )
    log(f"# done: {meta.get('timing', {})}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
