#!/usr/bin/env python
"""Generate leaf value net training data: blueprint river states solved exactly
with BatchRiverSolver (see pokerbot/search/value_data.py and docs/value_net.md).

python scripts/gen_value_data.py --blueprint runs/dcfr4_distilled_v2 --out data/value_river \
    --samples 20000 --mix 0.5,0.25,0.25 --iterations 400 --batch 512 --seed 0 [--resume]

Writes one shard per --shard-size samples plus meta.json. Safe to run detached
for hours: logs are flushed per shard, shards are written atomically, and
--resume skips the shards that exist (each shard is seeded from (seed, index)).
Any error stops the run with a traceback.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pokerbot.search.blueprint import make_blueprint  # noqa: E402
from pokerbot.search.value_data import GenConfig, generate  # noqa: E402


def _floats(text: str) -> tuple[float, ...]:
    return tuple(float(x) for x in text.split(","))


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def main(argv: list[str] | None = None) -> int:
    d = GenConfig()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--blueprint", required=True, help="distilled Deep CFR run directory")
    ap.add_argument("--out", required=True, help="output shard directory")
    ap.add_argument("--samples", type=int, required=True)
    ap.add_argument("--mix", type=_floats, default=d.mix, help="self-play,perturbed,random")
    ap.add_argument("--iterations", type=int, default=d.iterations, help="DCFR iterations")
    ap.add_argument("--batch", type=int, default=d.batch, help="instances per solve")
    ap.add_argument("--seed", type=int, default=d.seed)
    ap.add_argument("--shard-size", type=int, default=d.shard_size)
    ap.add_argument("--explore", type=float, default=d.explore, help="self-play uniform mixing")
    ap.add_argument("--n-envs", type=int, default=d.n_envs, help="self-play lockstep slots")
    ap.add_argument("--c-range", type=_floats, default=d.c_range, help="random states' c: lo,hi")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resume", action="store_true", help="continue an existing output dir")
    args = ap.parse_args(argv)
    cfg = GenConfig(
        samples=args.samples,
        mix=args.mix,
        iterations=args.iterations,
        batch=args.batch,
        seed=args.seed,
        shard_size=args.shard_size,
        explore=args.explore,
        n_envs=args.n_envs,
        c_range=tuple(int(x) for x in args.c_range),
    )
    log(f"# gen_value_data: {cfg.to_dict()}, blueprint {args.blueprint}, out {args.out}")
    bp = make_blueprint(f"neural:{args.blueprint}", device=args.device)
    meta = generate(
        bp,
        args.out,
        cfg,
        device=args.device,
        resume=args.resume,
        blueprint_path=str(Path(args.blueprint).resolve()),
        log=log,
    )
    log(f"# done: {meta.get('timing', {})}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
