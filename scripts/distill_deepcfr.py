#!/usr/bin/env python
"""Distill a Deep CFR run's SD-CFR average strategy into one policy net per seat.

python scripts/distill_deepcfr.py --run runs/dcfr4 --stride 10 --out runs/dcfr4_distilled
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pokerbot.blueprint.deepcfr.distill import DistillConfig, distill  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="Deep CFR run directory")
    ap.add_argument("--out", required=True, help="output run directory")
    ap.add_argument("--stride", type=int, default=10, help="every k-th net of the average")
    ap.add_argument("--last-n", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--data", default=None, help="cache of the generated rows (.pt)")
    for name, default in DistillConfig().__dict__.items():
        if isinstance(default, tuple):
            ap.add_argument(f"--{name.replace('_', '-')}", type=float, nargs="*", default=default)
        else:
            ap.add_argument(f"--{name.replace('_', '-')}", type=type(default), default=default)
    args = ap.parse_args(argv)
    vals = {k: getattr(args, k) for k in DistillConfig().__dict__}
    vals["street_mix"] = tuple(vals["street_mix"] or ())
    cfg = DistillConfig(**vals)
    rep = distill(
        args.run, args.out, cfg, args.stride, args.last_n, args.device, data_path=args.data
    )
    print(rep, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
