#!/usr/bin/env python
"""Train the neural blueprint (Deep CFR with the SD-CFR average).

python scripts/train_deepcfr.py --config configs/deepcfr_tiny.yaml
python scripts/train_deepcfr.py --config configs/deepcfr_4070ti.yaml --out runs/dcfr1
python scripts/train_deepcfr.py --config configs/deepcfr_4070ti.yaml --resume runs/dcfr1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pokerbot.blueprint.deepcfr.config import DeepCFRConfig  # noqa: E402
from pokerbot.blueprint.deepcfr.trainer import DeepCFRTrainer  # noqa: E402
from pokerbot.config import run_info  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", required=True, help="YAML run config")
    ap.add_argument("--out", help="run directory (default: out_dir from the config)")
    ap.add_argument("--resume", help="resume from this run directory (its last saved state)")
    ap.add_argument("--iterations", type=int, help="override the number of CFR iterations")
    ap.add_argument("--device", help="override the device (cpu, cuda, cuda:1, auto)")
    args = ap.parse_args(argv)

    cfg = DeepCFRConfig.load(args.config)
    if args.device:
        cfg.device = args.device
    out = args.resume or args.out or cfg.out_dir
    info = run_info(args.config)
    print(f"# git {info['git']} | config {info['config']} | out {out}", flush=True)
    trainer = DeepCFRTrainer(cfg, out, resume=bool(args.resume))
    try:
        trainer.run(args.iterations)
    finally:
        trainer.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
