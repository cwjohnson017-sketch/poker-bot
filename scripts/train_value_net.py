#!/usr/bin/env python
"""Train the river leaf value net on solved river samples (shard_*.pt directories).

python scripts/train_value_net.py --data runs/vn_data --out runs/vn/river.pt --steps 20000
python scripts/train_value_net.py --data d1 --data d2 --heldout-data d3 --out vn.pt --buckets 128

Writes the checkpoint ``--out`` and the held-out report next to it (``.json``).
Every field of ValueNetConfig and ValueTrainConfig is a flag (``--residual-head``,
``--huber-delta 0.5``, ...).
"""

import argparse
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pokerbot.search.value_net import ValueNetConfig  # noqa: E402
from pokerbot.search.value_train import ValueTrainConfig, train_value_net  # noqa: E402


def _add_fields(ap: argparse.ArgumentParser, cls: type) -> None:
    for f in dataclasses.fields(cls):
        flag = f"--{f.name.replace('_', '-')}"
        if isinstance(f.default, bool):
            ap.add_argument(flag, action=argparse.BooleanOptionalAction, default=f.default)
        else:
            ap.add_argument(flag, type=type(f.default), default=f.default)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", action="append", required=True, help="shard directory (repeat)")
    ap.add_argument("--heldout-data", action="append", default=None, help="held-out shards")
    ap.add_argument("--out", required=True, help="checkpoint path (.pt)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--quiet", action="store_true")
    _add_fields(ap, ValueNetConfig)
    _add_fields(ap, ValueTrainConfig)
    args = ap.parse_args(argv)
    net_cfg = ValueNetConfig(
        **{f.name: getattr(args, f.name) for f in dataclasses.fields(ValueNetConfig)}
    )
    cfg = ValueTrainConfig(
        **{f.name: getattr(args, f.name) for f in dataclasses.fields(ValueTrainConfig)}
    )
    log = None if args.quiet else (lambda *a, **k: print(*a, **k, flush=True))
    train_value_net(args.data, args.out, net_cfg, cfg, args.heldout_data, args.device, log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
