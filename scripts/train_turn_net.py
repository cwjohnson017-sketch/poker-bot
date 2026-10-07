#!/usr/bin/env python
"""Train the turn-end leaf value net on bootstrapped turn-end shards
(scripts/gen_turn_data.py; boards [n, 4]), or with --kind turn_start the
turn-start net on turn-root shards (scripts/gen_turn_start_data.py).

python scripts/train_turn_net.py --data data/value_turn --out runs/vn/turn.pt --steps 20000
python scripts/train_turn_net.py --data d1 --heldout-data d2 --out turn.pt --spread-buckets 8
python scripts/train_turn_net.py --kind turn_start --data data/value_turn_start --out ts.pt

--spread-buckets 1 buckets combos by their mean river strength only (1-D);
k > 1 (default 8) splits each of buckets / k mean buckets into k sub-buckets by the spread
of the river strength (2-D: separates draws from made hands). Every field of
ValueNetConfig and ValueTrainConfig is a flag, as in scripts/train_value_net.py.
The checkpoint records kind turn_end, so a search agent with leaf.net pointing
at it uses one net row per leaf (TurnEndLeafEvaluator); a turn_start checkpoint
values the flop-end leaves of depth_streets 0 flop solves (FlopEndLeafEvaluator).
Shard directories whose meta.json names another kind are rejected.
"""

import argparse
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pokerbot.search.turn_net import (  # noqa: E402
    DEFAULT_SPREAD_BUCKETS,
    KIND,
    TURN_KINDS,
    train_turn_net,
)
from pokerbot.search.value_net import ValueNetConfig  # noqa: E402
from pokerbot.search.value_train import ValueTrainConfig  # noqa: E402


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
    ap.add_argument(
        "--spread-buckets", type=int, default=DEFAULT_SPREAD_BUCKETS, help="1 = 1-D buckets"
    )
    ap.add_argument("--kind", choices=TURN_KINDS, default=KIND, help="turn_end or turn_start")
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
    train_turn_net(
        args.data,
        args.out,
        net_cfg,
        cfg,
        args.spread_buckets,
        args.heldout_data,
        args.device,
        log,
        kind=args.kind,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
