#!/usr/bin/env python
"""Held-out report of value nets on fixed shards (compare nets on the same data).

For each checkpoint (a river or a turn-end net, by its kind), the report of
:func:`pokerbot.search.value_train.heldout_report` on the shards of ``--data``:
MAE / RMSE / range-weighted MAE / game-value error in pot units, overall, per
pot bin and per source, with the zero and bucket-oracle references.

python scripts/eval_value_net.py --data runs/value_net/river_heldout \
    --net runs/value_net/river_v1.pt --net runs/value_net/river_v2.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pokerbot.search.turn_net import TurnFeatureCache, checkpoint_kind  # noqa: E402
from pokerbot.search.value_net import load_value_net  # noqa: E402
from pokerbot.search.value_train import (  # noqa: E402
    ValueData,
    format_report,
    heldout_report,
    load_shards,
    sample_stats,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", action="append", required=True, help="shard directory (repeat)")
    ap.add_argument("--net", action="append", required=True, help="checkpoint (repeat)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--out", help="JSON with every report")
    args = ap.parse_args(argv)
    dev = torch.device(args.device)
    raw = load_shards(args.data)
    if args.max_samples:
        raw = {k: v[: args.max_samples] for k, v in raw.items()}
    reports = {}
    datasets: dict[tuple, ValueData] = {}
    for path in args.net:
        kind = checkpoint_kind(path)
        net = load_value_net(path, dev).eval()
        sb = (getattr(net, "meta", None) or {}).get("turn_features", {}).get("spread_buckets", 8)
        key = (kind, sb if kind == "turn_end" else 0)
        if key not in datasets:
            cache = None
            if kind == "turn_end":
                cache = TurnFeatureCache(dev, sb, max_boards=None)
            datasets[key] = ValueData(raw, dev, 0.0, cache)
        ds = datasets[key]
        idx = torch.arange(len(ds), device=dev)
        st = sample_stats(net, ds, idx, dev, dev.type == "cuda")
        rep = heldout_report(st, ds.c[idx].cpu(), ds.source[idx.cpu()])
        reports[path] = rep
        print(f"== {path} ({kind}) on {len(ds):,} samples from {', '.join(args.data)}")
        print(format_report(rep), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(reports, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
