#!/usr/bin/env python
"""Turn-end net shards from saved turn-end states (e.g. on-policy leaf states).

Labels ``--states`` (a ``torch.save`` of ``boards [n, 4]``, ``c``, ``stack``,
``ranges [n, 2, 1326]``, as written by ``scripts/gen_onpolicy_data.py``) with
targets bootstrapped from ``--river-net`` (its exact river-card average,
``turn_data.label_states``) and writes turn-end shards with ``source``
``--source`` (3 = on-policy).

python scripts/label_turn_states.py --river-net runs/value_net/river_v2.pt \
    --states runs/value_net/river_onpolicy/turn_states.pt --out runs/value_net/turn_onpolicy

``--shards DIR`` relabels the states of an existing turn-end shard directory
instead (e.g. turn_b with a newer river net), shard by shard, keeping each
shard's name and source tags. Most of gen_turn_data.py's time is the
self-play that produces the states, so this is several times faster than
generating new data.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pokerbot.search.turn_data import label_states  # noqa: E402
from pokerbot.search.turn_net import load_leaf_predictor  # noqa: E402
from pokerbot.search.value_data import shard_name  # noqa: E402
from pokerbot.search.value_train import save_shard  # noqa: E402


def relabel_shards(args: argparse.Namespace, dev: torch.device, t0: float) -> int:
    """Relabel every shard of ``args.shards`` into ``args.out`` (same names)."""
    src = Path(args.shards)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((src / "meta.json").read_text())
    if meta.get("kind") != "turn_end":
        raise SystemExit(f"{src} is not a turn-end shard directory (kind {meta.get('kind')})")
    river = load_leaf_predictor(args.river_net, dev)
    paths = sorted(src.glob("shard_*.pt"))
    n = 0
    for i, path in enumerate(paths):
        dst = out / path.name
        if dst.exists():  # resumable
            continue
        st = torch.load(path, weights_only=True)
        part = {
            "boards": st["boards"].long(),
            "c": st["c"].long(),
            "stack": st["stack"].long(),
            "ranges": st["ranges"].float(),
            "source": st["source"],
        }
        save_shard(dst, label_states(river, part, dev))
        n += int(part["c"].shape[0])
        if (i + 1) % 25 == 0 or i + 1 == len(paths):
            print(f"# {i + 1}/{len(paths)} shards, {time.time() - t0:.0f}s", flush=True)
    meta = {**meta, "river_net": str(Path(args.river_net).resolve()), "relabelled_from": str(src)}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"# relabelled {n} states from {src} into {out} in {time.time() - t0:.0f}s", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--river-net", required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--states", help="a torch.save of turn-end states")
    src.add_argument("--shards", help="a turn-end shard directory to relabel")
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", type=int, default=3)
    ap.add_argument("--shard-size", type=int, default=4096)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    dev = torch.device(args.device)
    t0 = time.time()
    if args.shards:
        return relabel_shards(args, dev, t0)
    st = torch.load(args.states, weights_only=True)
    n = int(st["c"].shape[0])
    states = {
        "boards": st["boards"].long(),
        "c": st["c"].long(),
        "stack": st["stack"].long(),
        "ranges": st["ranges"].float(),
        "source": torch.full((n,), args.source, dtype=torch.uint8),
    }
    river = load_leaf_predictor(args.river_net, dev)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for s in range(0, n, args.shard_size):
        part = {k: v[s : s + args.shard_size] for k, v in states.items()}
        rows = label_states(river, part, dev)
        save_shard(out / shard_name(s // args.shard_size), rows)
    meta = {
        "kind": "turn_end",
        "river_net": str(Path(args.river_net).resolve()),
        "states": str(Path(args.states).resolve()),
        "source": args.source,
        "samples": n,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"# labelled {n} turn-end states into {out} in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
