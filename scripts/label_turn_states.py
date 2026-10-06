#!/usr/bin/env python
"""Turn-end net shards from saved turn-end states (e.g. on-policy leaf states).

Labels ``--states`` (a ``torch.save`` of ``boards [n, 4]``, ``c``, ``stack``,
``ranges [n, 2, 1326]``, as written by ``scripts/gen_onpolicy_data.py``) with
targets bootstrapped from ``--river-net`` (its exact river-card average,
``turn_data.label_states``) and writes turn-end shards with ``source``
``--source`` (3 = on-policy).

python scripts/label_turn_states.py --river-net runs/value_net/river_v2.pt \
    --states runs/value_net/river_onpolicy/turn_states.pt --out runs/value_net/turn_onpolicy
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--river-net", required=True)
    ap.add_argument("--states", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", type=int, default=3)
    ap.add_argument("--shard-size", type=int, default=4096)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    dev = torch.device(args.device)
    t0 = time.time()
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
