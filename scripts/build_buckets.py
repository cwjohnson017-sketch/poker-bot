#!/usr/bin/env python
"""Build postflop card-bucket tables for the MCCFR blueprint.

python scripts/build_buckets.py --config configs/buckets_tiny.yaml
python scripts/build_buckets.py --config configs/buckets_hunl.yaml --device cuda
python scripts/build_buckets.py --config configs/buckets_hunl.yaml --streets river --limit 5000
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pokerbot.abstraction.buckets import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
