#!/usr/bin/env python
"""Train the tabular MCCFR blueprint from a YAML config.

python scripts/train_mccfr.py --config configs/mccfr_small.yaml
python scripts/train_mccfr.py --config configs/mccfr_hunl.yaml --threads 16
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pokerbot.blueprint.mccfr.train import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
