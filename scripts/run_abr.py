#!/usr/bin/env python
"""Approximate best response: train a DQN best response to a frozen policy on the
vectorized env, then report what it wins in mbb/h.

python scripts/run_abr.py --config configs/abr_tiny.yaml
python scripts/run_abr.py --config configs/abr_4070ti.yaml --opponent neural:runs/deepcfr/it40.pt \
    --log-dir runs/abr/it40 --out results/abr_it40.json
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pokerbot.eval.abr import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
