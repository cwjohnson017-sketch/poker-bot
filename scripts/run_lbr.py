#!/usr/bin/env python
"""Local best response against an agent: mbb/h with CI and a per-street breakdown.

python scripts/run_lbr.py --opponent always_call --hands 2000
python scripts/run_lbr.py --opponent blueprint:runs/mccfr/final.pkl --hands 20000 --workers 8
python scripts/run_lbr.py --config configs/lbr.yaml
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pokerbot.eval.lbr import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
