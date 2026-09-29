#!/usr/bin/env python
"""Checkpoint ladder: duplicate matches between agents, ratings, JSON + markdown.

python scripts/run_ladder.py --agents always_call always_raise random equity --hands 2000
python scripts/run_ladder.py --checkpoints "runs/mccfr/ckpt_*.pkl" --kind blueprint \
    --schedule newest -k 5 --anchor equity --hands 20000 --workers 8 --out results/ladder.json
python scripts/run_ladder.py --config configs/ladder.yaml
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pokerbot.eval.ladder import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
