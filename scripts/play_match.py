#!/usr/bin/env python
"""Play a match between two agents and print mbb/h with a 95% bootstrap CI.

python scripts/play_match.py --a always_call --b equity --hands 2000 --duplicate --seed 0
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pokerbot.eval.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
