#!/usr/bin/env python
"""Connect an agent (default: you, at the terminal) to a match server.

python scripts/play_client.py --agent human --port 18791
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pokerbot.protocol.client import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
