#!/usr/bin/env python
"""Serve a match over the line protocol.

Seat 0 waits for a remote client, seat 1 is the equity baseline::

    python scripts/serve_match.py --seat0 remote --seat1 equity --hands 100
    python scripts/play_client.py --agent human      # in another terminal
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pokerbot.protocol.server import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
