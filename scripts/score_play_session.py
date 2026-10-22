#!/usr/bin/env python
"""Score a play session with AIVAT: raw, luck-adjusted and AIVAT win rates.

    python scripts/score_play_session.py runs/play/20261021-140509
    python scripts/score_play_session.py runs/play/20261021-140509 --values range

Reads ``hands.jsonl`` and ``solves/`` of the session directory
(:mod:`pokerbot.play.log`) and prints the person's win rate (and the bot's,
its negative) three ways, with 95% bootstrap CIs and per-hand standard
deviations (:func:`pokerbot.eval.aivat.score_session`). The action terms are
recomputed from the logged arrays, so this also scores sessions logged before
the play app showed AIVAT.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pokerbot.eval.aivat import score_session  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("session", nargs="+", help="play session directories")
    p.add_argument(
        "--values",
        choices=("pair", "range"),
        default="pair",
        help="action values against the opponent's actual hand (pair) or range",
    )
    args = p.parse_args(argv)
    for d in args.session:
        print(score_session(d, values=args.values).summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
