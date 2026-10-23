"""The play app: a person against the bot at a local web table, every hand logged.

* :mod:`.session`: :class:`PlaySession`, the hand loop (heads-up, evaluation mode)
* :mod:`.server`: :class:`PlayServer`, the HTTP server and its JSON API
* :mod:`.log`: :class:`SessionLog`, the session directory
* :mod:`.stats`: running win rates: raw, luck-adjusted and AIVAT
* :mod:`.pokerstars`: PokerStars-format hand histories

Entry point: ``scripts/play_web.py``. See ``pokerbot/play/README.md``.
"""

from .log import SessionLog
from .server import PlayServer
from .session import PlaySession, deal, parse_action

__all__ = ["PlayServer", "PlaySession", "SessionLog", "deal", "parse_action"]
