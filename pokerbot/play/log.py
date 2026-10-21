"""The on-disk record of a play session: one directory per session.

* ``session.json``: what was played. The game, the bot spec and its search
  config, the seed, the fixed sample size, the git commit, start and resume
  times.
* ``dealt.jsonl``: one line per dealt hand, written before the hand is played.
* ``hands.jsonl``: one line per finished hand. The deck, the button, every
  action with its decision time, the payoffs, the luck-adjusted result, and the
  bot's decisions (actions, the one taken, timings, search stats).
* ``solves/hand_<n>.npz``: the bot's arrays for hand ``n``, per decision ``k``:
  ``d<k>_strategy`` ``[1326, A]``, ``d<k>_reach`` ``[2, 1326]`` and, for
  searched decisions, ``d<k>_values`` ``[1326, A]`` (expected chips per action).
* ``hands.txt``: the hands in the repo's plain-text history format
  (:mod:`pokerbot.eval.history`).
* ``pokerstars.txt``: the hands in PokerStars' text format, from the person's
  seat (:mod:`.pokerstars`).
* ``errors.log``: the traceback, if the session stopped on an error.

A hand that was dealt but never finished (the server stopped mid-hand) has a
line in ``dealt.jsonl`` and none in ``hands.jsonl``: it is abandoned, never
dealt again, and reported as such.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np


def _append_line(path: Path, obj: Any) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, separators=(",", ":")) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _append_text(path: Path, text: str) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())


def _read_lines(path: Path) -> Iterator[dict]:
    if not path.exists():
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def decision_arrays(decisions: Sequence[dict]) -> dict[str, np.ndarray]:
    """The ``.npz`` arrays of a hand's bot decisions (see the module docstring)."""
    out: dict[str, np.ndarray] = {}
    for k, d in enumerate(decisions):
        out[f"d{k}_strategy"] = np.asarray(d["strategy"], dtype=np.float32)
        out[f"d{k}_reach"] = np.asarray(d["reach"], dtype=np.float32)
        if d.get("values") is not None:
            out[f"d{k}_values"] = np.asarray(d["values"], dtype=np.float32)
    return out


def decision_summary(d: dict) -> dict:
    """The JSON part of one bot decision (everything but the arrays)."""
    stats = {
        k: v
        for k, v in (d.get("stats") or {}).items()
        if isinstance(v, (int, float, str, bool)) or v is None
    }
    return {
        "street": int(d["street"]),
        "source": d["source"],
        "actions": [[int(k), int(a)] for k, a in d["actions"]],
        "taken": int(d["taken"]),
        "seconds": float(d["seconds"]),
        "record_seconds": float(d.get("record_seconds", 0.0)),
        "has_values": d.get("values") is not None,
        "stats": stats,
    }


class SessionLog:
    """Append-only files of one session. ``meta`` is written on creation; an
    existing directory is reopened (resume) and its ``session.json`` kept."""

    def __init__(self, directory: str | Path, meta: dict | None = None) -> None:
        self.dir = Path(directory)
        self.meta_path = self.dir / "session.json"
        if self.meta_path.exists():
            self.meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        else:
            if meta is None:
                raise FileNotFoundError(f"{self.meta_path} does not exist and no meta was given")
            self.dir.mkdir(parents=True, exist_ok=True)
            self.meta = dict(meta)
            self.save_meta()
        (self.dir / "solves").mkdir(exist_ok=True)

    def save_meta(self) -> None:
        tmp = self.meta_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.meta, indent=2), encoding="utf-8")
        os.replace(tmp, self.meta_path)

    # -- reading ------------------------------------------------------------

    def dealt(self) -> list[dict]:
        return list(_read_lines(self.dir / "dealt.jsonl"))

    def hands(self) -> list[dict]:
        return list(_read_lines(self.dir / "hands.jsonl"))

    def solves(self, hand: int) -> dict[str, np.ndarray]:
        path = self.dir / "solves" / f"hand_{hand:06d}.npz"
        with np.load(path) as z:
            return {k: z[k] for k in z.files}

    # -- writing ------------------------------------------------------------

    def note_dealt(self, hand: int, deck: Sequence[int], button: int, when: str) -> None:
        _append_line(
            self.dir / "dealt.jsonl",
            {"hand": int(hand), "button": int(button), "deck": [int(c) for c in deck], "at": when},
        )

    def write_hand(
        self,
        record: dict,
        decisions: Sequence[dict],
        text: str | None = None,
        pokerstars: str | None = None,
    ) -> None:
        """One finished hand: the arrays first, then the JSON line that points at
        them, then the text formats."""
        rec = dict(record)
        arrays = decision_arrays(decisions)
        rec["bot_decisions"] = [decision_summary(d) for d in decisions]
        if arrays:
            name = f"solves/hand_{int(rec['hand']):06d}.npz"
            np.savez_compressed(self.dir / name, **arrays)
            rec["solves"] = name
        _append_line(self.dir / "hands.jsonl", rec)
        if text:
            _append_text(self.dir / "hands.txt", text.rstrip("\n") + "\n\n")
        if pokerstars:
            _append_text(self.dir / "pokerstars.txt", pokerstars.rstrip("\n") + "\n\n\n")
