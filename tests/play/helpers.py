"""Drive a PlaySession from a test the way the browser does."""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np

from pokerbot.play import PlaySession, SessionLog

META = {"session": "20260101-000000", "mode": "evaluation"}


def new_log(tmp_path, name: str = "s") -> SessionLog:
    return SessionLog(tmp_path / name, dict(META))


def calling_station(view: dict) -> tuple[str, int | None]:
    legal = view["hand"]["legal"]
    return ("check", None) if legal["can_check"] else ("call", None)


def mixed(seed: int = 0) -> Callable[[dict], tuple[str, int | None]]:
    """Folds a fifth of the time when facing a bet, min-raises a fifth, else checks or calls."""
    rng = np.random.default_rng(seed)

    def choose(view: dict) -> tuple[str, int | None]:
        legal = view["hand"]["legal"]
        u = rng.random()
        if legal["can_fold"] and u < 0.2:
            return "fold", None
        if legal["min_raise_to"] > 0 and u > 0.8:
            return "raise", min(legal["min_raise_to"], legal["max_raise_to"])
        return calling_station(view)

    return choose


def drive(
    session: PlaySession,
    choose: Callable[[dict], tuple[str, int | None]],
    timeout: float = 120.0,
    on_view: Callable[[dict], None] | None = None,
    stop_after: int | None = None,
    since: int = 0,
) -> dict:
    """Deal and act until the session finishes (or, with ``stop_after``, stop it
    at the person's first decision of hand number ``stop_after``), starting from
    the first view newer than ``since``. Returns the last view."""
    seen = since
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        v = session.snapshot(since=seen, timeout=5.0)
        if v["version"] == seen:
            continue
        seen = v["version"]
        if on_view is not None:
            on_view(v)
        status = v["status"]
        if status in ("ready", "hand_over"):
            session.request_next()
        elif status == "human":
            if stop_after is not None and v["hand"]["number"] >= stop_after:
                session.stop()
                continue
            ok, err = session.submit_action(*choose(v))
            assert ok, err
        elif status == "finished":
            return v
        elif status == "error":
            raise AssertionError(v["message"])
    raise TimeoutError(f"session still {session.snapshot()['status']}")
