"""Match server: seats local agents and remote protocol clients and drives
hands through :mod:`pokerbot.eval.match`.

    # seat 0 waits for a remote client, seat 1 is the equity baseline
    python scripts/serve_match.py --seat0 remote --seat1 equity --hands 100
    python scripts/play_client.py --agent human        # in another terminal
"""

from __future__ import annotations

import argparse
import socket
import sys
from collections.abc import Callable, Sequence
from types import ModuleType
from typing import Any, TextIO

from ..agents import AGENTS, make_agent
from ..config import game_config
from ..engine_select import get_engine
from ..eval.match import MatchResult, run_duplicate_match, run_match
from .client import RemoteAgent

DEFAULT_PORT = 18791


def serve_match(
    config: Any,
    seats: Sequence[Any | None],
    num_hands: int,
    seed: int = 0,
    host: str = "127.0.0.1",
    port: int = DEFAULT_PORT,
    duplicate: bool = False,
    engine: ModuleType | None = None,
    ready: Callable[[int], None] | None = None,
    accept_timeout: float | None = None,
    history: TextIO | None = None,
    on_illegal: str = "fold",
) -> MatchResult:
    """Run a match where every ``None`` entry of ``seats`` is filled by a remote
    client connecting to ``host:port``. ``ready(port)`` is called once the
    socket is listening (useful with ``port=0``). With ``duplicate=True`` the
    match is heads-up duplicate and ``num_hands // 2`` deals are played.
    Illegal remote actions become fold/check by default."""
    engine = engine or get_engine()
    if len(seats) != config.num_players:
        raise ValueError("need one seat entry per player")
    remotes: list[RemoteAgent] = []
    agents = list(seats)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((host, port))
        srv.listen()
        srv.settimeout(accept_timeout)
        if ready is not None:
            ready(srv.getsockname()[1])
        try:
            for i, a in enumerate(agents):
                if a is None:
                    remote = RemoteAgent.accept(srv, config)
                    remotes.append(remote)
                    agents[i] = remote
            if duplicate:
                return run_duplicate_match(
                    agents[0],
                    agents[1],
                    config,
                    max(1, num_hands // 2),
                    seed,
                    engine,
                    history,
                    on_illegal=on_illegal,
                )
            return run_match(
                agents, config, num_hands, seed, engine, history, on_illegal=on_illegal
            )
        finally:
            for r in remotes:
                r.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Serve a poker match over the line protocol")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--seat0", default="remote", help=f"'remote' or one of {sorted(AGENTS)}")
    ap.add_argument("--seat1", default="remote")
    ap.add_argument("--hands", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--duplicate", action="store_true")
    ap.add_argument("--engine", default=None, choices=["auto", "reference", "rust"])
    args = ap.parse_args(argv)
    engine = get_engine(args.engine)
    config = game_config(None, engine)
    seats = [None if s == "remote" else make_agent(s) for s in (args.seat0, args.seat1)]
    res = serve_match(
        config,
        seats,
        args.hands,
        args.seed,
        args.host,
        args.port,
        args.duplicate,
        engine,
        ready=lambda p: print(f"listening on {args.host}:{p}", flush=True),
    )
    print(res.summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
