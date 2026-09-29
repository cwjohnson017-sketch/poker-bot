"""Protocol endpoints for one seat.

* :class:`RemoteAgent` is the server-side proxy: an ``Agent`` whose decisions
  are made by a client on the other end of a socket.
* :func:`run_client` is the client side: it connects to a server and plays a
  local ``Agent`` in the seat the server assigns.

Run a client from the command line::

    python scripts/play_client.py --agent human --port 18791
"""

from __future__ import annotations

import argparse
import contextlib
import socket
import sys
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, TextIO

import numpy as np

from ..agents import AGENTS, make_agent
from ..engine_select import get_engine
from ..eval.masking import MaskedState
from .messages import (
    ProtocolError,
    decode_action,
    decode_gamedef,
    decode_hello,
    decode_matchstate,
    encode_action,
    encode_endhand,
    encode_gamedef,
    encode_hello,
    encode_matchstate,
    rebuild_state,
    split_response,
    visible_seats,
)


def open_stream(sock: socket.socket) -> TextIO:
    return sock.makefile("rw", encoding="ascii", newline="\n")


def send_line(stream: TextIO, line: str) -> None:
    stream.write(line + "\n")
    stream.flush()


def recv_line(stream: TextIO) -> str:
    line = stream.readline()
    if not line:
        raise ConnectionError("connection closed")
    return line.rstrip("\r\n")


class RemoteAgent:
    """Server-side ``Agent`` backed by a protocol client."""

    def __init__(self, stream: TextIO, name: str = "remote", sock: socket.socket | None = None):
        self.stream = stream
        self.name = name
        self.sock = sock
        self.seat = -1
        self.hand_no = -1

    @classmethod
    def accept(cls, server_sock: socket.socket, config: Any) -> RemoteAgent:
        """Accept one client, do the VERSION/GAMEDEF handshake."""
        conn, _addr = server_sock.accept()
        stream = open_stream(conn)
        name = decode_hello(recv_line(stream))
        send_line(stream, encode_gamedef(config))
        return cls(stream, name, conn)

    def new_hand(self, seat: int, config: Any) -> None:
        self.seat = seat
        self.hand_no += 1

    def act(self, state: Any, seat: int, rng: np.random.Generator) -> Any:
        line = encode_matchstate(state, seat, self.hand_no)
        send_line(self.stream, line)
        return decode_action(split_response(line, recv_line(self.stream)))

    def observe_end(self, state: Any) -> None:
        send_line(self.stream, encode_endhand(state, self.seat, self.hand_no))

    def close(self) -> None:
        with contextlib.suppress(OSError):
            send_line(self.stream, "BYE")
        try:
            self.stream.close()
        finally:
            if self.sock is not None:
                self.sock.close()


@dataclass
class ClientStats:
    hands: int = 0
    total: int = 0
    payoffs: list[int] = field(default_factory=list)


def run_client(
    agent: Any,
    host: str = "127.0.0.1",
    port: int = 18791,
    engine: ModuleType | None = None,
    seed: int | None = None,
    timeout: float | None = None,
) -> ClientStats:
    """Connect to a match server and play ``agent`` until the server says BYE."""
    engine = engine or get_engine()
    rng = np.random.default_rng(seed)
    stats = ClientStats()
    with socket.create_connection((host, port), timeout=timeout) as sock:
        stream = open_stream(sock)
        send_line(stream, encode_hello(getattr(agent, "name", "client")))
        config = decode_gamedef(recv_line(stream), engine)
        current_hand = None
        while True:
            try:
                line = recv_line(stream)
            except ConnectionError:
                break
            if line == "BYE":
                break
            ms = decode_matchstate(line)
            if ms.hand_no != current_hand:
                agent.new_hand(ms.seat, config)
                current_hand = ms.hand_no
            state = rebuild_state(ms, config, engine, rng)
            if ms.ended:
                shown = [p for p in visible_seats(ms) if p != ms.seat]
                agent.observe_end(MaskedState(state, ms.seat, config, rng, revealed=shown))
                pay = ms.payoffs[ms.seat] if ms.payoffs else 0
                stats.hands += 1
                stats.total += pay
                stats.payoffs.append(pay)
                continue
            if state.is_terminal or state.current_player != ms.seat:
                raise ProtocolError(f"asked to act out of turn: {line!r}")
            view = MaskedState(state, ms.seat, config, rng)
            action = agent.act(view, ms.seat, rng)
            send_line(stream, f"{line}:{encode_action(action)}")
        stream.close()
    return stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Play an agent against a pokerbot match server")
    ap.add_argument("--agent", default="human", help=f"one of {sorted(AGENTS)}")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18791)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--engine", default=None, choices=["auto", "reference", "rust"])
    args = ap.parse_args(argv)
    engine = get_engine(args.engine)
    stats = run_client(make_agent(args.agent), args.host, args.port, engine, args.seed)
    print(f"played {stats.hands} hands, net {stats.total:+d} chips")
    return 0


if __name__ == "__main__":
    sys.exit(main())
