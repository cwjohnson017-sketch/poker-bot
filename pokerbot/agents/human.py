"""Text-prompt agent for a human at a terminal."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ..reference.cards import cards_to_str
from .base import BaseAgent

_KIND_CHARS = {0: "f", 1: "c", 2: "r"}

HELP = (
    "commands: f = fold, c/k = check or call, r <amount> = raise to <amount> "
    "(street total), a = all-in, ? = help"
)


def describe_state(state: Any, seat: int) -> str:
    street = ["preflop", "flop", "turn", "river"][state.street]
    lines = [
        f"-- {street} | board [{cards_to_str(state.board, ' ')}] | pot {state.pot}",
        f"   you: seat {seat}, cards [{cards_to_str(state.hole_cards(seat), ' ')}]"
        f"{' (button)' if state.button == seat else ''}",
    ]
    stacks, bets, folded = state.stacks, state.street_bets, state.folded
    for i in range(state.num_players):
        tag = "folded" if folded[i] else f"stack {stacks[i]}, bet {bets[i]}"
        lines.append(f"   seat {i}{' *' if i == seat else ''}: {tag}")
    hist = " ".join(
        f"{p}:{_KIND_CHARS[a.kind]}{a.amount if a.kind == 2 else ''}"
        for s, p, a in state.history
        if s == state.street
    )
    if hist:
        lines.append(f"   this street: {hist}")
    return "\n".join(lines)


class HumanCLIAgent(BaseAgent):
    name = "human"

    def __init__(
        self,
        input_fn: Callable[[str], str] = input,
        output_fn: Callable[[str], None] = print,
        name: str | None = None,
    ) -> None:
        super().__init__(name)
        self.input_fn = input_fn
        self.output_fn = output_fn

    def new_hand(self, seat: int, config: Any) -> None:
        super().new_hand(seat, config)
        self.output_fn(f"== new hand: you are seat {seat}")

    def act(self, state: Any, seat: int, rng: np.random.Generator) -> Any:
        legal = state.legal_actions()
        self.output_fn(describe_state(state, seat))
        opts = []
        if legal.can_fold:
            opts.append("f")
        opts.append("k" if legal.can_check else f"c {legal.call_amount}")
        if legal.min_raise_to > 0:
            if legal.max_raise_to > legal.min_raise_to:
                opts.append(f"r {legal.min_raise_to}..{legal.max_raise_to}")
            opts.append(f"a {legal.max_raise_to}")
        prompt = f"[{' | '.join(opts)}] > "
        while True:
            try:
                line = self.input_fn(prompt).strip().lower()
            except EOFError:
                line = "c"
            parts = line.split()
            if not parts:
                continue
            cmd = parts[0]
            if cmd in ("?", "h", "help"):
                self.output_fn(HELP)
                continue
            if cmd == "f" and legal.can_fold:
                return self.fold()
            if cmd in ("c", "k"):
                return self.check_call()
            if cmd == "a" and legal.min_raise_to > 0:
                return self.raise_to(legal.max_raise_to)
            if cmd == "r" and legal.min_raise_to > 0 and len(parts) == 2 and parts[1].isdigit():
                amt = int(parts[1])
                if legal.min_raise_to <= amt <= legal.max_raise_to or amt == legal.max_raise_to:
                    return self.raise_to(amt)
            self.output_fn(f"illegal or unknown command {line!r}; {HELP}")

    def observe_end(self, state: Any) -> None:
        if self.seat < 0:
            return
        shown = []
        for i in range(state.num_players):
            cards = state.hole_cards(i)
            if cards and i != self.seat:
                shown.append(f"seat {i} shows [{cards_to_str(cards, ' ')}]")
        board = cards_to_str(state.board, " ")
        pay = state.payoffs()[self.seat]
        extra = ("; " + "; ".join(shown)) if shown else ""
        self.output_fn(f"== hand over: board [{board}]{extra}; you {pay:+d}")
