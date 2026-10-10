"""Luck-adjusted results for heads-up bot-vs-bot hands (variance reduction).

The evaluator sees both hands, so card luck can be taken out of each hand's
result without bias: both corrections below have zero mean given everything
dealt before them.

* **All-in EV.** A hand that ends in a called all-in before the river is
  scored ``stake * (2 E - 1)``, its expectation over the undealt board,
  instead of the realized runout.
* **Street control variates.** When a new street is dealt with ``c`` chips
  in per player, seat 0's result gets ``-2c * (E_new - E_old)``: the change
  in its check-down value caused by the new cards.

``E`` is seat 0's equity against seat 1's actual hand given the board seen so
far (exact from the flop on, Monte Carlo preflop;
:func:`pokerbot.env.equity.street_equities`). A hand checked down from the
flop scores exactly its flop equity.

These are the chance terms of AIVAT (Burch et al., AAAI 2018) with a
check-down value function. AIVAT's action terms and imaginary observations
are not implemented.
"""

from __future__ import annotations

from collections.abc import Sequence
from types import ModuleType
from typing import Any

import numpy as np
import torch

from ..engine_select import get_engine
from ..env.cards import make_generator
from ..env.equity import street_equities

HandRecord = tuple[int, Sequence[int], Any]
"""``(button, deck, terminal GameState)`` of one played hand."""


def replay_streets(
    record: HandRecord, config: Any, engine: ModuleType
) -> tuple[list[tuple[int, int, int]], int | None, int]:
    """Replay one heads-up hand from its deck and action history.

    Returns ``(transitions, allin_street, stake)``: ``transitions`` lists
    ``(old_street, new_street, chips in per player)`` for every street dealt
    while the hand continued; ``allin_street`` is the street on which a
    called all-in ended the hand before the river (None otherwise, including
    folds and river showdowns); ``stake`` is the chips each seat had in at the
    end (the amount a showdown moves).
    """
    button, deck, state = record
    start = [int(x) for x in config.stacks]
    s = engine.GameState.new_hand(config, int(button), [int(c) for c in deck])
    transitions: list[tuple[int, int, int]] = []
    for _, _, action in state.history:
        before = s.street
        s.apply(action)
        if not s.is_terminal and s.street != before:
            transitions.append((int(before), int(s.street), start[0] - int(s.stacks[0])))
    if not s.is_terminal or list(s.payoffs()) != list(state.payoffs()):
        raise RuntimeError("replaying the action history did not reproduce the hand")
    stake = min(start[q] - int(state.stacks[q]) for q in (0, 1))
    allin = None
    if not any(state.folded):
        last = int(state.history[-1][0]) if state.history else 0
        if last < 3:
            allin = last
    return transitions, allin, stake


@torch.no_grad()
def luck_adjusted(
    hands: Sequence[HandRecord],
    config: Any,
    engine: ModuleType | None = None,
    preflop_samples: int = 2048,
    seed: int = 0,
    device: torch.device | str = "cpu",
) -> np.ndarray:
    """``[n, 2]`` luck-adjusted chip results of heads-up ``hands`` (float).

    Rows are zero-sum like the raw payoffs. Unbiased: the mean over many hands
    estimates the same win rate as the raw results, with less variance.
    """
    engine = engine or get_engine()
    if config.num_players != 2:
        raise ValueError("luck adjustment is heads-up only")
    n = len(hands)
    out = np.zeros((n, 2))
    if n == 0:
        return out
    info = [replay_streets(h, config, engine) for h in hands]
    raw = np.array([float(h[2].payoffs()[0]) for h in hands])
    need = [i for i, (tr, allin, _) in enumerate(info) if tr or allin is not None]
    adj0 = raw.copy()
    if need:
        # one equity row per distinct deal: the two seatings of a duplicate deal
        # share it, so its Monte Carlo noise cancels like the cards do
        rows: dict[tuple[int, ...], int] = {}
        row_of = []
        for i in need:
            deal = tuple(int(c) for c in hands[i][1][:9])
            board = [int(c) for c in hands[i][2].board]
            if board != list(deal[4 : 4 + len(board)]):
                raise RuntimeError("board does not follow the deck order")
            row_of.append(rows.setdefault(deal, len(rows)))
        dev = torch.device(device)
        decks = torch.tensor(list(rows), dtype=torch.long, device=dev)
        eq = street_equities(
            decks[:, 0:2], decks[:, 2:4], decks[:, 4:9], preflop_samples, make_generator(seed, dev)
        )
        E = eq.double().cpu().numpy()
        for row, i in zip(row_of, need, strict=True):
            transitions, allin, stake = info[i]
            v = raw[i] if allin is None else stake * (2.0 * E[row, allin] - 1.0)
            for old, new, c in transitions:
                v -= 2.0 * c * (E[row, new] - E[row, old])
            adj0[i] = v
    out[:, 0] = adj0
    out[:, 1] = -adj0
    return out
