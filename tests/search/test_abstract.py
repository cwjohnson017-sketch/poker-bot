"""Scalar abstract actions agree with the tensor code in pokerbot.env.actions,
and range reconstruction follows the blueprint."""

from __future__ import annotations

import numpy as np
import torch

from pokerbot.engine_select import get_engine
from pokerbot.env.actions import DEFAULT_SPEC, legal_mask, raise_targets
from pokerbot.search.abstract import (
    CardView,
    legal_options,
    make_state,
    map_concrete,
    raises_this_street,
)
from pokerbot.search.blueprint import TabularBlueprintFromCallable, range_reach
from pokerbot.search.combos import combo_cards, combo_index


def _tensor_view(state, spec):
    tab = spec.tables("cpu")
    p = state.current_player
    bets = list(state.street_bets)
    la = state.legal_actions()
    t = lambda x: torch.tensor([int(x)])  # noqa: E731
    street = t(state.street)
    max_bet = max(bets)
    to_call = max_bet - bets[p]
    targets = raise_targets(
        tab, street, t(state.pot), t(max_bet), t(to_call), t(la.min_raise_to), t(la.max_raise_to)
    )
    mask = legal_mask(
        tab,
        street,
        torch.tensor([True]),
        t(to_call),
        torch.tensor([la.min_raise_to > 0]),
        t(raises_this_street(state)),
        targets,
        t(la.max_raise_to),
    )
    return mask[0].tolist(), targets[0].tolist()


def test_legal_options_match_env_actions():
    engine = get_engine()
    rng = np.random.default_rng(0)
    checked = 0
    for hand in range(150):
        stacks = [int(rng.integers(300, 5000)), int(rng.integers(300, 5000))]
        cfg = engine.GameConfig(num_players=2, stacks=stacks, small_blind=50, big_blind=100)
        s = engine.GameState.new_hand(cfg, hand % 2, rng.permutation(52).tolist())
        while not s.is_terminal:
            opts = legal_options(s, DEFAULT_SPEC)
            mask, targets = _tensor_view(s, DEFAULT_SPEC)
            assert [o.index for o in opts] == [i for i, m in enumerate(mask) if m]
            for o in opts:
                if o.kind == 2:
                    assert o.amount == targets[o.index]
                    assert s.legal_actions().is_legal(engine.Action.raise_to(o.amount))
            o = opts[int(rng.integers(len(opts)))]
            s.apply(
                engine.Action.raise_to(o.amount)
                if o.kind == 2
                else (engine.Action.fold() if o.kind == 0 else engine.Action.check_call())
            )
            checked += 1
    assert checked > 300


def test_pseudo_harmonic_mapping():
    engine = get_engine()
    cfg = engine.GameConfig()
    s = make_state(engine, cfg, 0, [0, 1, 2], [])
    s.apply(engine.Action.check_call())
    s.apply(engine.Action.check_call())  # flop, pot 200, BB acts first
    opts = legal_options(s, DEFAULT_SPEC)
    by_idx = {o.index: o.amount for o in opts}
    assert map_concrete(s, DEFAULT_SPEC, 2, by_idx[3]) == {3: 1.0}
    m = map_concrete(s, DEFAULT_SPEC, 2, 110)  # 0.55 pot, between 0.33 and 0.75
    assert set(m) == {2, 3} and abs(sum(m.values()) - 1) < 1e-9
    a, b, x = 100 / 200, 150 / 200, 110 / 200  # 0.33 pot is clamped to the 100 min bet
    assert abs(m[2] - (b - x) * (1 + a) / ((b - a) * (1 + x))) < 1e-9
    assert map_concrete(s, DEFAULT_SPEC, 1, 0) == {1: 1.0}


def test_card_view_overrides_cards():
    engine = get_engine()
    cfg = engine.GameConfig()
    s = make_state(engine, cfg, 0, [0, 1, 2, 3, 4], [])
    for _ in range(3):
        s.apply(engine.Action.check_call())
    view = CardView(s, [10, 11, 12, 13, 14]).with_hole(1, [40, 41])
    assert view.board == [10, 11, 12]
    assert view.hole_cards(1) == [40, 41] and view.hole_cards(0) == []
    assert view.infoset_key(1)[-3:] == bytes([1, 40, 41])
    assert view.public_key()[2:5] == bytes([10, 11, 12])
    plain = CardView(s)
    assert plain.public_key() == s.public_key()


def test_range_reach_follows_a_card_dependent_blueprint():
    engine = get_engine()
    cfg = engine.GameConfig()

    def pairs_raise(state, player):
        h = state.hole_cards(player)
        opts = {o.index: o for o in legal_options(state, DEFAULT_SPEC)}
        pair = h[0] // 4 == h[1] // 4
        if pair and 2 in opts:
            return {2: 1.0}
        return {1: 1.0}

    bp = TabularBlueprintFromCallable(pairs_raise)
    s = make_state(engine, cfg, 0, [], [])
    s.apply(engine.Action.raise_to(250))  # button raises 2.5x: only pairs do that
    s.apply(engine.Action.check_call())
    board = [4, 9, 14]
    r = range_reach(bp, cfg, 0, board, s.history, engine, exclude={1: [20]})
    button_pairs = r[0] > 0
    for c in range(1326):
        a, b = combo_cards(c)
        is_pair = a // 4 == b // 4
        on_board = a in board or b in board
        assert bool(button_pairs[c]) == (is_pair and not on_board)
    # BB calls with non-pairs (it never raises), and loses combos blocked by the exclusion
    assert float(r[1, combo_index(20, 30)]) == 0.0
    assert float(r[1, combo_index(21, 30)]) == 1.0
    assert float(r[1, combo_index(0, 1)]) == 0.0  # a pair would have 3-bet
    assert float(r[1, combo_index(0, 5)]) == 1.0
