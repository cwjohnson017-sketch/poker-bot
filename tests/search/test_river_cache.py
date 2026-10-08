"""Continual resolving below turn-end value leaves: the per-river-card values
of a river net averaged over the river cards."""

from __future__ import annotations

import pytest
import torch

from pokerbot.engine_select import get_engine
from pokerbot.env.actions import ActionSpec
from pokerbot.search import value_ranges as vrg
from pokerbot.search.abstract import make_state
from pokerbot.search.combos import NUM_COMBOS, blocked_sum, valid_mask
from pokerbot.search.tree import TreeConfig, build_tree
from pokerbot.search.value_leaf import ShowdownOracle, ValueLeafEvaluator

C = NUM_COMBOS
BIG = (("fold",), ("check_call",), ("raise", 1.0), ("allin",))
PASSIVE_STREET = (("fold",), ("check_call",))
SPEC = ActionSpec(streets=(BIG, BIG, BIG, PASSIVE_STREET), max_raises=2)
BOARD = [4, 9, 14, 19, 24]


class _ToyRiver:
    """A nonlinear river 'net' of both ranges, the board, ``c`` and ``stack`` (0 on
    combos that hit the board)."""

    def predict(self, boards, ranges, c, stack):
        valid = vrg.board_valid(boards.to(ranges.device))[:, None, :]
        r = ranges / ranges.sum(-1, keepdim=True).clamp(min=1e-30)
        z = 300.0 * r - 200.0 * r.flip(1) + (boards.sum(1).to(r.dtype) / 200.0)[:, None, None]
        z = z + (c.to(r.dtype) / 1000.0 - stack.to(r.dtype) / 20000.0)[:, None, None]
        z = z + torch.tensor([0.1, -0.2], dtype=r.dtype, device=r.device)[None, :, None]
        return torch.tanh(z) * valid


# --------------------------------------------------------------------------- per-card values


@pytest.mark.parametrize("river", [ShowdownOracle, _ToyRiver])
def test_card_values_are_the_terms_of_the_river_average(river):
    """``card_values`` gives the per-river-card values whose chance average (weight
    1/44, the solver's ``1 / (52 - 4 - 4)``) is ``values()``, for an exact and a
    nonlinear predictor, leaves on several turn boards, a subset in any order."""
    engine = get_engine()
    cfg = engine.GameConfig(num_players=2, stacks=[1000] * 2, small_blind=50, big_blind=100)
    s = make_state(engine, cfg, 0, BOARD, [])
    s.apply(engine.Action.raise_to(250))
    s.apply(engine.Action.check_call())
    tc = TreeConfig(
        spec=SPEC, depth_streets=1, max_nodes=3000, chance_cards=3, leaf_mode="value_net"
    )
    tree = build_tree(cfg, s.button, s.board, s.history, tc)  # flop tree: 3 turn boards
    vl = ValueLeafEvaluator(tree, river(), chunk=48 * 3)  # several chunks
    L = vl.num_leaves
    assert len(vl.boards4) == 3 and L > 6
    g = torch.Generator().manual_seed(4)
    reach = torch.rand(2, L, C, generator=g, dtype=torch.float64) ** 2  # also on the board
    reach[:, 0] = 0.0  # an unreached leaf
    leaves = torch.cat([torch.tensor([0]), 1 + torch.randperm(L - 1, generator=g)[: L // 2]])
    assert vl.cards_per_pair == 44
    rows = vl.net_rows
    for p in (0, 1):
        want = vl.values(p, reach)
        cards, v = vl.card_values(p, reach, leaves)
        assert v.shape == (len(leaves), 48, C) and v.dtype == torch.float64
        assert torch.equal(cards, vl.cards[leaves])
        scale = float(want.abs().max())
        torch.testing.assert_close(v.sum(1) / 44, want[leaves], rtol=1e-9, atol=1e-9 * scale)
        assert bool((v[0] == 0).all())  # unreached leaf
        # two terms from the definition
        for i, j in ((1, 0), (len(leaves) - 1, 31)):
            leaf, x = int(leaves[i]), int(cards[i, j])
            b4 = list(vl.boards4[int(vl.leaf_board[leaf])])
            assert x not in b4
            ok = valid_mask([x]).double() * valid_mask(b4)
            r = reach[:, leaf] * ok  # both reaches at the river root b4 + x
            o = vl.oop_seat
            ev = river().predict(
                torch.tensor([[*b4, x]]),
                r[[o, 1 - o]][None],
                vl.c[leaf : leaf + 1].double(),
                vl.stack[leaf : leaf + 1].double(),
            )[0, 0 if p == o else 1]
            ref = ok * blocked_sum(r[1 - p]) * (2 * int(vl.c[leaf])) * ev
            torch.testing.assert_close(v[i, j], ref, rtol=1e-9, atol=1e-9 * scale)
    assert vl.net_rows - rows == 2 * (L + len(leaves)) * 48  # values() + the subset's rows
    cards, v = vl.card_values(1, reach)  # every leaf by default
    assert v.shape == (L, 48, C) and torch.equal(cards, vl.cards)
    assert vl.card_values(0, reach, [])[1].shape == (0, 48, C)
