import numpy as np
import pytest
import torch

from pokerbot.agents import AlwaysCallAgent
from pokerbot.agents.policy import PolicyAgentBase, legal_vector
from pokerbot.config import game_config
from pokerbot.env import equity_river
from pokerbot.env.cards import cards_from_str
from pokerbot.eval import play_hand
from pokerbot.eval.lbr import (
    COMBOS,
    LBRAgent,
    blocked_mask,
    combo_index,
    main,
    range_equity,
    run_lbr,
)
from tests.helpers import make_deck


def test_combo_index_matches_table():
    for i in (0, 1, 50, 777, 1325):
        a, b = COMBOS[i]
        assert combo_index(a, b) == i and combo_index(b, a) == i


def test_range_equity_matches_exact_river_equity():
    hero = cards_from_str("AhKd")
    board = cards_from_str("2c7h9sTdQc")
    w = (~blocked_mask(hero + board)).astype(float)
    eq = range_equity(hero, board, w / w.sum())
    ref = float(equity_river(torch.tensor([hero]), torch.tensor([board]))[0])
    assert eq == pytest.approx(ref, abs=1e-6)
    # against a single known hand on the turn: enumerate the 44 rivers
    w = np.zeros(len(COMBOS))
    w[combo_index(*cards_from_str("7c7d"))] = 1.0
    assert range_equity(hero, board[:4], w) == 0.0  # AK drawing dead vs a set
    # QJ on 2 7 9 T vs a set: 8 straight outs of 44 rivers
    assert range_equity(cards_from_str("QhJh"), board[:4], w) == pytest.approx(8 / 44)


class PairRaiser(PolicyAgentBase):
    """Raises (smallest abstract raise) with pocket pairs, otherwise checks/calls."""

    name = "pair_raiser"

    def policy(self, state, seat):
        h = state.hole_cards(seat)
        legal = legal_vector(state, self.spec)
        if h[0] // 4 == h[1] // 4:
            for i in range(2, len(legal)):
                if legal[i]:
                    return {i: 1.0}
        return {1: 1.0}


@pytest.mark.parametrize("opp_hole,expect_pairs", [("9c9d", True), ("As4d", False)])
def test_range_update_is_bayesian_and_normalized(opp_hole, expect_pairs):
    cfg = game_config({})
    opp = PairRaiser()
    lbr = LBRAgent(opp, max_runouts=8)
    # seat 0 = opponent on the button (acts first preflop), seat 1 = LBR
    deck = make_deck([opp_hole, "KhQh"], "2c7h3s8d5c")
    final = play_hand(cfg, [opp, lbr], 0, deck, np.random.default_rng(0))
    assert lbr.range is not None
    r = lbr.sync_range(final, 1)  # catch up with the rest of the hand
    dead = cards_from_str("KhQh") + list(final.board)
    assert r.sum() == pytest.approx(1.0)
    assert (r >= 0).all()
    is_pair = COMBOS[:, 0] // 4 == COMBOS[:, 1] // 4
    pair_mass = r[is_pair].sum()
    # likelihoods are floored at 1e-4, so the "impossible" part keeps a sliver
    assert pair_mass > 0.99 if expect_pairs else pair_mass < 0.01
    # no combo using LBR's cards or the board keeps any weight
    assert r[blocked_mask(dead)].sum() == 0.0
    assert r[combo_index(*cards_from_str(opp_hole))] > 0


def test_lbr_beats_always_call_clearly():
    res = run_lbr(AlwaysCallAgent(), hands=150, seed=0, max_runouts=16)
    assert res.hands == 150
    assert res.adj.ci_low > 5000  # mbb/h; a calling station is very exploitable
    assert res.adj.mbb_per_hand > 10000
    rows = {r["street"]: r for r in res.by_street()}
    assert sum(r["hands_ended"] for r in rows.values()) == 150
    assert rows["preflop"]["lbr_actions"] == {"check_call": 150}
    assert sum(r["contribution_mbb"] for r in rows.values()) == pytest.approx(res.adj.mbb_per_hand)
    assert "LBR vs always_call" in res.summary()


def test_lbr_cli(tmp_path, capsys):
    out = tmp_path / "lbr.json"
    argv = ["--opponent", "uniform", "--hands", "20", "--runouts", "8", "--out", str(out)]
    assert main(argv) == 0
    assert "LBR vs uniform" in capsys.readouterr().out
    assert out.exists()
