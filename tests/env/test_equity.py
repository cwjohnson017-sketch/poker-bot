import itertools

import torch

from pokerbot.env.cards import NO_CARD, cards_from_str, make_generator, shuffled_decks
from pokerbot.env.equity import equity_histogram, equity_river, equity_vs_random, sample_unknown

from .naive_eval import naive7


def _brute_river(hole, board):
    used = set(hole) | set(board)
    rest = [c for c in range(52) if c not in used]
    me = naive7(list(hole) + list(board))
    score = 0.0
    n = 0
    for o in itertools.combinations(rest, 2):
        v = naive7(list(o) + list(board))
        score += 1.0 if me > v else 0.5 if me == v else 0.0
        n += 1
    assert n == 990
    return score / n


def test_equity_river_exact_matches_brute_force():
    d = shuffled_decks(3, make_generator(1))
    hole, board = d[:, :2], d[:, 2:7]
    fast = equity_river(hole, board)
    for i in range(3):
        assert abs(float(fast[i]) - _brute_river(hole[i].tolist(), board[i].tolist())) < 1e-6
    # chunking does not change results
    assert torch.equal(equity_river(hole, board, max_rows=990), fast)


def test_nuts_and_board_plays():
    hole = torch.tensor([cards_from_str("AsKs"), cards_from_str("2c3d")])
    board = torch.tensor([cards_from_str("QsJsTs4h5h"), cards_from_str("AhKhQhJhTh")])
    eq = equity_river(hole, board)
    assert float(eq[0]) == 1.0  # royal flush
    assert float(eq[1]) == 0.5  # royal on board: everyone splits


def test_preflop_equity_vs_random():
    g = make_generator(0)
    hole = torch.tensor([cards_from_str("AsAh")] * 8 + [cards_from_str("7c2d")] * 8)
    eq = equity_vs_random(hole, None, n_samples=4000, generator=g)
    aa, sevtwo = float(eq[:8].mean()), float(eq[8:].mean())
    assert abs(aa - 0.852) < 0.01
    assert abs(sevtwo - 0.346) < 0.012


def test_monte_carlo_matches_exact_on_river():
    d = shuffled_decks(64, make_generator(2))
    hole, board = d[:, :2], d[:, 2:7]
    exact = equity_river(hole, board)
    mc = equity_vs_random(hole, board, n_samples=3000, generator=make_generator(3), max_rows=50_000)
    assert (mc - exact).abs().max() < 0.05
    assert (mc - exact).abs().mean() < 0.012


def test_mixed_streets_and_sampling_excludes_known():
    d = shuffled_decks(256, make_generator(4))
    hole = d[:, :2]
    board = d[:, 2:7].clone()
    blen = torch.tensor([0, 3, 4, 5]).repeat(64)
    board[torch.arange(5)[None, :] >= blen[:, None]] = NO_CARD
    known = torch.cat([hole, board], 1)
    s = sample_unknown(known, 7, make_generator(5))
    assert (s < 52).all()
    assert not (s[:, :, None] == known[:, None, :]).any()
    assert all(len(set(row)) == 7 for row in s.tolist())
    eq = equity_vs_random(hole, board, n_samples=64, generator=make_generator(6))
    assert eq.shape == (256,) and ((eq >= 0) & (eq <= 1)).all()


def test_histogram():
    d = shuffled_decks(32, make_generator(7))
    hole, board = d[:, :2], d[:, 2:7]
    # complete board: a single bin holding the exact equity
    h = equity_histogram(hole, board, n_runouts=4, generator=make_generator(8))
    exact = equity_river(hole, board)
    assert torch.allclose(h.sum(1), torch.ones(32))
    assert torch.equal(h.argmax(1), (exact * 10).long().clamp(max=9))
    assert (h.max(1).values == 1).all()
    # flop: mean of runout equities approximates Monte Carlo equity
    flop = torch.cat([board[:, :3], torch.full((32, 2), NO_CARD)], 1)
    h, mean_eq = equity_histogram(
        hole, flop, n_runouts=200, generator=make_generator(9), return_equity=True
    )
    mc = equity_vs_random(hole, flop, n_samples=4000, generator=make_generator(10))
    assert torch.allclose(h.sum(1), torch.ones(32))
    assert (mean_eq - mc).abs().max() < 0.06
    # sampled-opponent variant has the same shape
    h2 = equity_histogram(hole, board[:, :4], n_runouts=8, n_opp=16, generator=make_generator(11))
    assert h2.shape == (32, 10)
