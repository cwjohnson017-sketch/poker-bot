from itertools import combinations

import numpy as np
import pytest
from helpers import naive_best, naive_rank5

from pokerbot.reference import (
    card_from_str,
    card_to_str,
    cards_from_str,
    evaluate5,
    evaluate6,
    evaluate7,
    evaluate_batch,
    hand_category,
)


def test_card_strings_roundtrip():
    for c in range(52):
        assert card_from_str(card_to_str(c)) == c
    assert card_from_str("2c") == 0
    assert card_from_str("As") == 51
    assert card_from_str("Td") == 8 * 4 + 1
    with pytest.raises(ValueError):
        card_from_str("1x")


@pytest.mark.parametrize(
    "hand,cat",
    [
        ("AsKsQsJsTs", 8),
        ("5d4d3d2dAd", 8),
        ("9c9d9h9s2c", 7),
        ("KcKdKh2s2c", 6),
        ("Ah9h7h4h2h", 5),
        ("Ac2d3h4s5c", 4),
        ("TcJdQhKsAc", 4),
        ("7c7d7h2s3c", 3),
        ("7c7d2h2s3c", 2),
        ("7c7d2h4s3c", 1),
        ("Ac9d7h4s3c", 0),
    ],
)
def test_categories_5(hand, cat):
    assert hand_category(evaluate5(cards_from_str(hand))) == cat


def test_known_orderings():
    e = lambda s: evaluate7(cards_from_str(s))  # noqa: E731
    # wheel straight loses to six-high straight
    assert e("Ac2d3h4s5c9dKh") < e("6c2d3h4s5c9dKh")
    # board plays: split
    assert e("2c3dAsKsQsJsTs") == e("4h5hAsKsQsJsTs")
    # kicker matters
    assert e("AhKd2c7s9dTh3c") > e("AhQd2c7s9dTh3c")
    # two pair: third pair cannot count, best kicker wins
    assert e("AcAdKcKd2c2dQh") > e("AcAdKcKd2c2dJh")
    # full house: trips over trips picks best pair
    assert e("AcAdAhKcKdKh2c") > e("AcAdAhQcQdQh2c")
    # flush vs straight
    assert e("2h5h9hJhKh3c4d") > e("9cTdJhQsKc2c3d")
    # 6-card
    assert hand_category(evaluate6(cards_from_str("AsKsQsJsTs2d"))) == 8


def test_bad_input():
    with pytest.raises(ValueError):
        evaluate7(cards_from_str("AsKsQsJsTs2d"))
    with pytest.raises(ValueError):
        evaluate5([0, 0, 1, 2, 3])


def _sign(x):
    return (x > 0) - (x < 0)


def test_evaluator_matches_naive_on_random_hands():
    rng = np.random.default_rng(1)
    hands = [[int(c) for c in rng.permutation(52)[:7]] for _ in range(3000)]
    mine = [evaluate7(h) for h in hands]
    naive = [naive_best(h) for h in hands]
    for m, nv in zip(mine, naive, strict=True):
        assert hand_category(m) == nv[0]
    for i in range(0, len(hands) - 1):
        a, b = i, i + 1
        assert _sign(mine[a] - mine[b]) == _sign((naive[a] > naive[b]) - (naive[a] < naive[b]))


def test_evaluator_matches_naive_same_board():
    """Pairs sharing a board: ties and near-ties are frequent here."""
    rng = np.random.default_rng(2)
    for _ in range(1500):
        d = [int(c) for c in rng.permutation(52)]
        board = list(d[4:9])
        h1, h2 = list(d[:2]) + board, list(d[2:4]) + board
        m = _sign(evaluate7(h1) - evaluate7(h2))
        n1, n2 = naive_best(h1), naive_best(h2)
        assert m == (n1 > n2) - (n1 < n2)


def test_batch_matches_scalar():
    rng = np.random.default_rng(3)
    for k, fn in ((5, evaluate5), (6, evaluate6), (7, evaluate7)):
        arr = np.array([rng.permutation(52)[:k] for _ in range(5000)], dtype=np.uint8)
        batch = evaluate_batch(arr)
        assert batch.dtype == np.int32 and batch.shape == (5000,)
        assert all(int(b) == fn(list(r)) for b, r in zip(batch, arr, strict=True))


def test_batch_covers_every_category():
    hands = [
        "AsKsQsJsTs2d3c",
        "9c9d9h9s2c3d4h",
        "KcKdKh2s2c3d4h",
        "Ah9h7h4h2h3c3d",
        "Ac2d3h4s5c9dKh",
        "7c7d7h2s3cJdQh",
        "7c7d2h2s3c3dAh",
        "7c7d2h4s9cJdKh",
        "Ac9d7h4s3cJhTd",
    ]
    arr = np.array([cards_from_str(h) for h in hands], dtype=np.uint8)
    batch = evaluate_batch(arr)
    assert [hand_category(int(b)) for b in batch] == [8, 7, 6, 5, 4, 3, 2, 1, 0]
    assert all(int(b) == evaluate7(cards_from_str(h)) for b, h in zip(batch, hands, strict=True))


def test_five_card_naive_orderings_sample():
    rng = np.random.default_rng(4)
    hands = [[int(c) for c in rng.permutation(52)[:5]] for _ in range(4000)]
    ranked = sorted(hands, key=evaluate5)
    naive = [naive_rank5(h) for h in ranked]
    assert all(naive[i] <= naive[i + 1] for i in range(len(naive) - 1))


@pytest.mark.slow
def test_full_five_card_enumeration_category_counts():
    combos = np.array(list(combinations(range(52), 5)), dtype=np.uint8)
    assert len(combos) == 2598960
    cats = evaluate_batch(combos) >> 20
    counts = np.bincount(cats, minlength=9)
    expected = [1302540, 1098240, 123552, 54912, 10200, 5108, 3744, 624, 40]
    assert counts.tolist() == expected
    # distinct hand values: 7462 equivalence classes
    assert len(np.unique(evaluate_batch(combos))) == 7462
