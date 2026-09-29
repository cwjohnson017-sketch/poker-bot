import itertools

import numpy as np
import pytest
import torch

from pokerbot.env.cards import card_from_str, card_to_str, cards_from_str, shuffled_decks, make_generator
from pokerbot.env.evaluator import (
    CATEGORY_SHIFT,
    evaluate5,
    evaluate6,
    evaluate7,
    evaluate7_batch,
    evaluate_batch,
    hand_category,
)

from .naive_eval import B5, naive5, naive5_np, naive7, naive7_np


def test_card_strings_roundtrip():
    assert card_from_str("2c") == 0
    assert card_from_str("As") == 51
    assert card_from_str("Td") == 8 * 4 + 1
    assert [card_to_str(c) for c in range(52)] == [r + s for r in "23456789TJQKA" for s in "cdhs"]
    assert cards_from_str("AsKd") == [51, 11 * 4 + 1]
    with pytest.raises(ValueError):
        card_from_str("1c")


def test_decks_are_permutations():
    d = shuffled_decks(512, make_generator(3))
    assert d.shape == (512, 52)
    assert (d.sort(1).values == torch.arange(52)).all()
    # every card appears in every deal position with roughly uniform frequency
    counts = torch.zeros(52, 52)
    d = shuffled_decks(20000, make_generator(4))
    counts.index_put_((torch.arange(52).expand(20000, 52), d), torch.ones(20000, 52), accumulate=True)
    assert counts.min() > 250 and counts.max() < 520


def test_all_five_card_hands():
    combos = torch.combinations(torch.arange(52), 5)
    r = evaluate_batch(combos)
    counts = torch.bincount(hand_category(r), minlength=9).tolist()
    assert counts == [1302540, 1098240, 123552, 54912, 10200, 5108, 3744, 624, 40]
    assert r.unique().numel() == 7462


def _assert_order_consistent(mine: np.ndarray, ref: np.ndarray) -> None:
    order = np.argsort(ref, kind="stable")
    m, v = mine[order], ref[order]
    dv, dm = np.diff(v), np.diff(m)
    assert np.all(dm[dv > 0] > 0), "strictly better reference hand not ranked higher"
    assert np.all(dm[dv == 0] == 0), "equal reference hands ranked differently"


def test_random_seven_card_hands_vs_naive():
    g = make_generator(123)
    cards = shuffled_decks(200_000, g)[:, :7]
    mine = evaluate7_batch(cards).numpy()
    ref = naive7_np(cards.numpy())
    _assert_order_consistent(mine, ref)
    assert np.array_equal(mine >> CATEGORY_SHIFT, ref // B5)


def test_rare_categories_vs_naive():
    # hands built from few ranks / one suit exercise quads, full houses and flushes
    g = torch.Generator().manual_seed(7)
    n = 50_000
    ranks = torch.randint(0, 13, (n, 1), generator=g) + torch.randint(0, 3, (n, 7), generator=g)
    ranks = ranks.clamp(max=12)
    suits = torch.randint(0, 4, (n, 7), generator=g)
    cards = ranks * 4 + suits
    ok = torch.tensor([len(set(row)) == 7 for row in cards.tolist()])
    cards = cards[ok]
    mine = evaluate7_batch(cards).numpy()
    ref = naive7_np(cards.numpy())
    _assert_order_consistent(mine, ref)
    cats = np.bincount(ref // B5, minlength=9)
    assert cats[6] > 100 and cats[7] > 10


def test_scalar_matches_pure_python():
    g = make_generator(9)
    decks = shuffled_decks(1500, g)
    hands = decks[:, :7].tolist()
    mine = np.array([evaluate7(h) for h in hands])
    ref = np.array([naive7(h) for h in hands])
    _assert_order_consistent(mine, ref)
    six = decks[:1500, :6].tolist()
    mine6 = np.array([evaluate6(h) for h in six])
    ref6 = np.array([max(naive5(c) for c in itertools.combinations(h, 5)) for h in six])
    _assert_order_consistent(mine6, ref6)
    five = decks[:1500, :5].tolist()
    _assert_order_consistent(np.array([evaluate5(h) for h in five]), np.array([naive5(h) for h in five]))


def test_named_hands():
    def ev(s):
        return evaluate7(cards_from_str(s))

    royal = ev("AsKsQsJsTs2c3d")
    steel_wheel = ev("As2s3s4s5sKdQd")
    quads = ev("AcAdAhAsKs2c3d")
    boat = ev("KcKdKhAsAd2c3d")
    flush = ev("2h5h9hJhKh3c4d")
    broadway = ev("AcKdQhJsTc2c3d")
    wheel = ev("Ac2d3h4s5c9d9h")  # wheel beats the pair of nines
    six_high = ev("2c3d4h5s6c9dJh")
    assert hand_category(royal) == 8 and hand_category(steel_wheel) == 8
    assert royal > steel_wheel > quads > boat > flush > broadway > six_high > wheel
    assert hand_category(wheel) == 4
    # two trips -> full house using the higher trips
    assert hand_category(ev("KcKdKh2c2d2hAs")) == 6
    assert ev("KcKdKh2c2d2hAs") == ev("KcKdKh2c2dQhAs")  # kings full of twos either way
    assert ev("KcKdKh2c2d2hAs") < ev("KcKdKh3c3d2hAs")
    # three pairs: best two pairs plus best kicker (the third pair's rank counts)
    assert ev("AcAdKcKdQcQd2h") == ev("AhAsKhKsQh3c2d")
    assert ev("AcAdKcKdQcQd2h") > ev("AcAdKcKdJcJd2h")
    # board plays: split
    assert ev("2c3dAsKsQsJsTs") == ev("4c5dAsKsQsJsTs")
    # kickers matter for one pair, not beyond five cards
    assert ev("AcAd9s8c7h3d2c") > ev("AhAs9d8h6c3c2d")
    assert ev("AcAd9s8c7h3d2c") == ev("AhAs9d8h7c4c2d")


def test_batch_shapes_and_category_helper():
    x = shuffled_decks(12, make_generator(1))[:, :7].view(3, 4, 7)
    r = evaluate_batch(x)
    assert r.shape == (3, 4)
    assert torch.equal(r.flatten(), evaluate7_batch(x.view(12, 7)))
    assert hand_category(int(r[0, 0])) == int(hand_category(r)[0, 0])
    with pytest.raises(ValueError):
        evaluate7([1, 1, 2, 3, 4, 5, 6])


def test_naive5_np_matches_scalar():
    decks = shuffled_decks(3000, make_generator(5))[:, :5].numpy()
    assert list(naive5_np(decks)) == [naive5(list(h)) for h in decks]
