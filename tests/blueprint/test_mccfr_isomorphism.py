"""Canonical suit-isomorphic indexing and preflop classes."""

import numpy as np
import poker_engine as pe
import pytest

BOARD_LEN = (0, 3, 4, 5)


def permute(cards, perm):
    return [(c >> 2) * 4 + perm[c & 3] for c in cards]


def test_class_counts():
    assert [pe.canonical_size(s) for s in range(4)] == [169, 1_286_792, 13_960_050, 123_156_254]


@pytest.mark.parametrize("street", range(4))
def test_index_invariant_under_suit_permutation_and_order(street):
    rng = np.random.default_rng(street)
    nb = BOARD_LEN[street]
    for _ in range(300):
        deck = rng.permutation(52)[: 2 + nb].tolist()
        hole, board = deck[:2], deck[2:]
        i = pe.canonical_index(street, hole, board)
        assert 0 <= i < pe.canonical_size(street)
        for _ in range(3):
            perm = rng.permutation(4).tolist()
            h2 = permute(hole, perm)[::-1]
            b2 = permute(board, perm)
            rng.shuffle(b2)
            assert pe.canonical_index(street, h2, b2) == i
        # The canonical representative indexes back to the same class.
        h, b = pe.canonical_unindex(street, i)
        assert pe.canonical_index(street, h, b) == i


def test_different_classes_get_different_indices():
    # Suited vs offsuit, flush draw vs none, paired board vs not.
    c = pe.card_from_str
    aks = pe.canonical_index(0, [c("As"), c("Ks")], [])
    ako = pe.canonical_index(0, [c("As"), c("Kd")], [])
    assert aks != ako
    fd = pe.canonical_index(1, [c("As"), c("Ks")], [c("2s"), c("7s"), c("9d")])
    nofd = pe.canonical_index(1, [c("As"), c("Ks")], [c("2c"), c("7s"), c("9d")])
    assert fd != nofd


def test_batch_matches_scalar():
    rng = np.random.default_rng(1)
    rows = np.stack([rng.permutation(52)[:6] for _ in range(200)]).astype(np.uint8)
    got = pe.canonical_index_batch(2, rows)
    want = [pe.canonical_index(2, r[:2].tolist(), r[2:].tolist()) for r in rows]
    assert got.tolist() == want


def test_preflop_classes():
    seen = {}
    for a in range(52):
        for b in range(a + 1, 52):
            k = pe.preflop_class([a, b])
            seen.setdefault(k, 0)
            seen[k] += 1
    assert len(seen) == 169
    assert sorted(set(seen.values())) == [4, 6, 12]
    c = pe.card_from_str
    assert pe.preflop_class([c("Ah"), c("As")]) == 168
    assert pe.preflop_class([c("Ks"), c("As")]) == 12 * 13 + 11  # AKs
    assert pe.preflop_class([c("Kd"), c("As")]) == 11 * 13 + 12  # AKo


def test_default_buckets_respect_isomorphism():
    ca = pe.CardAbstraction([169, 10, 10, 10], hs_samples=32)
    rng = np.random.default_rng(5)
    for street in (1, 2, 3):
        nb = BOARD_LEN[street]
        for _ in range(20):
            d = rng.permutation(52)[: 2 + nb].tolist()
            perm = rng.permutation(4).tolist()
            b1 = ca.bucket(street, d[:2], d[2:])
            b2 = ca.bucket(street, permute(d[:2], perm), permute(d[2:], perm))
            assert b1 == b2
            assert 0 <= b1 < 10
