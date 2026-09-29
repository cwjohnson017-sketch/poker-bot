"""Torch helpers around the Rust canonical index."""

from __future__ import annotations

import itertools
import math

import numpy as np
import poker_engine as pe
import pytest
import torch

from pokerbot.abstraction.isomorphism import (
    BOARD_LEN,
    cache_path,
    canonical_index_batch,
    enumerate_canonical,
    orbit_sizes,
    representatives,
    unindex_batch,
)


def random_hands(street: int, n: int, rng: np.random.Generator) -> np.ndarray:
    return np.stack([rng.permutation(52)[: 2 + BOARD_LEN[street]] for _ in range(n)])


def permute_suits(cards: np.ndarray, perm: np.ndarray) -> np.ndarray:
    return (cards // 4) * 4 + perm[cards % 4]


@pytest.mark.parametrize("street", range(4))
def test_suit_permutations_and_reordering_keep_the_index(street):
    rng = np.random.default_rng(street)
    hands = random_hands(street, 500, rng)
    idx = canonical_index_batch(street, torch.from_numpy(hands))
    assert idx.dtype == torch.long and idx.shape == (500,)
    assert int(idx.max()) < pe.canonical_size(street)
    assert idx.tolist() == [
        pe.canonical_index(street, h[:2].tolist(), h[2:].tolist()) for h in hands
    ]
    for _ in range(4):
        perm = rng.permutation(4)
        other = permute_suits(hands, perm)
        other[:, :2] = other[:, 1::-1]  # swap the hole cards
        if street:
            other[:, 2:] = rng.permuted(other[:, 2:], axis=1)  # shuffle the board
        assert torch.equal(canonical_index_batch(street, other), idx)
    # The representative re-indexes to the same value.
    reps = unindex_batch(street, idx.numpy())
    assert torch.equal(canonical_index_batch(street, reps), idx)


@pytest.mark.parametrize("street", range(4))
def test_unindex_roundtrip_of_random_indices(street):
    rng = np.random.default_rng(10 + street)
    idx = rng.integers(0, pe.canonical_size(street), 2000)
    reps = unindex_batch(street, idx)
    assert reps.dtype == np.uint8 and reps.shape == (2000, 2 + BOARD_LEN[street])
    assert np.array_equal(canonical_index_batch(street, reps).numpy(), idx)
    # Every row holds distinct cards, each round sorted ascending.
    assert all(len(set(r)) == len(r) for r in reps.tolist())
    assert (np.diff(reps[:, :2].astype(int), axis=1) > 0).all()
    if street:
        assert (np.diff(reps[:, 2:].astype(int), axis=1) > 0).all()


def test_enumerate_canonical_is_cached_and_complete(tmp_path):
    pre = enumerate_canonical(0, tmp_path)
    assert pre.shape == (169, 2) and pre.dtype == torch.uint8
    assert cache_path(0, tmp_path).exists()
    assert torch.equal(canonical_index_batch(0, pre), torch.arange(169))
    flop = enumerate_canonical(1, tmp_path)
    assert flop.shape == (pe.canonical_size(1), 5)
    # A second call loads the file; representatives() gathers from it.
    assert torch.equal(enumerate_canonical(1, tmp_path)[:1000], flop[:1000])
    idx = np.array([0, 17, 123_456, pe.canonical_size(1) - 1])
    assert np.array_equal(representatives(1, idx, tmp_path), flop[idx].numpy())
    assert np.array_equal(representatives(1, idx, tmp_path), unindex_batch(1, idx))
    # All flop classes index back to themselves; orbit sizes add up to every
    # (hole set, board set) combination.
    assert torch.equal(canonical_index_batch(1, flop), torch.arange(flop.shape[0]))
    w = orbit_sizes(1, flop)
    assert int(w.sum()) == math.comb(52, 2) * math.comb(50, 3)
    assert int(orbit_sizes(0, pre).sum()) == 1326


def _orbit_bruteforce(street: int, hand: list[int]) -> int:
    images = set()
    for perm in itertools.permutations(range(4)):
        img = [(c // 4) * 4 + perm[c % 4] for c in hand]
        images.add((frozenset(img[:2]), frozenset(img[2:])))
    return len(images)


@pytest.mark.parametrize("street", [2, 3])
def test_orbit_sizes_match_bruteforce(street):
    rng = np.random.default_rng(3)
    idx = rng.integers(0, pe.canonical_size(street), 300)
    reps = unindex_batch(street, idx)
    w = orbit_sizes(street, torch.from_numpy(reps)).tolist()
    assert w == [_orbit_bruteforce(street, r) for r in reps.tolist()]
    # Any member of the class gives the same orbit size.
    hands = random_hands(street, 200, rng)
    assert orbit_sizes(street, torch.from_numpy(hands)).tolist() == [
        _orbit_bruteforce(street, h) for h in hands.tolist()
    ]
