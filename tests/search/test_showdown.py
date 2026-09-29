"""(a) The sorted-strength showdown trick and the fold kernel equal the naive products."""

from __future__ import annotations

import itertools

import torch

from pokerbot.search.combos import (
    NUM_COMBOS,
    combo_cards,
    combo_index,
    conflict_matrix,
    valid_mask,
)
from pokerbot.search.showdown import ShowdownTables, fold_values, naive_showdown


def test_combo_index_roundtrip():
    for i, (a, b) in enumerate(itertools.combinations(range(52), 2)):
        assert combo_index(a, b) == i
        assert combo_index(b, a) == i
        assert combo_cards(i) == (a, b)
    assert i + 1 == NUM_COMBOS


def _random_setup(seed: int, rows: int):
    g = torch.Generator().manual_seed(seed)
    boards = [torch.randperm(52, generator=g)[:5].tolist() for _ in range(3)]
    reach = torch.rand(rows, NUM_COMBOS, generator=g, dtype=torch.float64)
    # sparse ranges exercise ties and empty tie groups
    reach = reach * (torch.rand(rows, NUM_COMBOS, generator=g) < 0.6)
    bid = torch.randint(0, 3, (rows,), generator=g)
    return boards, reach, bid


def test_showdown_trick_matches_naive_product():
    boards, reach, bid = _random_setup(0, 9)
    # a paired, a flushy and a straight-heavy board make many ties
    boards += [[0, 1, 2, 16, 32], [48, 44, 40, 36, 3], [12, 16, 20, 24, 51]]
    bid = torch.cat([bid, torch.tensor([3, 4, 5])])
    reach = torch.cat([reach, torch.rand(3, NUM_COMBOS, dtype=torch.float64)])
    tables = ShowdownTables(boards)
    fast = tables.showdown(reach, bid)
    for i in range(len(bid)):
        ref = naive_showdown(reach[i : i + 1], boards[int(bid[i])])[0]
        assert torch.allclose(fast[i], ref, atol=1e-9), (i, (fast[i] - ref).abs().max())


def test_fold_values_match_naive_product():
    boards, reach, bid = _random_setup(1, 4)
    ok_pairs = ~conflict_matrix()
    for i in range(4):
        valid = valid_mask(boards[int(bid[i])])
        fast = fold_values(reach[i : i + 1], valid[None])[0]
        r = reach[i] * valid
        ref = (ok_pairs.double() @ r) * valid
        assert torch.allclose(fast, ref, atol=1e-9)
