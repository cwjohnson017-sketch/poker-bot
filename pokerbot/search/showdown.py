"""Vectorised range-vs-range showdown and fold values with card removal.

For a complete 5-card board, every combo gets a strength from
:func:`pokerbot.env.evaluator.evaluate7_batch`. The showdown value of combo
``c`` against an opponent reach vector ``r`` is

    sd(c) = sum_{c' disjoint from c and the board} r(c') * sign(s(c) - s(c'))
          = W(c) - L(c)

and is computed in ``O(n log n)`` once per board plus ``O(n)`` per query
instead of the naive ``1326 x 1326`` product:

* sort the combos by strength once per board; a prefix sum of ``r`` in that
  order gives the mass of all strictly weaker combos at the start of ``c``'s
  tie group (``W_all``) and of all strictly stronger ones after its end;
* card removal: combos sharing a card with ``c`` must not count. Every card
  ``x`` lies in exactly 51 combos; per card we keep those 51 combos sorted by
  strength, and a prefix sum along each card list gives the weaker (stronger)
  mass that holds ``x``. By inclusion-exclusion
  ``W(c) = W_all(c) - W_x1(c) - W_x2(c) + [c itself]``, and ``c`` itself is
  never strictly weaker than ``c``, so the last term is zero.

All index tables are built once per board and every query is a handful of
gathers and cumsums batched over rows (rows may use different boards).
:func:`naive_showdown` is the dense reference used in tests.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

import torch

from ..env.evaluator import evaluate7_batch
from .combos import NUM_COMBOS, blocked_sum, combo_table, conflict_matrix, valid_masks

C = NUM_COMBOS


def combo_strengths(boards: torch.Tensor) -> torch.Tensor:
    """``[B, 5]`` boards -> ``[B, 1326]`` long strengths, ``-1`` for combos that
    conflict with the board."""
    dev = boards.device
    cards = combo_table(dev)
    B = boards.shape[0]
    out = torch.empty(B, C, dtype=torch.long, device=dev)
    step = 128
    for s in range(0, B, step):
        b = boards[s : s + step].long()
        n = b.shape[0]
        seven = torch.cat([b[:, None, :].expand(n, C, 5), cards[None].expand(n, C, 2)], 2)
        out[s : s + n] = evaluate7_batch(seven.reshape(-1, 7)).view(n, C)
    valid = valid_masks(boards.tolist(), dev)
    return torch.where(valid, out, torch.full_like(out, -1))


@lru_cache(maxsize=4)
def _card_lists(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """``[52, 51]`` combos holding each card, and which of the combo's two
    cards (0 = low, 1 = high) that card is."""
    cards = combo_table("cpu")
    lists = torch.empty(52, 51, dtype=torch.long)
    which = torch.empty(52, 51, dtype=torch.long)
    for x in range(52):
        idx = ((cards[:, 0] == x) | (cards[:, 1] == x)).nonzero().flatten()
        lists[x] = idx
        which[x] = (cards[idx, 1] == x).long()
    return lists.to(device), which.to(device)


class ShowdownTables:
    """Per-board index tables for :meth:`showdown` on a list of 5-card boards."""

    def __init__(self, boards: Sequence[Sequence[int]], device: torch.device | str = "cpu"):
        dev = torch.device(device)
        self.device = dev
        self.boards = [tuple(int(c) for c in b) for b in boards]
        NB = len(self.boards)
        self.num_boards = NB
        if NB == 0:
            return
        parts = [self._build(self.boards[i : i + 512], dev) for i in range(0, NB, 512)]
        for name in ("valid", "order", "lo", "hi", "card_list", "cflo", "cfhi"):
            setattr(self, name, torch.cat([p[name] for p in parts]))
        cards = combo_table(dev)
        self.ctot = (cards * 52 + 51).reshape(1, 2 * C)  # end of each card's prefix row

    @staticmethod
    def _build(boards: list[tuple[int, ...]], dev: torch.device) -> dict[str, torch.Tensor]:
        NB = len(boards)
        bt = torch.tensor(boards, dtype=torch.long, device=dev)
        strength = combo_strengths(bt)  # [NB, C]
        s_sorted, order = torch.sort(strength, dim=1, stable=True)
        lo_s = torch.searchsorted(s_sorted, s_sorted, right=False)
        hi_s = torch.searchsorted(s_sorted, s_sorted, right=True)
        lo = torch.empty_like(lo_s).scatter_(1, order, lo_s)
        hi = torch.empty_like(hi_s).scatter_(1, order, hi_s)
        lists, which = _card_lists(dev)
        s_card = strength[:, lists]  # [NB, 52, 51]
        sc_sorted, perm = torch.sort(s_card, dim=2, stable=True)
        card_list = torch.gather(lists[None].expand(NB, -1, -1), 2, perm)
        which_sorted = torch.gather(which[None].expand(NB, -1, -1), 2, perm)
        flat_s = sc_sorted.reshape(NB * 52, 51)
        clo_s = torch.searchsorted(flat_s, flat_s, right=False).view(NB, 52, 51)
        chi_s = torch.searchsorted(flat_s, flat_s, right=True).view(NB, 52, 51)
        base = (torch.arange(52, device=dev) * 52)[None, :, None]
        target = (card_list * 2 + which_sorted).view(NB, -1)
        cflo = torch.empty(NB, 2 * C, dtype=torch.long, device=dev)
        cfhi = torch.empty(NB, 2 * C, dtype=torch.long, device=dev)
        cflo.scatter_(1, target, (clo_s + base).view(NB, -1))
        cfhi.scatter_(1, target, (chi_s + base).view(NB, -1))
        ix = torch.long  # gather indices; int64 avoids a conversion per query
        return {
            "valid": strength >= 0,
            "order": order.to(ix),
            "lo": lo.to(ix),
            "hi": hi.to(ix),
            "card_list": card_list.reshape(NB, 52 * 51).to(ix),
            "cflo": cflo.to(ix),
            "cfhi": cfhi.to(ix),
        }

    def index_of(self, board: Sequence[int]) -> int:
        return self.boards.index(tuple(int(c) for c in board))

    def showdown(self, reach: torch.Tensor, board_ids: torch.Tensor) -> torch.Tensor:
        """``[M, 1326]`` opponent reach rows, ``[M]`` board ids -> ``[M, 1326]``
        showdown sign sums ``W - L`` (zero for combos blocked by the board)."""
        M = reach.shape[0]
        if M == 0:
            return reach.clone()
        valid = self.valid[board_ids]
        r = reach * valid
        z = r.new_zeros(M, 1)
        sp = torch.cat([z, r.gather(1, self.order[board_ids]).cumsum(1)], 1)
        w_all = sp.gather(1, self.lo[board_ids])
        l_all = sp[:, -1:] - sp.gather(1, self.hi[board_ids])
        rc = r.gather(1, self.card_list[board_ids]).view(M, 52, 51).cumsum(2)
        pc = torch.cat([r.new_zeros(M, 52, 1), rc], 2).view(M, 52 * 52)
        w_c = pc.gather(1, self.cflo[board_ids]).view(M, C, 2).sum(2)
        tot_c = pc.gather(1, self.ctot.expand(M, -1)).view(M, C, 2)
        l_c = (tot_c - pc.gather(1, self.cfhi[board_ids]).view(M, C, 2)).sum(2)
        return ((w_all - w_c) - (l_all - l_c)) * valid


def fold_values(reach: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Opponent reach mass disjoint from each combo, zero for invalid combos."""
    return blocked_sum(reach * valid) * valid


def naive_showdown(reach: torch.Tensor, board: Sequence[int]) -> torch.Tensor:
    """Dense ``1326 x 1326`` reference for :meth:`ShowdownTables.showdown`."""
    dev = reach.device
    s = combo_strengths(torch.tensor([list(board)], device=dev))[0]
    valid = s >= 0
    sign = torch.sign(s[:, None] - s[None, :]).to(reach.dtype)
    ok = (~conflict_matrix(dev)) & valid[:, None] & valid[None, :]
    M = sign * ok
    return (reach * valid) @ M.t() * valid
