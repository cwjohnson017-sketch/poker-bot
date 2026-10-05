"""Depth-limit leaf values from a river value network (DeepStack style).

With ``TreeConfig.leaf_mode = "value_net"`` a depth-limit leaf is a ``VALUE``
terminal at the end of turn betting, on a 4-card board ``b4``. Its values are
the exact chance average over the river card ``x`` of a **river-start** value
net ``N_R``, evaluated on both players' current reaches (the net is nonlinear
in both ranges, so it is re-evaluated as the solver's strategies change):

    v_i(leaf)(c) = sum_{x not in b4} (1 / 44) * [c avoids x] * m^x_-i(c) * pot * ev^x_i(c)

* ``r^x_p = pi_p(leaf) * [avoids x]`` are the reaches at the river root
  ``b4 + x``; the net gets them in ``(OOP, IP)`` order (OOP = the non-button
  seat, first to act on the river) together with ``c`` (chips each player has
  committed, equal at a river root) and the chips behind;
* ``ev^x_p(c) = v_p(c) / m_-p(c) / pot`` is the net's output: counterfactual
  value per unit of disjoint opponent mass, in pot units (``pot = 2c``);
* ``m^x_-i(c) = blocked_sum(r^x_-i)(c)`` is the opponent's mass at the river
  root that is disjoint from ``c``;
* ``44 = 52 - 4 - 4``: for any two disjoint hands exactly 44 river cards avoid
  both, so this is the solver's own chance-node identity and card removal
  stays exact (the net never has to learn the river-card average).

Every (leaf, river card) pair is one net row; rows are laid out leaf-major
(48 per leaf) and evaluated in chunks of whole leaves. The opponent masses of
the 48 masked copies come from one per-card sum per leaf:
``m^x(c) = m(c) - S[x] + r({x, c1}) + r({x, c2})`` for ``c = {c1, c2}``
avoiding ``x``, with ``S[x]`` the mass of combos holding ``x``. They are
never materialised: ``sum_x ev^x * m^x`` is one batched matmul per chunk (see
:meth:`ValueLeafEvaluator._combine`), so beyond the net the cost of a call is
about one write of the net input and one read of the net output.

Providers here (anything with ``values(player, reach[2, L, 1326], cached) ->
[L, 1326]`` works with :class:`~pokerbot.search.solver.TerminalEvaluator`;
``cached`` is true for the solver's regret updates, false for every other
evaluation):

* :class:`ValueLeafEvaluator` - the net, optionally re-run only every
  ``every`` regret updates per player (cached ``ev`` re-weighted by the
  current ``m``);
* :class:`FixedLeafValues` - precomputed values, ignoring the reaches;
* :class:`ShowdownOracle` - an exact ``predict`` for a checked-down river,
  for tests and as a sanity baseline.
"""

from __future__ import annotations

import warnings
from typing import Any, Protocol

import torch

from .combos import (
    NUM_CARDS,
    NUM_COMBOS,
    avoids_card,
    blocked_sum,
    combo_table,
    incidence,
    valid_masks,
)
from .showdown import ShowdownTables
from .tree import VALUE, SubgameTree

C = NUM_COMBOS
TURN_LEN = 4
RIVERS = NUM_CARDS - TURN_LEN  # river cards per turn board (48)
RIVERS_PER_PAIR = NUM_CARDS - TURN_LEN - 4  # river cards avoiding two disjoint hands (44)


class RiverPredictor(Protocol):
    def predict(
        self,
        boards: torch.Tensor,  # [n, 5] long
        ranges: torch.Tensor,  # [n, 2, 1326], (OOP, IP), any positive scale
        c: torch.Tensor,  # [n] chips each player committed
        stack: torch.Tensor,  # [n] chips behind
    ) -> torch.Tensor:  # [n, 2, 1326] ev in pot units, (OOP, IP), 0 on invalid combos
        ...


class LeafValueProvider(Protocol):
    def values(self, player: int, reach: torch.Tensor, cached: bool = False) -> torch.Tensor: ...


class ValueLeafEvaluator:
    """Values of every ``VALUE`` node of ``tree`` from a river value net.

    ``values(player, reach)`` takes both players' reaches at the ``VALUE``
    nodes, ``[2, L, 1326]`` in the order of ``ids`` (the ``VALUE`` node ids
    sorted by turn board, then node id; :class:`~pokerbot.search.solver.TerminalEvaluator`
    uses a provider's ``ids`` when it has them), and returns ``player``'s
    counterfactual values ``[L, 1326]`` in chips (solver convention).
    ``every > 1`` re-runs the net only on every ``every``-th regret update
    (``cached=True`` call) per player and otherwise reuses the cached ``ev``
    (stored as ``cache_dtype``) with the current opponent masses; other calls
    always run the net. ``every = 1`` is exact. ``chunk`` is the number of net
    rows per ``predict`` call (rounded down to whole leaves).

    The predictor must return finite values, also for rows where a player's
    range is empty (unreached leaves), and 0 on combos that conflict with the
    5-card board. The first net call checks the latter; if it fails, a warning
    is issued and those entries are zeroed in place from then on.
    """

    def __init__(
        self,
        tree: SubgameTree,
        predictor: RiverPredictor,
        every: int = 1,
        chunk: int = 16384,
        cache_dtype: torch.dtype = torch.float16,
    ) -> None:
        dev = tree.device
        self.device = dev
        self.predictor = predictor
        self.every = max(1, int(every))
        self.leaves_per_chunk = max(1, int(chunk) // RIVERS)
        self.cache_dtype = cache_dtype
        nodes = (tree.kind == VALUE).nonzero().flatten().tolist()
        L = len(nodes)
        self.num_leaves = L
        b4_index: dict[tuple[int, ...], int] = {}
        node_b4: dict[int, int] = {}
        for n in nodes:
            board = tuple(int(x) for x in tree.boards[int(tree.board_id[n])])
            if len(board) != TURN_LEN:
                raise ValueError(
                    f"value-net leaf {n} is on a {len(board)}-card board, but only a river "
                    "value net (turn-end leaves, 4-card boards) is supported; a flop solve "
                    "needs depth_streets >= 1 (depth_streets 0 would need a turn net)"
                )
            node_b4[n] = b4_index.setdefault(board, len(b4_index))
        # evaluation order: grouped by turn board, so a run of leaves shares its river cards
        nodes.sort(key=lambda n: (node_b4[n], n))
        self.ids = torch.tensor(nodes, dtype=torch.long, device=dev)
        leaf_b4 = [node_b4[n] for n in nodes]
        stacks = [min(int(s) for s in tree.states[n].stacks) for n in nodes]
        oop = [1 - int(tree.states[n].button) for n in nodes]
        contrib = tree.contrib[self.ids]
        if L and bool((contrib[:, 0] != contrib[:, 1]).any()):
            raise ValueError("value-net leaves need equal contributions (a river root)")
        if len(set(oop)) > 1:
            raise ValueError("value-net leaves of one tree must share the button")
        self.oop_seat = oop[0] if oop else 1
        boards4 = list(b4_index)
        rivers = [[x for x in range(NUM_CARDS) if x not in b] for b in boards4]
        self.boards4 = boards4
        # 5-card board id = (turn board id) * 48 + (river slot)
        self.boards5 = torch.tensor(
            [list(b) + [x] for b, xs in zip(boards4, rivers, strict=True) for x in xs],
            dtype=torch.long,
            device=dev,
        ).view(-1, 5)
        lb = torch.tensor(leaf_b4, dtype=torch.long, device=dev)
        slots = torch.arange(RIVERS, device=dev)
        rivers_t = torch.tensor(rivers, dtype=torch.long, device=dev).view(-1, RIVERS)
        self.cards = rivers_t[lb]
        self._river_avoid = avoids_card(dev)[rivers_t]  # [B4, 48, C] bool
        self._river_avoid_f: dict[torch.dtype, torch.Tensor] = {}
        # per chunk of whole leaves: runs (start, end, turn board) of leaves on one board
        self._runs: list[list[tuple[int, int, int]]] = []
        for l0 in range(0, L, self.leaves_per_chunk):
            l1 = min(L, l0 + self.leaves_per_chunk)
            runs: list[tuple[int, int, int]] = []
            for j in range(l0, l1):
                if runs and runs[-1][2] == leaf_b4[j]:
                    runs[-1] = (runs[-1][0], j + 1, leaf_b4[j])
                else:
                    runs.append((j, j + 1, leaf_b4[j]))
            self._runs.append(runs)
        self._zero_output: bool | None = None  # None until the first net call checks
        # one row per (leaf, river card), leaf-major: row = leaf * 48 + slot
        self.row_leaf = torch.arange(L, device=dev).repeat_interleave(RIVERS)
        self.row_card = self.cards.flatten()
        self.row_board = (lb[:, None] * RIVERS + slots[None, :]).flatten()
        self.c = contrib[:, 0].clone()  # [L] long
        self.stack = torch.tensor(stacks, dtype=torch.long, device=dev)
        self.oop = torch.tensor(oop, dtype=torch.long, device=dev)
        self.valid = (
            valid_masks(boards4, dev)[lb] if L else torch.zeros(0, C, dtype=torch.bool, device=dev)
        )
        self.row_c = self.c.float().repeat_interleave(RIVERS)
        self.row_stack = self.stack.float().repeat_interleave(RIVERS)
        self.avoid = avoids_card(dev)  # [52, C] bool
        # the 51 combos holding each card
        self.card_combos = incidence(dev).t().nonzero()[:, 1].view(NUM_CARDS, NUM_CARDS - 1)
        cards = combo_table(dev)
        self.c1, self.c2 = cards[:, 0].contiguous(), cards[:, 1].contiguous()
        self.c12 = cards.t().contiguous()  # [2, C]
        self.pair_ab = (self.c1 * NUM_CARDS + self.c2).contiguous()
        self.pair_ba = (self.c2 * NUM_CARDS + self.c1).contiguous()
        self._calls = [0, 0]
        self._cache: list[torch.Tensor | None] = [None, None]
        self._board_ids: torch.Tensor | None = None  # predictor board-cache ids
        self._board_gen: Any = None
        self.net_calls = 0  # predict() calls
        self.net_rows = 0  # rows sent to predict()

    @property
    def num_rows(self) -> int:
        return self.num_leaves * RIVERS

    def reset_cache(self) -> None:
        self._calls = [0, 0]
        self._cache = [None, None]

    # -- pieces ---------------------------------------------------------------

    def _pair_table(self, opp: torch.Tensor, S: torch.Tensor, cards: torch.Tensor) -> torch.Tensor:
        """``Q[l, j, y] = opp_l({x_j, y}) - S_l[x_j] / 2`` for the river cards
        ``x_j = cards[l, j]``, as ``[Lc, R, 52]``."""
        Lc = opp.shape[0]
        pair = opp.new_zeros(Lc, NUM_CARDS * NUM_CARDS)  # pair[x, y] = opp({x, y})
        pair[:, self.pair_ab] = opp
        pair[:, self.pair_ba] = opp
        Q = pair.view(Lc, NUM_CARDS, NUM_CARDS) - 0.5 * S[:, :, None]
        return Q.gather(1, cards[:, :, None].expand(-1, -1, NUM_CARDS))

    def opponent_mass(self, opp: torch.Tensor, cards: torch.Tensor) -> torch.Tensor:
        """``m^x(c) = blocked_sum(opp * [avoids x])(c)`` for ``opp [Lc, C]`` and
        ``cards [Lc, R]``, as ``[Lc, R, C]``: ``m(c) + Q[x, c1] + Q[x, c2]``.
        Only meaningful where ``c`` avoids ``x``. A reference:
        :meth:`values` never materialises it."""
        S = opp @ incidence(opp.device, opp.dtype)  # [Lc, 52]
        m = opp.sum(1, keepdim=True) - S[:, self.c1] - S[:, self.c2] + opp  # blocked_sum
        Qx = self._pair_table(opp, S, cards)
        return m[:, None, :] + Qx[:, :, self.c1] + Qx[:, :, self.c2]

    def net_ev(self, player: int, ordered: torch.Tensor, chunk: int) -> torch.Tensor:
        """``ev^x_player`` (pot units) of the leaves of chunk ``chunk`` from the
        net, as a ``[Lc, 48, C]`` view, zero on combos holding ``x``.
        ``ordered`` is ``[L, 2, C]``: both reaches in ``(OOP, IP)`` order,
        masked by the leaf boards."""
        runs = self._runs[chunk]
        l0, l1 = runs[0][0], runs[-1][1]
        Lc = l1 - l0
        n = Lc * RIVERS
        dt = ordered.dtype
        avoid = self._river_avoid_f.get(dt)
        if avoid is None:
            avoid = self._river_avoid_f[dt] = self._river_avoid.to(dt)
        ranges = ordered.new_empty(Lc, RIVERS, 2, C)
        for a, b, u in runs:  # leaves on one turn board share the 48 river masks
            torch.mul(ordered[a:b, None], avoid[u][None, :, None, :], out=ranges[a - l0 : b - l0])
        rows = slice(l0 * RIVERS, l1 * RIVERS)
        with torch.no_grad():
            ids = self._predictor_board_ids()
            if ids is None:
                ev = self.predictor.predict(
                    self.boards5[self.row_board[rows]],
                    ranges.view(n, 2, C),
                    self.row_c[rows],
                    self.row_stack[rows],
                )
            else:
                ev = self.predictor.predict_ids(
                    ids[self.row_board[rows]],
                    ranges.view(n, 2, C),
                    self.row_c[rows],
                    self.row_stack[rows],
                )
        self.net_calls += 1
        self.net_rows += n
        ev = ev.to(dt).reshape(Lc, RIVERS, 2, C)
        hit = None
        if self._zero_output is None:  # first call: does the predictor keep the contract?
            hit = self.card_combos[self.cards[l0:l1]]  # [Lc, 48, 51]
            idx = hit[:, :, None, :].expand(-1, -1, 2, -1)
            self._zero_output = bool((ev.gather(3, idx) != 0).any())
            if self._zero_output:
                warnings.warn(
                    "the value-net predictor returns nonzero values on combos that conflict "
                    "with the board; zeroing them",
                    stacklevel=2,
                )
        ev_p = ev[:, :, 0 if player == self.oop_seat else 1]
        if self._zero_output:
            if hit is None:
                hit = self.card_combos[self.cards[l0:l1]]
            ev_p.scatter_(2, hit, 0.0)
        return ev_p

    def _predictor_board_ids(self) -> torch.Tensor | None:
        """Ids of the 5-card boards in a predictor with a board cache
        (``board_ids`` / ``predict_ids``), looked up once and again only when the
        predictor's cache was reset; ``None`` for a plain ``predict``."""
        pred = self.predictor
        if not hasattr(pred, "predict_ids"):
            return None
        cache = getattr(pred, "cache", None)
        gen = getattr(cache, "generation", None)
        if self._board_ids is None or gen != self._board_gen:
            self._board_ids = pred.board_ids(self.boards5)
            self._board_gen = getattr(cache, "generation", None)
        return self._board_ids

    def _combine(
        self,
        ev: torch.Tensor,
        opp: torch.Tensor,
        S: torch.Tensor,
        m: torch.Tensor,
        cards: torch.Tensor,
    ) -> torch.Tensor:
        """``sum_x ev^x(c) * m^x(c)`` as ``[Lc, C]``, without materialising
        ``m^x``. With ``m^x(c) = m(c) + Q[x, c1] + Q[x, c2]`` it is
        ``m(c) * T[52, c] + T[c1, c] + T[c2, c]`` for ``T = [Q | 1]^T @ ev``:
        one batched matmul, ``[Lc, 53, C]``. ``ev`` must be zero on the combos
        holding ``x``."""
        Lc = ev.shape[0]
        Qx = self._pair_table(opp, S, cards)  # [Lc, R, 52]
        Q1 = torch.cat([Qx, Qx.new_ones(Lc, RIVERS, 1)], 2)  # [Lc, R, 53]
        T = torch.matmul(Q1.transpose(1, 2), ev)  # [Lc, 53, C]
        w = m * T[:, NUM_CARDS]
        w += T.gather(1, self.c12[None].expand(Lc, -1, -1)).sum(1)
        return w

    # -- provider -------------------------------------------------------------

    @torch.no_grad()
    def values(self, player: int, reach: torch.Tensor, cached: bool = False) -> torch.Tensor:
        """``player``'s values ``[L, C]`` (chips) at the ``VALUE`` nodes, from both
        players' reaches ``[2, L, C]``. ``cached=True`` (the solver's regret
        updates) lets ``every > 1`` reuse the ``ev`` of an earlier call. Other
        calls (values, best responses, exploitability) run the net and leave the
        cache alone."""
        L = self.num_leaves
        out = reach.new_zeros(L, C)
        if L == 0:
            return out
        if reach.shape != (2, L, C):
            raise ValueError(f"reach must be [2, {L}, {C}], got {tuple(reach.shape)}")
        dt = reach.dtype
        rv = reach * self.valid
        refresh, cache = True, None
        if cached and self.every > 1:
            k = self._calls[player]
            self._calls[player] += 1
            cache = self._cache[player]
            if cache is None:
                cache = torch.empty(L, RIVERS, C, device=self.device, dtype=self.cache_dtype)
                self._cache[player] = cache
            else:
                refresh = k % self.every == 0
        opp = rv[1 - player]
        S = opp @ incidence(opp.device, dt)  # [L, 52]: mass holding each card
        m = opp.sum(1, keepdim=True) - S[:, self.c1] - S[:, self.c2] + opp  # blocked_sum
        ordered = None
        if refresh:
            o = self.oop_seat
            ordered = torch.stack([rv[o], rv[1 - o]], 1)  # [L, 2, C], (OOP, IP)
        scale = self.c.to(dt) * (2.0 / RIVERS_PER_PAIR)  # pot / 44
        for k, runs in enumerate(self._runs):
            l0, l1 = runs[0][0], runs[-1][1]
            if refresh:
                ev = self.net_ev(player, ordered, k)
                if cache is not None:
                    cache[l0:l1] = ev
            else:
                ev = cache[l0:l1].to(dt)
            w = self._combine(ev, opp[l0:l1], S[l0:l1], m[l0:l1], self.cards[l0:l1])
            w = torch.nan_to_num_(w, nan=0.0, posinf=0.0, neginf=0.0)
            out[l0:l1] = w * scale[l0:l1, None] * self.valid[l0:l1]
        return out


class FixedLeafValues:
    """A provider returning precomputed values ``[2, L, 1326]`` (chips, solver
    convention) for the ``VALUE`` nodes, whatever the reaches."""

    def __init__(self, values: torch.Tensor) -> None:
        self.v = values

    def values(self, player: int, reach: torch.Tensor, cached: bool = False) -> torch.Tensor:
        return self.v[player].to(reach.device, reach.dtype)


class ShowdownOracle:
    """Exact ``predict`` for a river that is checked down (both players check,
    showdown): ``ev_p = 0.5 * showdown(r_-p) / m_-p`` in pot units (0 where
    ``m_-p = 0`` and on combos that conflict with the board). Works in the
    dtype of ``ranges``; showdown tables are cached per board."""

    _FIELDS = ("valid", "order", "lo", "hi", "card_list", "cflo", "cfhi")

    def __init__(self) -> None:
        self._index: dict[tuple[int, ...], int] = {}
        self._tables: ShowdownTables | None = None
        self._device: torch.device | None = None

    def _table_ids(self, boards: torch.Tensor) -> torch.Tensor:
        dev = boards.device
        if self._device != dev:
            self._index, self._tables, self._device = {}, None, dev
        uniq, inv = torch.unique(boards.long().sort(1).values, dim=0, return_inverse=True)
        keys = [tuple(b) for b in uniq.tolist()]
        new = [k for k in keys if k not in self._index]
        if new:
            t = ShowdownTables(new, dev)
            if self._tables is None:
                self._tables = t
            else:
                old = self._tables
                for name in self._FIELDS:
                    setattr(old, name, torch.cat([getattr(old, name), getattr(t, name)]))
                old.boards = old.boards + t.boards
                old.num_boards = len(old.boards)
            for k in new:
                self._index[k] = len(self._index)
        pos = torch.tensor([self._index[k] for k in keys], dtype=torch.long, device=dev)
        return pos[inv]

    @torch.no_grad()
    def predict(
        self, boards: torch.Tensor, ranges: torch.Tensor, c: Any = None, stack: Any = None
    ) -> torch.Tensor:
        out = torch.zeros_like(ranges)
        if ranges.shape[0] == 0:
            return out
        ids = self._table_ids(boards.to(ranges.device))
        t = self._tables
        valid = t.valid[ids]
        for p in (0, 1):
            r = ranges[:, 1 - p] * valid
            sd = t.showdown(r, ids)
            m = blocked_sum(r) * valid
            ev = (0.5 * sd / m.clamp(min=torch.finfo(m.dtype).tiny)).clamp(-0.5, 0.5)
            out[:, p] = torch.where(m > 0, ev, 0.0)
        return out


__all__ = [
    "FixedLeafValues",
    "LeafValueProvider",
    "RiverPredictor",
    "ShowdownOracle",
    "ValueLeafEvaluator",
]
