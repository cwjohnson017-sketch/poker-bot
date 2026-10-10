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
:meth:`RiverAverage._combine`), so beyond the net the cost of a call is
about one write of the net input and one read of the net output.

The terms of the average, ``v^x_i(c) = [c avoids x] * m^x_-i(c) * pot *
ev^x_i(c)``, are the values at the river roots below a leaf: the solver's
value of a chance child dealing ``x`` (``sum_x v^x / 44`` is the leaf value).
:meth:`RiverAverage.card_values` returns them for some leaves; the
continual-resolving cache stores them for the river search
(:meth:`~pokerbot.search.gadget.ContinualCache.store`).

The same chance average without a tree is :func:`river_average` (states given
as tensors, both players at once); it is the target of the **turn-end net**
(:mod:`.turn_net`, DeepStack's auxiliary net), which predicts it directly:

    ev_TE_p(c) = v_p(c) / (m_-p(c) * pot),   m_-p = blocked_sum(pi_-p) on b4

(``sum_x [c avoids x] m^x_-p(c) = 44 m_-p(c)``, so ``ev_TE`` is a convex
combination of the ``ev^x``). :class:`TurnEndLeafEvaluator` uses such a net,
one row per leaf instead of 48; it has no per-card values, so nothing is
cached for the river below its leaves.

Providers here (anything with ``values(player, reach[2, L, 1326], cached) ->
[L, 1326]`` works with :class:`~pokerbot.search.solver.TerminalEvaluator`;
``cached`` is true for the solver's regret updates, false for every other
evaluation):

* :class:`ValueLeafEvaluator` - the river net averaged over the river cards,
  optionally re-run only every ``every`` regret updates per player (cached
  ``ev`` re-weighted by the current ``m``);
* :class:`TurnEndLeafEvaluator` - a turn-end net, one row per leaf;
* :class:`FlopEndLeafEvaluator` - flop-end leaves (a flop solve with
  ``depth_streets: 0``, 3-card boards): a **turn-start** net (``kind ==
  "turn_start"``, :class:`~.turn_net.TurnStartPredictor`) averaged over the 49
  turn cards by the same identity (:class:`RiverAverage` with
  ``board_len = 3``: ``1 / 45`` per turn card avoiding both hands);
* :class:`FixedLeafValues` - precomputed values, ignoring the reaches;
* :class:`ShowdownOracle` - an exact river ``predict`` for a checked-down
  river, for tests and as a sanity baseline.

:func:`make_leaf_evaluator` picks the provider from the predictor's ``kind``.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
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
FLOP_LEN = 3
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


def predictor_kind(predictor: Any) -> str:
    """``"river"`` (river-start net: boards ``[n, 5]``), ``"turn_end"``
    (turn-end net: boards ``[n, 4]``) or ``"turn_start"`` (turn-start net:
    boards ``[n, 4]``, values at the turn root), from the predictor's ``kind``
    attribute (default river)."""
    return str(getattr(predictor, "kind", "river"))


# -- turn-end leaves of a tree ----------------------------------------------------


def turn_end_leaves(tree: SubgameTree) -> dict[str, Any]:
    """The ``VALUE`` nodes of ``tree`` (turn-end leaves) in evaluation order
    (grouped by turn board, then node id) and their layout: ``nodes``, the
    distinct ``boards4``, ``leaf_b4`` (board index per leaf), ``c`` and
    ``stack`` (long ``[L]``), ``oop`` (the OOP seat per leaf) and ``oop_seat``."""
    return _value_leaf_layout(
        tree,
        TURN_LEN,
        "only turn-end leaves (4-card boards) are supported (a river net or a turn-end "
        "net); a flop solve with depth_streets 0 needs a turn-start net (FlopEndLeafEvaluator)",
    )


def flop_end_leaves(tree: SubgameTree) -> dict[str, Any]:
    """:func:`turn_end_leaves` for the ``VALUE`` nodes at the end of flop
    betting (3-card boards; a flop solve with ``depth_streets: 0``): the same
    keys, ``boards4`` / ``leaf_b4`` holding the distinct flop boards and the
    board index per leaf."""
    return _value_leaf_layout(
        tree,
        FLOP_LEN,
        "a turn-start net values only flop-end leaves (3-card boards, depth_streets 0 on "
        "the flop); turn-end leaves need a turn-end or river net (leaf.turn_net for turn "
        "decisions)",
    )


def _value_leaf_layout(tree: SubgameTree, board_len: int, why: str) -> dict[str, Any]:
    dev = tree.device
    nodes = (tree.kind == VALUE).nonzero().flatten().tolist()
    b4_index: dict[tuple[int, ...], int] = {}
    node_b4: dict[int, int] = {}
    for n in nodes:
        board = tuple(int(x) for x in tree.boards[int(tree.board_id[n])])
        if len(board) != board_len:
            raise ValueError(f"value-net leaf {n} is on a {len(board)}-card board, but {why}")
        node_b4[n] = b4_index.setdefault(board, len(b4_index))
    # evaluation order: grouped by turn board, so a run of leaves shares its river cards
    nodes.sort(key=lambda n: (node_b4[n], n))
    ids = torch.tensor(nodes, dtype=torch.long, device=dev)
    oop = [1 - int(tree.states[n].button) for n in nodes]
    contrib = tree.contrib[ids]
    if nodes and bool((contrib[:, 0] != contrib[:, 1]).any()):
        raise ValueError("value-net leaves need equal contributions (a river root)")
    if len(set(oop)) > 1:
        raise ValueError("value-net leaves of one tree must share the button")
    return {
        "nodes": nodes,
        "ids": ids,
        "boards4": list(b4_index),
        "leaf_b4": [node_b4[n] for n in nodes],
        "c": contrib[:, 0].clone(),
        "stack": torch.tensor(
            [min(int(s) for s in tree.states[n].stacks) for n in nodes],
            dtype=torch.long,
            device=dev,
        ),
        "oop": oop,
        "oop_seat": oop[0] if oop else 1,
    }


# -- the chance average over river cards --------------------------------------------


class RiverAverage:
    """Turn-end values as the exact chance average of a river predictor.

    The class is generic in the street: ``board_len`` (class attribute, 4
    here) is the leaves' board length, the predictor gets boards of
    ``board_len + 1`` cards, and the chance weight is ``1 / (52 - board_len -
    4)`` per dealt card avoiding both hands. :class:`FlopEndLeafEvaluator`
    (``board_len = 3``) averages a turn-start net over the turn cards the same
    way; names below say "river" for the dealt card.

    Leaves ``l`` are given by ``leaf_b4[l]`` (index into the distinct turn
    boards ``boards4``; leaves on one board should be consecutive, so a run of
    them shares its river masks), the committed chips ``c [L]`` and the chips
    behind ``stack [L]``. ``oop_seat`` says which row of the reaches is the OOP
    player (0 or 1). :meth:`values` gives one player's values (solver
    convention, chips), :meth:`values_both` both from one net pass.

    ``every > 1`` re-runs the net only on every ``every``-th ``cached=True``
    call of :meth:`values` per player and otherwise reuses the cached ``ev``
    (stored as ``cache_dtype``) with the current opponent masses; other calls
    always run the net. ``chunk`` is the number of net rows per ``predict``
    call (rounded down to whole leaves).

    The predictor must return finite values, also for rows where a player's
    range is empty (unreached leaves), and 0 on combos that conflict with the
    5-card board. The first net call checks the latter; if it fails, a warning
    is issued and those entries are zeroed in place from then on.
    """

    board_len = TURN_LEN

    def __init__(
        self,
        predictor: RiverPredictor,
        boards4: Sequence[Sequence[int]],
        leaf_b4: Sequence[int],
        c: torch.Tensor,
        stack: torch.Tensor,
        oop_seat: int,
        device: torch.device | str,
        every: int = 1,
        chunk: int = 16384,
        cache_dtype: torch.dtype = torch.float16,
    ) -> None:
        dev = self.device = torch.device(device)
        self.predictor = predictor
        self.every = max(1, int(every))
        R = self.cards_per_leaf = NUM_CARDS - self.board_len  # dealt cards per board (48)
        self.cards_per_pair = R - 4  # dealt cards avoiding two disjoint hands (44)
        self.leaves_per_chunk = max(1, int(chunk) // R)
        self.cache_dtype = cache_dtype
        leaf_b4 = [int(u) for u in leaf_b4]
        L = len(leaf_b4)
        self.num_leaves = L
        self.oop_seat = int(oop_seat)
        boards4 = [tuple(int(x) for x in b) for b in boards4]
        rivers = [[x for x in range(NUM_CARDS) if x not in b] for b in boards4]
        self.boards4 = boards4
        # 5-card board id = (turn board id) * 48 + (river slot)
        self.boards5 = torch.tensor(
            [list(b) + [x] for b, xs in zip(boards4, rivers, strict=True) for x in xs],
            dtype=torch.long,
            device=dev,
        ).view(-1, self.board_len + 1)
        lb = self.leaf_board = torch.tensor(leaf_b4, dtype=torch.long, device=dev)
        slots = torch.arange(R, device=dev)
        rivers_t = torch.tensor(rivers, dtype=torch.long, device=dev).view(-1, R)
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
        self.row_leaf = torch.arange(L, device=dev).repeat_interleave(R)
        self.row_card = self.cards.flatten()
        self.row_board = (lb[:, None] * R + slots[None, :]).flatten()
        self.c = torch.as_tensor(c, device=dev).long().reshape(L)
        self.stack = torch.as_tensor(stack, device=dev).long().reshape(L)
        self.valid = (
            valid_masks(boards4, dev)[lb] if L else torch.zeros(0, C, dtype=torch.bool, device=dev)
        )
        self.row_c = self.c.float().repeat_interleave(R)
        self.row_stack = self.stack.float().repeat_interleave(R)
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
        return self.num_leaves * self.cards_per_leaf

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

    def _net_ev_both(self, ordered: torch.Tensor, chunk: int) -> torch.Tensor:
        """``ev^x`` (pot units) of both players for the leaves of chunk
        ``chunk`` from the net, as ``[Lc, 48, 2, C]`` in ``(OOP, IP)`` order,
        zero on combos holding ``x``. ``ordered`` is ``[L, 2, C]``: both reaches
        in ``(OOP, IP)`` order, masked by the leaf boards."""
        runs = self._runs[chunk]
        l0, l1 = runs[0][0], runs[-1][1]
        Lc = l1 - l0
        R = self.cards_per_leaf
        n = Lc * R
        dt = ordered.dtype
        avoid = self._river_avoid_f.get(dt)
        if avoid is None:
            avoid = self._river_avoid_f[dt] = self._river_avoid.to(dt)
        ranges = ordered.new_empty(Lc, R, 2, C)
        for a, b, u in runs:  # leaves on one turn board share the 48 river masks
            torch.mul(ordered[a:b, None], avoid[u][None, :, None, :], out=ranges[a - l0 : b - l0])
        ev = self._predict(slice(l0 * R, l1 * R), ranges.view(n, 2, C))
        ev = ev.to(dt).reshape(Lc, R, 2, C)
        hit = None
        if self._zero_output is None:  # first call: does the predictor keep the contract?
            hit = self.card_combos[self.cards[l0:l1]]  # [Lc, 48, 51]
            idx = hit[:, :, None, :].expand(-1, -1, 2, -1)
            self._zero_output = bool((ev.gather(3, idx) != 0).any())
            if self._zero_output:
                warnings.warn(
                    "the value-net predictor returns nonzero values on combos that conflict "
                    "with the board; zeroing them",
                    stacklevel=3,
                )
        if self._zero_output:
            if hit is None:
                hit = self.card_combos[self.cards[l0:l1]]
            ev.scatter_(3, hit[:, :, None, :].expand(-1, -1, 2, -1), 0.0)
        return ev

    def _predict(self, rows: slice | torch.Tensor, ranges: torch.Tensor) -> torch.Tensor:
        """The predictor's output ``[n, 2, C]`` for the net rows ``rows`` (a slice
        or an index tensor into the leaf-major rows) given their ranges ``[n, 2, C]``."""
        with torch.no_grad():
            ids = self._predictor_board_ids()
            if ids is None:
                ev = self.predictor.predict(
                    self.boards5[self.row_board[rows]],
                    ranges,
                    self.row_c[rows],
                    self.row_stack[rows],
                )
            else:
                ev = self.predictor.predict_ids(
                    ids[self.row_board[rows]], ranges, self.row_c[rows], self.row_stack[rows]
                )
        self.net_calls += 1
        self.net_rows += int(ranges.shape[0])
        return ev

    def net_ev(self, player: int, ordered: torch.Tensor, chunk: int) -> torch.Tensor:
        """``ev^x_player`` of the leaves of chunk ``chunk``, a ``[Lc, 48, C]`` view
        of :meth:`_net_ev_both`."""
        return self._net_ev_both(ordered, chunk)[:, :, 0 if player == self.oop_seat else 1]

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
        Q1 = torch.cat([Qx, Qx.new_ones(Lc, self.cards_per_leaf, 1)], 2)  # [Lc, R, 53]
        T = torch.matmul(Q1.transpose(1, 2), ev)  # [Lc, 53, C]
        w = m * T[:, NUM_CARDS]
        w += T.gather(1, self.c12[None].expand(Lc, -1, -1)).sum(1)
        return w

    def _prepare(self, reach: torch.Tensor) -> torch.Tensor:
        L = self.num_leaves
        if reach.shape != (2, L, C):
            raise ValueError(f"reach must be [2, {L}, {C}], got {tuple(reach.shape)}")
        return reach * self.valid

    def _opp_sums(self, rv: torch.Tensor, player: int) -> tuple[torch.Tensor, ...]:
        """Opponent reach, its per-card sums ``S [L, 52]`` and ``blocked_sum``."""
        opp = rv[1 - player]
        S = opp @ incidence(opp.device, rv.dtype)  # [L, 52]: mass holding each card
        m = opp.sum(1, keepdim=True) - S[:, self.c1] - S[:, self.c2] + opp  # blocked_sum
        return opp, S, m

    def _ordered(self, rv: torch.Tensor) -> torch.Tensor:
        o = self.oop_seat
        return torch.stack([rv[o], rv[1 - o]], 1)  # [L, 2, C], (OOP, IP)

    # -- provider -------------------------------------------------------------

    @torch.no_grad()
    def values(self, player: int, reach: torch.Tensor, cached: bool = False) -> torch.Tensor:
        """``player``'s values ``[L, C]`` (chips) at the leaves, from both
        players' reaches ``[2, L, C]``. ``cached=True`` (the solver's regret
        updates) lets ``every > 1`` reuse the ``ev`` of an earlier call. Other
        calls (values, best responses, exploitability) run the net and leave the
        cache alone."""
        L = self.num_leaves
        out = reach.new_zeros(L, C)
        if L == 0:
            return out
        rv = self._prepare(reach)
        dt = reach.dtype
        refresh, cache = True, None
        if cached and self.every > 1:
            k = self._calls[player]
            self._calls[player] += 1
            cache = self._cache[player]
            if cache is None:
                R = self.cards_per_leaf
                cache = torch.empty(L, R, C, device=self.device, dtype=self.cache_dtype)
                self._cache[player] = cache
            else:
                refresh = k % self.every == 0
        opp, S, m = self._opp_sums(rv, player)
        ordered = self._ordered(rv) if refresh else None
        scale = self.c.to(dt) * (2.0 / self.cards_per_pair)  # pot / 44
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

    @torch.no_grad()
    def values_both(self, reach: torch.Tensor) -> torch.Tensor:
        """Both players' values ``[2, L, C]`` (chips, rows in the order of
        ``reach``) from one net pass per chunk (no caching)."""
        L = self.num_leaves
        out = reach.new_zeros(2, L, C)
        if L == 0:
            return out
        rv = self._prepare(reach)
        sums = [self._opp_sums(rv, p) for p in (0, 1)]
        ordered = self._ordered(rv)
        scale = self.c.to(reach.dtype) * (2.0 / self.cards_per_pair)
        for k, runs in enumerate(self._runs):
            l0, l1 = runs[0][0], runs[-1][1]
            ev = self._net_ev_both(ordered, k)
            for p in (0, 1):
                opp, S, m = sums[p]
                ev_p = ev[:, :, 0 if p == self.oop_seat else 1]
                w = self._combine(ev_p, opp[l0:l1], S[l0:l1], m[l0:l1], self.cards[l0:l1])
                w = torch.nan_to_num_(w, nan=0.0, posinf=0.0, neginf=0.0)
                out[p, l0:l1] = w * scale[l0:l1, None] * self.valid[l0:l1]
        return out

    @torch.no_grad()
    def card_values(
        self,
        player: int,
        reach: torch.Tensor,
        leaves: Sequence[int] | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``player``'s values at the roots of the next street below some leaves,
        one per dealt card: the terms of the chance average in :meth:`values`.

        ``reach`` is ``[2, L, C]`` as for :meth:`values` (every leaf, in this
        provider's leaf order), ``leaves`` the positions of the leaves wanted (all
        by default). Returns ``cards [n, R]`` (the dealt cards ``x`` per leaf, as
        :attr:`cards`) and ``v [n, R, C]`` in chips:

            v[l, j](c) = [c avoids x] * m^x_-p(c) * pot * ev^x_p(c),   x = cards[l, j]

        with ``m^x_-p = blocked_sum(pi_-p * [avoids x])`` and ``ev^x`` the
        predictor's output on board ``b + x`` for both reaches masked by ``x``
        (``(OOP, IP)`` order), ``pot = 2 c``. This is the solver's value of a
        chance child dealing ``x`` (not multiplied by the chance weight), so
        ``v.sum(1) / cards_per_pair == values(player, reach)[leaves]``. Runs the
        net on the ``n * R`` rows of these leaves only (no caching)."""
        dev = self.device
        L, R = self.num_leaves, self.cards_per_leaf
        idx = (
            torch.arange(L, device=dev)
            if leaves is None
            else torch.as_tensor(leaves, dtype=torch.long, device=dev).reshape(-1)
        )
        n = int(idx.numel())
        out = reach.new_zeros(n, R, C)
        if n == 0:
            return self.cards[idx], out
        rv = self._prepare(reach)
        dt = reach.dtype
        ordered = self._ordered(rv)  # [L, 2, C], (OOP, IP)
        opp = rv[1 - player]
        col = 0 if player == self.oop_seat else 1
        pot = 2.0 * self.c.to(dt)
        slots = torch.arange(R, device=dev)
        for s in range(0, n, self.leaves_per_chunk):
            li = idx[s : s + self.leaves_per_chunk]
            k = int(li.numel())
            avoid = self._river_avoid[self.leaf_board[li]]  # [k, R, C] bool
            ranges = ordered[li][:, None] * avoid[:, :, None, :].to(dt)  # [k, R, 2, C]
            rows = (li[:, None] * R + slots[None, :]).flatten()
            ev = self._predict(rows, ranges.view(k * R, 2, C)).to(dt).view(k, R, 2, C)
            m = self.opponent_mass(opp[li], self.cards[li])  # [k, R, C]
            v = torch.where(avoid, ev[:, :, col] * m, 0.0)
            v = torch.nan_to_num_(v, nan=0.0, posinf=0.0, neginf=0.0)
            out[s : s + k] = v * pot[li, None, None] * self.valid[li][:, None, :]
        return self.cards[idx], out


class ValueLeafEvaluator(RiverAverage):
    """Values of every ``VALUE`` node of ``tree`` from a river value net.

    ``values(player, reach)`` takes both players' reaches at the ``VALUE``
    nodes, ``[2, L, 1326]`` in the order of ``ids`` (the ``VALUE`` node ids
    sorted by turn board, then node id; :class:`~pokerbot.search.solver.TerminalEvaluator`
    uses a provider's ``ids`` when it has them), and returns ``player``'s
    counterfactual values ``[L, 1326]`` in chips (solver convention). See
    :class:`RiverAverage` for ``every``, ``chunk`` and the predictor contract.
    """

    def __init__(
        self,
        tree: SubgameTree,
        predictor: RiverPredictor,
        every: int = 1,
        chunk: int = 16384,
        cache_dtype: torch.dtype = torch.float16,
    ) -> None:
        lay = turn_end_leaves(tree)
        self.ids = lay["ids"]
        super().__init__(
            predictor,
            lay["boards4"],
            lay["leaf_b4"],
            lay["c"],
            lay["stack"],
            lay["oop_seat"],
            tree.device,
            every,
            chunk,
            cache_dtype,
        )
        self.oop = torch.tensor(lay["oop"], dtype=torch.long, device=self.device)


class FlopEndLeafEvaluator(RiverAverage):
    """Values of every ``VALUE`` node at the end of flop betting (3-card
    boards ``b3``: a flop solve with ``depth_streets: 0``) from a **turn-start**
    value net ``N_TS`` (``kind == "turn_start"``), by the exact chance average
    over the 49 turn cards ``t``:

        v_i(c) = sum_{t not in b3} (1 / 45) * [c avoids t] * m^t_-i(c) * pot * ev^t_i(c)

    with ``ev^t = N_TS(b3 + t, reaches masked by t, c, stack)`` in pot units per
    unit of disjoint opponent mass on the 4-card board and ``45 = 52 - 3 - 4``.
    This is :class:`ValueLeafEvaluator` one street earlier
    (:class:`RiverAverage` with ``board_len = 3``): 49 net rows per leaf, laid
    out leaf-major, the same ``ids``, ``values(player, reach[2, L, 1326],
    cached)``, ``every`` caching and predictor contract (0 on combos holding
    the turn card).
    """

    board_len = FLOP_LEN

    def __init__(
        self,
        tree: SubgameTree,
        predictor: Any,
        every: int = 1,
        chunk: int = 16384,
        cache_dtype: torch.dtype = torch.float16,
    ) -> None:
        lay = flop_end_leaves(tree)
        self.ids = lay["ids"]
        super().__init__(
            predictor,
            lay["boards4"],
            lay["leaf_b4"],
            lay["c"],
            lay["stack"],
            lay["oop_seat"],
            tree.device,
            every,
            chunk,
            cache_dtype,
        )
        self.oop = torch.tensor(lay["oop"], dtype=torch.long, device=self.device)


@torch.no_grad()
def river_average(
    predictor: RiverPredictor,
    boards4: torch.Tensor,
    c: torch.Tensor,
    stack: torch.Tensor,
    reach: torch.Tensor,
    oop_first: bool = True,
    chunk: int = 16384,
) -> torch.Tensor:
    """Turn-end values of both players by the exact chance average of a river
    predictor over the 48 river cards: the same math as
    :meth:`ValueLeafEvaluator.values`, without a tree.

    ``boards4 [L, 4]``, ``c [L]`` (chips each player committed), ``stack [L]``
    (chips behind), ``reach [2, L, 1326]`` (any non-negative scale; zeroed on
    the board's combos) in ``(OOP, IP)`` order when ``oop_first``, else
    ``(IP, OOP)``. Returns ``[2, L, 1326]`` counterfactual values in chips, rows
    in the order of ``reach``. ``chunk`` net rows per ``predict`` call (whole
    leaves; leaves are grouped by board first)."""
    dev = reach.device
    b = torch.as_tensor(boards4, device=dev).long().reshape(-1, TURN_LEN)
    L = b.shape[0]
    out = reach.new_zeros(2, L, C)
    if L == 0:
        return out
    c = torch.as_tensor(c, device=dev).long().reshape(L)
    stack = torch.as_tensor(stack, device=dev).long().reshape(L)
    _, inv = torch.unique(b.sort(1).values, dim=0, return_inverse=True)
    order = torch.argsort(inv, stable=True)  # leaves grouped by board
    per = max(1, int(chunk) // RIVERS)
    for s in range(0, L, per):
        idx = order[s : s + per]
        u, local = torch.unique(inv[idx], return_inverse=True)  # sorted: runs stay together
        first = torch.full((u.numel(),), idx.numel(), dtype=torch.long, device=dev)
        first.scatter_reduce_(0, local, torch.arange(idx.numel(), device=dev), "amin")
        core = RiverAverage(
            predictor,
            b[idx[first]].tolist(),
            local.tolist(),
            c[idx],
            stack[idx],
            0 if oop_first else 1,
            dev,
            chunk=per * RIVERS,
        )
        out[:, idx] = core.values_both(reach[:, idx])
    return out


class TurnEndLeafEvaluator:
    """Values of every ``VALUE`` node of ``tree`` from a **turn-end** value net:
    one ``predict`` row per leaf (boards ``[L, 4]``), instead of 48.

    The net predicts ``ev_TE_p(c) = v_p(c) / (m_-p(c) * pot)`` (pot units per
    unit of disjoint opponent mass on the 4-card board), so
    ``v_p = ev_TE_p * blocked_sum(pi_-p) * pot``. Same interface, ``ids`` and
    ``every`` caching (``ev`` cached per player, re-weighted by the current
    opponent mass) as :class:`ValueLeafEvaluator`.
    """

    def __init__(
        self,
        tree: SubgameTree,
        predictor: Any,
        every: int = 1,
        cache_dtype: torch.dtype = torch.float16,
    ) -> None:
        lay = turn_end_leaves(tree)
        dev = self.device = tree.device
        self.predictor = predictor
        self.every = max(1, int(every))
        self.cache_dtype = cache_dtype
        self.ids = lay["ids"]
        L = self.num_leaves = len(lay["nodes"])
        self.oop_seat = lay["oop_seat"]
        lb = torch.tensor(lay["leaf_b4"], dtype=torch.long, device=dev)
        boards4 = torch.tensor(lay["boards4"], dtype=torch.long, device=dev).view(-1, TURN_LEN)
        self.boards = boards4[lb]  # [L, 4]
        self.c = lay["c"]
        self.stack = lay["stack"]
        self.pot = 2.0 * self.c.double()
        self.valid = (
            valid_masks(lay["boards4"], dev)[lb]
            if L
            else torch.zeros(0, C, dtype=torch.bool, device=dev)
        )
        self._calls = [0, 0]
        self._cache: list[torch.Tensor | None] = [None, None]
        self._board_ids: torch.Tensor | None = None
        self._board_gen: Any = None
        self.net_calls = 0
        self.net_rows = 0

    @property
    def num_rows(self) -> int:
        return self.num_leaves

    def reset_cache(self) -> None:
        self._calls = [0, 0]
        self._cache = [None, None]

    def _predictor_board_ids(self) -> torch.Tensor | None:
        pred = self.predictor
        if not hasattr(pred, "predict_ids"):
            return None
        cache = getattr(pred, "cache", None)
        gen = getattr(cache, "generation", None)
        if self._board_ids is None or gen != self._board_gen:
            self._board_ids = pred.board_ids(self.boards)
            self._board_gen = getattr(cache, "generation", None)
        return self._board_ids

    def net_ev(self, rv: torch.Tensor) -> torch.Tensor:
        """``ev_TE [L, 2, C]`` (pot units, ``(OOP, IP)``) for the masked reaches ``rv``."""
        o = self.oop_seat
        ordered = torch.stack([rv[o], rv[1 - o]], 1)  # [L, 2, C]
        with torch.no_grad():
            ids = self._predictor_board_ids()
            if ids is None:
                ev = self.predictor.predict(self.boards, ordered, self.c, self.stack)
            else:
                ev = self.predictor.predict_ids(ids, ordered, self.c, self.stack)
        self.net_calls += 1
        self.net_rows += self.num_leaves
        return ev.to(rv.dtype)

    @torch.no_grad()
    def values(self, player: int, reach: torch.Tensor, cached: bool = False) -> torch.Tensor:
        L = self.num_leaves
        if L == 0:
            return reach.new_zeros(0, C)
        if reach.shape != (2, L, C):
            raise ValueError(f"reach must be [2, {L}, {C}], got {tuple(reach.shape)}")
        dt = reach.dtype
        rv = reach * self.valid
        refresh, cache = True, None
        if cached and self.every > 1:
            k = self._calls[player]
            self._calls[player] += 1
            cache = self._cache[player]
            refresh = cache is None or k % self.every == 0
        if refresh:
            ev = self.net_ev(rv)[:, 0 if player == self.oop_seat else 1]
            if cached and self.every > 1:
                self._cache[player] = ev.to(self.cache_dtype)
        else:
            ev = cache.to(dt)
        m = blocked_sum(rv[1 - player])
        w = ev * m * self.pot.to(dt)[:, None]
        w = torch.nan_to_num_(w, nan=0.0, posinf=0.0, neginf=0.0)
        return w * self.valid


def make_leaf_evaluator(
    tree: SubgameTree, predictor: Any, every: int = 1, **kwargs: Any
) -> RiverAverage | TurnEndLeafEvaluator:
    """The leaf-value provider for ``predictor``: :class:`TurnEndLeafEvaluator`
    for a turn-end net (``kind == "turn_end"``), :class:`FlopEndLeafEvaluator`
    for a turn-start net (``kind == "turn_start"``, flop-end leaves only), else
    :class:`ValueLeafEvaluator`."""
    kind = predictor_kind(predictor)
    if kind == "turn_end":
        return TurnEndLeafEvaluator(tree, predictor, every=every, **kwargs)
    if kind == "turn_start":
        return FlopEndLeafEvaluator(tree, predictor, every=every, **kwargs)
    if kind == "river":
        return ValueLeafEvaluator(tree, predictor, every=every, **kwargs)
    raise ValueError(f"unknown value-net kind {kind!r} (expected river, turn_end or turn_start)")


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
    dtype of ``ranges``; showdown tables are cached per board (cleared first
    when adding boards would exceed ``max_boards``, if set)."""

    kind = "river"
    _FIELDS = ("valid", "order", "lo", "hi", "card_list", "cflo", "cfhi")

    def __init__(self, max_boards: int | None = None) -> None:
        self.max_boards = max_boards
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
        if new and self.max_boards is not None and len(self._index) + len(new) > self.max_boards:
            self._index, self._tables = {}, None
            new = keys
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
    "FlopEndLeafEvaluator",
    "LeafValueProvider",
    "RiverAverage",
    "RiverPredictor",
    "ShowdownOracle",
    "TurnEndLeafEvaluator",
    "ValueLeafEvaluator",
    "make_leaf_evaluator",
    "predictor_kind",
    "flop_end_leaves",
    "river_average",
    "turn_end_leaves",
]
