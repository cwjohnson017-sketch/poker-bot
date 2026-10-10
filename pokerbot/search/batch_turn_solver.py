"""Range-vs-range CFR on ``B`` turn subgames at once that share one betting tree,
with turn-end leaves valued by a turn-end value net.

A turn subgame starts at the turn root (after the turn card, before any turn
action) with both players having committed the same ``c`` chips, so its public
betting tree depends only on ``c``, the button and the action spec
(:func:`turn_tree`). :class:`BatchTurnSolver` solves ``B`` such instances
together, each with its own 4-card board and its own ranges. The tree has:

* decision nodes (turn betting);
* fold terminals;
* all-in showdowns before the river (a bet and a call that put a player all
  in): the exact average over the 48 river cards, see below;
* ``VALUE`` leaves at the end of turn betting (check-check or bet-call with
  chips behind), valued by a **turn-end predictor** on both players' current
  reaches, as :class:`~.value_leaf.TurnEndLeafEvaluator` does in
  :class:`~.solver.RangeSolver`: one ``predict`` row per (leaf, instance),
  ``v_p = ev_p * blocked_sum(r_-p) * pot_leaf``.

The algorithm, layout and update order are :class:`~.batch_solver.BatchRiverSolver`'s
(DCFR or CFR+ with alternating updates, regrets on edges, segments of equal
fan-out), on the ``1128`` combos disjoint from a 4-card board. The leaf values
of the updating player are computed at the start of every backward pass from
both players' current reaches, exactly where ``RangeSolver`` calls its
leaf-value provider, so a batch solve matches one ``RangeSolver`` per
instance on the same tree. ``leaf_every > 1`` re-runs the net only on every
n-th regret update per player (cached ``ev`` re-weighted by the current
opponent mass, as ``leaf.net_every``). Root values, best responses and
exploitability are computed in the game the leaf model defines: the leaves
are evaluated on both players' reaches under the strategy being scored (the
best responder's own reach under the average strategy, as in
``RangeSolver.values``). No chance nodes, gadget or locks; no CUDA graph (the
predictor call is not capturable in general).

**All-in run-outs.** For a showdown on ``b4`` with stake ``s`` (the smaller
contribution), ``v(c) = s * sum_c' r(c') E_b[c', c]`` with

    E_b[c', c] = (1 / 44) * sum_{x not in b4} [c, c' avoid x] * sign(s_x(c) - s_x(c'))

over pairs ``c, c'`` disjoint from each other and from ``b4`` (``s_x`` the
strength on ``b4 + x``; ``44 = 52 - 4 - 4`` cards avoid two disjoint hands).
This is ``RangeSolver``'s dense all-in matrix (``solver.allin_matrix``) on the
compact combo layout. One ``[1128, 1128]`` matrix per instance serves every
all-in node of the tree (``B`` batched matmuls per update). Memory: 5.1 MB per
instance in float32 (2.6 GB at ``B = 512``), next to ``5 * N * 1128 * 4``
bytes of tree tensors (4.1 MB at ``N = 183``, the largest blueprint turn tree
at 100bb). Built once per solve from ``48 * 1128^2`` integer compares per
instance (:func:`allin_turn_matrix`); skipped when the tree has no all-in
showdown. Run-out rows on the ``O(1326)`` sorted-strength kernel instead
would need 48 rows per (all-in node, instance) per update, about 25 times the
matmul's memory traffic at the blueprint spec's ~26 all-in nodes.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from ..engine_select import get_engine
from ..env.actions import ActionSpec
from .abstract import contributions, make_state
from .batch_solver import BatchRiverSolver, Strategy, _Kernels
from .combos import NUM_CARDS, NUM_COMBOS, conflict_matrix
from .solver import SolverConfig
from .tree import DECISION, FOLD_NODE, SHOWDOWN, VALUE, SubgameTree, TreeBuilder, TreeConfig
from .value_leaf import predictor_kind

C = NUM_COMBOS
TURN_LEN = 4
CV4 = 1128  # combos disjoint from a 4-card board: C(48, 2)
RIVERS = NUM_CARDS - TURN_LEN  # river cards per turn board (48)
RIVERS_PER_PAIR = NUM_CARDS - TURN_LEN - 4  # river cards avoiding two disjoint hands (44)
_BUILD_BOARD = (0, 13, 26, 39)  # any 4 cards: the turn tree does not depend on them


def turn_tree(
    game_config: Any,
    c: int,
    spec: ActionSpec,
    button: int = 1,
    device: torch.device | str = "cpu",
) -> SubgameTree:
    """Turn-root tree of a hand where both players have committed ``c`` chips,
    with ``VALUE`` leaves at the end of turn betting (``depth_streets 0``,
    ``leaf_mode value_net``).

    ``c == big blind``: the button limps and the big blind checks; otherwise
    the button raises to ``c`` and the big blind calls. The flop is checked
    through. With ``button=1`` seat 0 (the big blind) acts first on the turn,
    so seat order is ``(OOP, IP)``. ``game_config`` is an engine ``GameConfig``.
    """
    engine = get_engine()
    c = int(c)
    board = list(_BUILD_BOARD)
    state = make_state(engine, game_config, button, board, [])
    limp = c == int(game_config.big_blind)
    try:
        state.apply(engine.Action.check_call() if limp else engine.Action.raise_to(c))
        for _ in range(3):  # the big blind checks or calls; check-check on the flop
            state.apply(engine.Action.check_call())
    except Exception as exc:
        raise ValueError(
            f"c={c} must be the big blind or a legal preflop raise-to below the stack"
        ) from exc
    if state.is_terminal or int(state.street) != 2 or contributions(state, game_config) != [c, c]:
        raise ValueError(f"c={c} does not lead to a turn decision with {c} chips each")
    tc = TreeConfig(spec=spec, depth_streets=0, max_nodes=10**7, leaf_mode="value_net")
    return TreeBuilder(
        game_config, button, board, state.history, (), tc, engine=engine, device=device
    ).build()


@torch.no_grad()
def allin_turn_matrix(
    boards4: torch.Tensor,
    cmap: torch.Tensor,
    dtype: torch.dtype = torch.float32,
    inst_step: int = 8,
    river_step: int = 8,
) -> torch.Tensor:
    """The river-averaged all-in matrices ``E [B, 1128, 1128]`` (module
    docstring) of ``boards4 [B, 4]`` on the compact layouts ``cmap [B, 1128]``
    (ascending ids of the combos disjoint from each board): ``E[b, c', c]``
    is ``(1/44) * sum_x sign(s_x(c) - s_x(c'))`` over the river cards ``x``
    that avoid both combos, zero when ``c`` and ``c'`` share a card. Counts
    are exact integers (``G - G^T``) before the scale. ``inst_step`` instances
    and ``river_step`` river cards are compared at a time (a
    ``[inst_step, river_step, 1128, 1128]`` bool temporary)."""
    from .turn_net import river_boards
    from .value_net import board_strengths

    dev = cmap.device
    B, cv = cmap.shape
    out = torch.empty(B, cv, cv, device=dev, dtype=dtype)
    if B == 0:
        return out
    rb = river_boards(boards4.to(dev).long())  # [B, 48, 5]
    conflict = conflict_matrix(dev)
    scale = 1.0 / RIVERS_PER_PAIR
    for b0 in range(0, B, inst_step):
        b1 = min(B, b0 + inst_step)
        nb = b1 - b0
        # strengths of the compact combos on each river board, -1 where a combo holds x
        s = board_strengths(rb[b0:b1].reshape(nb * RIVERS, 5)).view(nb, RIVERS, C)
        s = s.gather(2, cmap[b0:b1, None, :].expand(nb, RIVERS, cv)).to(torch.int32)
        G = torch.zeros(nb, cv, cv, device=dev, dtype=torch.int16)
        for x0 in range(0, RIVERS, river_step):
            sx = s[:, x0 : x0 + river_step]  # [nb, xs, cv]
            # G[c', c] += [c beats c' on b4 + x and c' avoids x] (c then avoids x too)
            beats = (sx[:, :, None, :] > sx[:, :, :, None]) & (sx[:, :, :, None] >= 0)
            G += beats.sum(1, dtype=torch.int16)
        cm = cmap[b0:b1]
        ok = ~conflict[cm[:, :, None], cm[:, None, :]]
        out[b0:b1] = (G - G.transpose(1, 2)).to(dtype) * ok * scale
    return out


class _TurnKernels(_Kernels):
    """Fold operators of 4-card boards and the river-averaged all-in matrices."""

    @staticmethod
    def _showdown_matrix(boards: torch.Tensor, cmap: torch.Tensor, dtype: torch.dtype):
        return allin_turn_matrix(boards, cmap, dtype)


@dataclass
class _Leaves:
    """The ``VALUE`` nodes: internal ids, both players' reach rows, the chips
    each player committed, the chips behind (``[L]`` long) and the OOP seat."""

    ids: torch.Tensor
    src: list[torch.Tensor]
    c: torch.Tensor
    stack: torch.Tensor
    pot: torch.Tensor  # [L] solver dtype, 2c
    oop: int


class BatchTurnSolver(BatchRiverSolver):
    """DCFR on ``B`` turn subgames sharing ``tree`` (from :func:`turn_tree`),
    with ``boards [B, 4]``, ``ranges [B, 2, 1326]`` (root reaches in seat
    order, any non-negative scale) and the turn-end ``predictor`` valuing the
    ``VALUE`` leaves: ``predict(boards [n, 4], ranges [n, 2, 1326] (OOP, IP),
    c [n], stack [n]) -> ev [n, 2, 1326]`` in pot units per unit of disjoint
    opponent mass (``kind == "turn_end"``, e.g. :class:`~.turn_net.TurnEndPredictor`
    or :class:`~.turn_data.RiverAveragePredictor`). A river predictor (``kind ==
    "river"``, e.g. :class:`~.value_leaf.ShowdownOracle`) is averaged over the
    river cards (wrapped in ``RiverAveragePredictor``).

    ``leaf_every``: run the net on every n-th regret update per player (other
    evaluations always run it). ``leaf_chunk``: net rows per ``predict`` call
    (whole leaves of all ``B`` instances). ``chunk_rows`` as in
    :class:`~.batch_solver.BatchRiverSolver`.

    API as :class:`~.batch_solver.BatchRiverSolver`: :meth:`solve`,
    :meth:`average_strategy` (``[N, B, 1326]`` BFS rows), :meth:`root_values`
    (``[B, 1326]`` chips) and :meth:`exploitability` (per instance, chips).
    """

    board_len = TURN_LEN
    num_valid = CV4

    def __init__(
        self,
        tree: SubgameTree,
        boards: torch.Tensor | Sequence[Sequence[int]],
        ranges: torch.Tensor,
        predictor: Any,
        cfg: SolverConfig | None = None,
        device: torch.device | str | None = None,
        chunk_rows: int = 2048,
        leaf_every: int = 1,
        leaf_chunk: int = 16384,
        cache_dtype: torch.dtype = torch.float16,
    ) -> None:
        kind = predictor_kind(predictor)
        if kind == "river":
            from .turn_data import RiverAveragePredictor

            predictor = RiverAveragePredictor(predictor)
        elif kind != "turn_end":
            raise ValueError(
                f"BatchTurnSolver values turn-end leaves: it needs a turn-end (or river) "
                f"predictor, got kind {kind!r}"
            )
        self.predictor = predictor
        self.leaf_every = max(1, int(leaf_every))
        self.leaf_chunk = int(leaf_chunk)
        self.cache_dtype = cache_dtype
        self._cached = False
        self._leaf_calls = [0, 0]
        self._leaf_cache: list[torch.Tensor | None] = [None, None]
        self._board_ids: torch.Tensor | None = None
        self._board_gen: Any = None
        self.net_calls = 0  # predict() calls
        self.net_rows = 0  # rows sent to predict()
        super().__init__(tree, boards, ranges, cfg, device, chunk_rows, cuda_graph=False)
        self.board_t = torch.tensor(self.boards, dtype=torch.long, device=self.device)

    # -- setup --------------------------------------------------------------

    @staticmethod
    def _check_tree(tree: SubgameTree) -> None:
        kinds = set(tree.kind.unique().tolist())
        if tree.root_street != 2 or not kinds <= {DECISION, FOLD_NODE, SHOWDOWN, VALUE}:
            raise ValueError(
                "BatchTurnSolver needs a turn tree (root on the turn, depth_streets 0, "
                "leaf_mode value_net: only decision, fold, all-in showdown and value nodes), "
                f"got root street {tree.root_street} and kinds {sorted(kinds)}"
            )
        if int(tree.kind[0]) != DECISION:
            raise ValueError("the root of the turn tree must be a decision node")
        if any(len(b) != TURN_LEN for b in tree.boards):
            raise ValueError("every node of a turn tree must be on the 4-card board")

    def _kernels(self, boards: torch.Tensor) -> _Kernels:
        return _TurnKernels(boards, self.cmap, self.dtype, showdown=self.showdowns is not None)

    def _layout(self, tree: SubgameTree) -> None:
        super()._layout(tree)
        kind = tree.kind.tolist()
        nodes = [n for n in range(self.N) if kind[n] == VALUE]
        self.leaves: _Leaves | None = None
        if not nodes:
            return
        dev = self.device
        ids = torch.tensor(nodes, dtype=torch.long, device=dev)
        contrib = tree.contrib.to(dev)[ids]
        if bool((contrib[:, 0] != contrib[:, 1]).any()):
            raise ValueError("value-net leaves need equal contributions (a river root)")
        oop = {1 - int(tree.states[n].button) for n in nodes}
        if len(oop) != 1:
            raise ValueError("value-net leaves of one tree must share the button")

        def lt(x: list[int]) -> torch.Tensor:
            return torch.tensor(x, dtype=torch.long, device=dev)

        c = contrib[:, 0].clone()
        self.leaves = _Leaves(
            ids=lt([self._inv[n] for n in nodes]),
            src=[lt([self._src[p][n] for n in nodes]) for p in (0, 1)],
            c=c,
            stack=lt([min(int(s) for s in tree.states[n].stacks) for n in nodes]),
            pot=2.0 * c.to(self.dtype),
            oop=oop.pop(),
        )

    @property
    def num_leaves(self) -> int:
        return 0 if self.leaves is None else int(self.leaves.ids.numel())

    def reset_cache(self) -> None:
        """Forget the cached leaf ``ev`` of ``leaf_every``."""
        self._leaf_calls = [0, 0]
        self._leaf_cache = [None, None]

    # -- leaves -------------------------------------------------------------

    def _predictor_board_ids(self) -> torch.Tensor | None:
        """``[B]`` board ids in a predictor with a board cache (``board_ids`` /
        ``predict_ids``), looked up again only when its cache was reset."""
        pred = self.predictor
        if not hasattr(pred, "predict_ids"):
            return None
        gen = getattr(getattr(pred, "cache", None), "generation", None)
        if self._board_ids is None or gen != self._board_gen:
            self._board_ids = pred.board_ids(self.board_t)
            self._board_gen = getattr(getattr(pred, "cache", None), "generation", None)
        return self._board_ids

    def _leaf_ev(self, player: int, r: list[torch.Tensor]) -> torch.Tensor:
        """``ev`` of ``player`` (pot units) at every (leaf, instance) from the
        net on both players' reaches ``r[p] [L, B, 1128]``, ``[L, B, 1128]``."""
        lv = self.leaves
        L, B, cv = r[0].shape
        col = 0 if player == lv.oop else 1
        seats = (lv.oop, 1 - lv.oop)  # (OOP, IP)
        out = r[0].new_empty(L, B, cv)
        ids = self._predictor_board_ids()
        per = max(1, self.leaf_chunk // max(1, B))
        for l0 in range(0, L, per):
            l1 = min(L, l0 + per)
            Lc = l1 - l0
            n = Lc * B
            idx = self.cmap[None].expand(Lc, B, cv)
            full = r[0].new_zeros(Lc, B, 2, C)
            for j, seat in enumerate(seats):
                full[:, :, j].scatter_(2, idx, r[seat][l0:l1])
            c = lv.c[l0:l1, None].expand(Lc, B).reshape(n)
            stack = lv.stack[l0:l1, None].expand(Lc, B).reshape(n)
            with torch.no_grad():
                if ids is None:
                    boards = self.board_t[None].expand(Lc, B, TURN_LEN).reshape(n, TURN_LEN)
                    ev = self.predictor.predict(boards, full.view(n, 2, C), c, stack)
                else:
                    rows = ids[None].expand(Lc, B).reshape(n)
                    ev = self.predictor.predict_ids(rows, full.view(n, 2, C), c, stack)
            self.net_calls += 1
            self.net_rows += n
            ev = ev.view(Lc, B, 2, C)[:, :, col].to(self.dtype)
            out[l0:l1] = ev.gather(2, idx)
        return out

    def _leaf_values(self, player: int, cached: bool) -> torch.Tensor:
        """``player``'s values ``[L, B, 1128]`` (chips) at the leaves from both
        players' reaches in the buffer: ``ev * m_-p * pot``."""
        lv = self.leaves
        R = self.reach
        r = [R.index_select(0, lv.src[p]) for p in (0, 1)]
        m = self.kern.fold(r[1 - player].clone())  # opponent mass disjoint from each combo
        refresh, cache = True, None
        if cached and self.leaf_every > 1:
            k = self._leaf_calls[player]
            self._leaf_calls[player] += 1
            cache = self._leaf_cache[player]
            refresh = cache is None or k % self.leaf_every == 0
        if refresh:
            ev = self._leaf_ev(player, r)
            if cached and self.leaf_every > 1:
                self._leaf_cache[player] = ev.to(self.cache_dtype, copy=True)
        else:
            ev = cache.to(self.dtype)
        w = ev.mul_(m).mul_(lv.pot[:, None, None])
        return torch.nan_to_num_(w, nan=0.0, posinf=0.0, neginf=0.0)

    # -- passes -------------------------------------------------------------

    def _terminals(self, i: int, v: torch.Tensor) -> None:
        super()._terminals(i, v)
        if self.leaves is not None:
            v.index_copy_(0, self.leaves.ids, self._leaf_values(i, self._cached))

    def _backward(self, i: int, sigma: Strategy, mode: str) -> torch.Tensor:
        self._cached = mode == "update"  # only regret updates may reuse cached leaf ev
        try:
            return super()._backward(i, sigma, mode)
        finally:
            self._cached = False

    def _values(self, player: int, sigma: Strategy, root: torch.Tensor, mode: str):
        """Root values ``[B, 1128]`` of ``player``; both reaches follow ``sigma``
        (the leaves need the player's own reach too)."""
        for p in (0, 1):
            self._reach_ok[p] = False
            self._forward(p, sigma, root[:, p])
        return self._backward(player, sigma, mode)[0]


__all__ = ["BatchTurnSolver", "allin_turn_matrix", "turn_tree"]
