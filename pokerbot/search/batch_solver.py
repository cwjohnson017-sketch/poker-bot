"""Range-vs-range CFR on ``B`` river subgames at once that share one betting tree.

On the river, betting never depends on the cards, so every river subgame in
which both players have committed the same ``c`` chips has the same public
tree. :class:`BatchRiverSolver` solves ``B`` such instances together, each
with its own 5-card board and its own ranges.

The algorithm is :class:`~pokerbot.search.solver.RangeSolver`'s, instance by
instance: alternating updates (player 0, then player 1), a forward pass for
the reaches, fold and showdown values of the updating player against the
opponent's reach, a backward pass accumulating instantaneous regrets at the
updater's nodes, the strategy sum weighted by the updater's reach, and
regret matching with a uniform fallback. DCFR discounts regrets exactly as
``RangeSolver._discount`` (with its float32 discount factors, so float64
solves match it to rounding). The strategy-sum discount is applied lazily:
weighting iteration ``t`` by ``t^gamma`` equals multiplying the sum by
``((t - 1) / t)^gamma`` before every add, up to one global factor that
cancels in the average. CFR+ (regrets floored at zero, linear averaging) is
also supported. No chance nodes, depth-limit leaves, gadget or locks.

Layout:

* Combos: each instance keeps only the ``1081`` combos disjoint from its
  board (``cmap[b]``, ascending combo ids), so every per-combo tensor is
  ``[rows, B, 1081]``. Inputs and outputs use the full 1326 combos.
* Regrets, strategy sums and the current strategy live on **edges**, stored
  at the row of the child node (the root row is unused).
* Internal node order: the root, then every node reached by an action of
  player 0 (sorted by depth, then by the number of siblings, then BFS id),
  then every node reached by an action of player 1. A *segment* is a run of
  ``m`` parents with ``k`` children each, all at one depth and with one
  acting player, so its edges view as ``[m, k, B, 1081]``: regret matching
  and the backward pass reduce over dim 1, with no scatter.
* Reaches share one ``[N + 1, B, 1081]`` buffer: player ``p``'s reach is
  stored only on the rows of ``p``'s edges (an opponent action leaves it
  unchanged), and rows ``0`` and ``N`` hold the root reaches of players 0
  and 1. ``src[p][n]`` is the row that holds ``p``'s reach at node ``n``.
  An update recomputes only the reach of the player whose strategy changed.
* Values of the updating player are an ``[N, B, 1081]`` buffer.

Terminal values, for all fold (or all showdown) nodes of all instances at
once. Both are linear in the opponent's reach, so its rows are gathered
already scaled by the stakes (one ``embedding_bag``):

* fold: ``+-stake * (opponent mass disjoint from c)``, two batched matmuls
  through the per-instance ``[1081, 52]`` card incidence (inclusion-exclusion
  as in :func:`~pokerbot.search.combos.blocked_sum`);
* showdown: ``min(contrib) * (r @ K_b)`` with a dense per-instance
  ``K_b[c', c] = sign(s(c) - s(c'))`` over disjoint pairs, built once
  (``B * 1081^2`` floats: 4.7 MB per instance in fp32). One batched GEMM per
  update reads every ``K_b`` once. On a GPU that is several times cheaper
  than the ``O(1326)``-per-row sorted-strength kernel of
  :mod:`pokerbot.search.showdown`, which makes a few dozen passes over every
  row.

Speed: the tree passes work on segments of at most ``chunk_rows`` rows of
1081 combos, so their temporaries stay in the GPU's L2 cache, and on CUDA
every iteration from the third on replays one captured CUDA graph (the
discounts and the strategy weight are device scalars).

``best_response`` values and :meth:`exploitability` are exact, per instance.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from ..engine_select import get_engine
from ..env.actions import ActionSpec
from .abstract import contributions, make_state
from .combos import NUM_COMBOS, combo_table, conflict_matrix, valid_masks
from .showdown import combo_strengths
from .solver import SolverConfig
from .tree import DECISION, FOLD_NODE, SHOWDOWN, SubgameTree, TreeBuilder, TreeConfig

C = NUM_COMBOS
CV = 1081  # combos disjoint from a 5-card board: C(47, 2)
_BUILD_BOARD = (0, 13, 26, 39, 1)  # any 5 cards: the river tree does not depend on them


def river_tree(
    game_config: Any,
    c: int,
    spec: ActionSpec,
    button: int = 1,
    device: torch.device | str = "cpu",
) -> SubgameTree:
    """River-root tree of a hand where both players have committed ``c`` chips.

    ``c == big blind``: the button limps and the big blind checks; otherwise
    the button raises to ``c`` and the big blind calls. Flop and turn are
    checked through. With ``button=1`` seat 0 (the big blind) acts first on the
    river, so seat order is ``(OOP, IP)``.
    """
    engine = get_engine()
    c = int(c)
    board = list(_BUILD_BOARD)
    state = make_state(engine, game_config, button, board, [])
    limp = c == int(game_config.big_blind)
    try:
        state.apply(engine.Action.check_call() if limp else engine.Action.raise_to(c))
        for _ in range(5):  # the big blind checks or calls; check-check on flop and turn
            state.apply(engine.Action.check_call())
    except Exception as exc:
        raise ValueError(
            f"c={c} must be the big blind or a legal preflop raise-to below the stack"
        ) from exc
    if state.is_terminal or int(state.street) != 3 or contributions(state, game_config) != [c, c]:
        raise ValueError(f"c={c} does not lead to a river decision with {c} chips each")
    tc = TreeConfig(spec=spec, max_nodes=10**7)
    return TreeBuilder(
        game_config, button, board, state.history, (), tc, engine=engine, device=device
    ).build()


@dataclass
class _Seg:
    """``m`` parents with ``k`` children each: internal child ids ``s .. e-1``."""

    s: int
    e: int
    m: int
    k: int
    p: int  # actor at the parents
    depth: int  # depth of the children
    par: torch.Tensor  # [m] internal parent ids
    src_par: torch.Tensor  # [m] reach rows of the actor at the parents
    orig: torch.Tensor  # [m * k] BFS ids of the children


Strategy = Callable[[_Seg], torch.Tensor]  # segment -> its [m, k, B, 1081] edge rows


@dataclass
class _Terminal:
    """All fold (or all showdown) nodes: internal ids, the reach rows of each
    player there, and the value coefficient for each updating player."""

    ids: torch.Tensor  # [n]
    src: list[torch.Tensor]  # per player, [n]
    coef: list[torch.Tensor]  # per updating player, [n, 1]


class _Kernels:
    """Per-instance fold and showdown operators on the compact combo layout:
    opponent reach rows ``r [n, B, 1081]`` -> values ``[n, B, 1081]``. The
    batched matmuls run over instances on transposed views."""

    def __init__(self, boards: torch.Tensor, cmap: torch.Tensor, dtype: torch.dtype) -> None:
        dev = cmap.device
        B = cmap.shape[0]
        inc = torch.zeros(B, CV, 52, device=dev, dtype=dtype)
        inc.scatter_(2, combo_table(dev)[cmap], 1.0)
        self.inc = inc
        # total - S[x1] - S[x2] = S @ (1/2 - inc^T), S = per-card mass
        self.fold_m = (0.5 - inc).transpose(1, 2).contiguous()
        strength = combo_strengths(boards).gather(1, cmap)  # [B, CV]
        conflict = conflict_matrix(dev)
        self.K = torch.empty(B, CV, CV, device=dev, dtype=dtype)
        step = 16
        for b in range(0, B, step):
            s = strength[b : b + step]
            cm = cmap[b : b + step]
            ok = ~conflict[cm[:, :, None], cm[:, None, :]]
            self.K[b : b + step] = torch.sign(s[:, None, :] - s[:, :, None]) * ok

    def fold(self, r: torch.Tensor) -> torch.Tensor:
        """Opponent mass disjoint from each combo (in place: ``r`` becomes the
        result, ``[n, B, 1081]``)."""
        rt = r.transpose(0, 1)
        rt.baddbmm_(torch.bmm(rt, self.inc), self.fold_m)
        return r

    def showdown(self, r: torch.Tensor) -> torch.Tensor:
        """Opponent mass that ``c`` beats minus the mass that beats ``c``
        (``[n, B, 1081]``, a transposed view)."""
        return torch.bmm(r.transpose(0, 1), self.K).transpose(0, 1)


class BatchRiverSolver:
    def __init__(
        self,
        tree: SubgameTree,
        boards: torch.Tensor | Sequence[Sequence[int]],
        ranges: torch.Tensor,
        cfg: SolverConfig | None = None,
        device: torch.device | str | None = None,
        chunk_rows: int = 2048,
        cuda_graph: bool = True,
    ) -> None:
        """``chunk_rows``: rows of 1081 combos per elementwise kernel in the
        tree passes (small chunks keep the temporaries in the GPU's L2 cache).
        ``cuda_graph``: on CUDA, replay iterations from one captured graph."""
        self.cfg = cfg or SolverConfig()
        if self.cfg.algorithm not in ("dcfr", "cfr+"):
            raise ValueError(f"unknown algorithm {self.cfg.algorithm!r}")
        dev = torch.device(device) if device is not None else tree.device
        self.device = dev
        self.dtype = getattr(torch, self.cfg.dtype)
        self.tree = tree
        self._check_tree(tree)
        bt = torch.as_tensor(
            boards.tolist() if isinstance(boards, torch.Tensor) else [list(b) for b in boards],
            dtype=torch.long,
        )
        if bt.dim() != 2 or bt.shape[1] != 5:
            raise ValueError(f"boards must be [B, 5], got {tuple(bt.shape)}")
        srt = bt.sort(1).values
        if bool((bt < 0).any() | (bt > 51).any() | (srt[:, 1:] == srt[:, :-1]).any()):
            raise ValueError("every board needs 5 distinct cards in 0..51")
        self.boards = [tuple(int(x) for x in b) for b in bt.tolist()]
        B = self.B = len(self.boards)
        self.valid = valid_masks(self.boards, dev)  # [B, C]
        self.cmap = self.valid.nonzero()[:, 1].view(B, CV)
        self.chunk_rows = int(chunk_rows)
        self.pot = int(tree.contrib[0].sum())
        self.ranges = self._compact_ranges(ranges)
        self._layout(tree)
        self.kern = _Kernels(bt.to(dev), self.cmap, self.dtype)
        N = self.N
        z = dict(device=dev, dtype=self.dtype)
        self.regret = torch.zeros(N, B, CV, **z)
        self.strat_sum = torch.zeros(N, B, CV, **z)
        self.sigma = torch.empty(N, B, CV, **z)
        self.sigma[0] = 1
        for g in self.segs:
            self.sigma[g.s : g.e] = 1.0 / g.k
        self.reach = torch.empty(N + 1, B, CV, **z)
        self.v = torch.empty(N, B, CV, **z)
        self._reach_ok = [False, False]
        self._scal = torch.zeros(3, **z).unbind(0)  # discounts a, b; strategy weight w
        self._use_graph = bool(cuda_graph) and dev.type == "cuda"
        self._graph: torch.cuda.CUDAGraph | None = None
        self.t = 0
        self.iterations_done = 0
        self.solve_time = 0.0

    # -- setup --------------------------------------------------------------

    @staticmethod
    def _check_tree(tree: SubgameTree) -> None:
        kinds = set(tree.kind.unique().tolist())
        if tree.root_street != 3 or not kinds <= {DECISION, FOLD_NODE, SHOWDOWN}:
            raise ValueError(
                "BatchRiverSolver needs a river tree (root on the river; only decision, "
                f"fold and showdown nodes), got root street {tree.root_street} and kinds "
                f"{sorted(kinds)}"
            )
        if int(tree.kind[0]) != DECISION:
            raise ValueError("the root of the river tree must be a decision node")

    def _compact_ranges(self, ranges: torch.Tensor) -> torch.Tensor:
        r = torch.as_tensor(ranges).to(self.device, self.dtype)
        if tuple(r.shape) != (self.B, 2, C):
            raise ValueError(f"ranges must be [{self.B}, 2, {C}], got {tuple(r.shape)}")
        return r.gather(2, self.cmap[:, None].expand(-1, 2, -1))

    def _layout(self, tree: SubgameTree) -> None:
        dev = self.device
        parent = tree.parent.tolist()
        actor = tree.actor.tolist()
        depth = tree.depth.tolist()
        kind = tree.kind.tolist()
        nch = tree.num_children.tolist()
        N = self.N = len(parent)
        pa = [actor[parent[n]] if n else -1 for n in range(N)]

        def key(n: int) -> tuple:
            return (pa[n], depth[n], nch[parent[n]] if n else 0)

        perm = sorted(range(N), key=lambda n: (*key(n), n))
        inv = [0] * N
        for i, n in enumerate(perm):
            inv[n] = i

        def lt(x: list[int]) -> torch.Tensor:
            return torch.tensor(x, dtype=torch.long, device=dev)

        self.perm = lt(perm)  # internal -> BFS id
        # src[p][n]: row of the reach buffer holding p's reach at (BFS) node n
        src = [[0] * N, [N] * N]
        for n in range(1, N):
            for p in (0, 1):
                src[p][n] = inv[n] if pa[n] == p else src[p][parent[n]]
        self.segs: list[_Seg] = []
        i = 1
        while i < N:
            j = i
            while j < N and key(perm[j]) == key(perm[i]):
                j += 1
            p, d, k = key(perm[i])
            per = max(1, self.chunk_rows // (k * self.B))
            for s in range(i, j, per * k):
                e = min(j, s + per * k)
                pars = [parent[perm[x]] for x in range(s, e, k)]
                self.segs.append(
                    _Seg(
                        s=s,
                        e=e,
                        m=(e - s) // k,
                        k=k,
                        p=p,
                        depth=d,
                        par=lt([inv[q] for q in pars]),
                        src_par=lt([src[p][q] for q in pars]),
                        orig=lt(perm[s:e]),
                    )
                )
            i = j
        self.max_depth = max(depth)
        self.by_depth = {
            d: [g for g in self.segs if g.depth == d] for d in range(1, self.max_depth + 1)
        }
        contrib = tree.contrib.tolist()
        folder = tree.folder.tolist()

        def terminal(kd: int) -> _Terminal | None:
            ids = [n for n in range(N) if kind[n] == kd]
            if not ids:
                return None
            if kd == FOLD_NODE:  # +-stake of the folder, for updating player 0, then 1
                c0 = [contrib[n][folder[n]] * (-1 if folder[n] == 0 else 1) for n in ids]
                coef = [c0, [-x for x in c0]]
            else:
                coef = [[min(contrib[n]) for n in ids]] * 2
            return _Terminal(
                ids=lt([inv[n] for n in ids]),
                src=[lt([src[p][n] for n in ids]) for p in (0, 1)],
                coef=[torch.tensor(x, device=dev, dtype=self.dtype).view(-1, 1) for x in coef],
            )

        self.folds = terminal(FOLD_NODE)
        self.showdowns = terminal(SHOWDOWN)

    # -- passes -------------------------------------------------------------

    def _view(self, x: torch.Tensor, g: _Seg) -> torch.Tensor:
        return x[g.s : g.e].view(g.m, g.k, self.B, CV)

    # A strategy is a function of a segment returning its ``[m, k, B, 1081]`` rows.

    def _current(self, g: _Seg) -> torch.Tensor:
        return self._view(self.sigma, g)

    def _average(self, g: _Seg) -> torch.Tensor:
        S = self._view(self.strat_sum, g)
        tot = S.sum(1, keepdim=True)
        return torch.where(tot > 0, S / tot.clamp(min=1e-30), self._current(g))

    def _forward(self, p: int, sigma: Strategy, root: torch.Tensor) -> None:
        """Player ``p``'s reach on the rows of ``p``'s edges (and its root row)."""
        R = self.reach
        R[0 if p == 0 else self.N] = root
        for d in range(1, self.max_depth + 1):
            for g in self.by_depth[d]:
                if g.p == p:
                    par = R.index_select(0, g.src_par).unsqueeze(1)
                    torch.mul(par, sigma(g), out=self._view(R, g))

    def _terminals(self, i: int, v: torch.Tensor) -> None:
        """Values of player ``i`` at the fold and showdown nodes, against the
        opponent's reach in the reach buffer. Both are linear in that reach,
        so the gathered rows are scaled by the stakes (one ``embedding_bag``)."""
        R = self.reach.view(self.N + 1, -1)
        for term, op in ((self.folds, self.kern.fold), (self.showdowns, self.kern.showdown)):
            if term is not None:
                r = F.embedding_bag(
                    term.src[1 - i][:, None], R, per_sample_weights=term.coef[i], mode="sum"
                )
                v.index_copy_(0, term.ids, op(r.view(-1, self.B, CV)))

    def _backward(self, i: int, sigma: Strategy, mode: str) -> torch.Tensor:
        """Values of player ``i`` against the opponent's reach in the buffer.
        ``mode``: "value" plays ``sigma``, "br" maximises at ``i``'s nodes,
        "update" plays ``sigma`` and also updates ``i``'s regrets, strategy sum
        and current strategy."""
        v = self.v
        self._terminals(i, v)
        for d in range(self.max_depth, 0, -1):
            for g in self.by_depth[d]:
                cv = self._view(v, g)
                if g.p != i:
                    out = cv.sum(1)
                elif mode == "br":
                    out = cv.amax(1)
                else:
                    out = (cv * sigma(g)).sum(1)
                v.index_copy_(0, g.par, out)
            if mode == "update":
                for g in self.by_depth[d]:
                    if g.p == i:
                        self._update_segment(g, i)
        return v

    # -- CFR ----------------------------------------------------------------

    def _set_scalars(self) -> None:
        """Iteration ``t``'s regret discounts and strategy-sum weight, as device
        scalars (so that a captured CUDA graph reads the current values)."""
        cfg = self.cfg
        t = self.t
        a = b = 1.0
        if cfg.algorithm == "dcfr" and t >= 2:
            tp = t - 1
            a = tp**cfg.alpha / (tp**cfg.alpha + 1)
            b = tp**cfg.beta / (tp**cfg.beta + 1)
            # RangeSolver's ``torch.where(R > 0, a, b)`` holds them in float32
            a, b = torch.tensor([a, b], dtype=torch.float32).tolist()
        w = float(t) if cfg.algorithm == "cfr+" else float(t) ** cfg.gamma
        for x, val in zip(self._scal, (a, b, w), strict=True):
            x.fill_(val)

    def _update_segment(self, g: _Seg, i: int) -> None:
        """Discount, add instantaneous regrets, add to the strategy sum and
        regret-match the edges of segment ``g`` (``i``'s; values complete)."""
        a, b, w = self._scal
        v = self.v
        R = self._view(self.regret, g)
        sig = self._view(self.sigma, g)
        inst = self._view(v, g) - v.index_select(0, g.par).unsqueeze(1)
        if self.cfg.algorithm == "dcfr":
            torch.addcmul(inst, R, torch.where(R > 0, a, b), out=R)
        else:
            R += inst
        reach = self.reach.index_select(0, g.src_par).unsqueeze(1)
        self._view(self.strat_sum, g).addcmul_(reach.mul_(w), sig)
        if self.cfg.algorithm == "cfr+":
            R.clamp_(min=0)
        torch.clamp(R, min=0, out=sig)
        tot = sig.sum(1, keepdim=True)
        pos = tot > 0
        scale = torch.where(pos, 1.0 / tot.clamp(min=1e-30), 0.0)
        torch.addcmul((~pos).to(self.dtype) / g.k, sig, scale, out=sig)

    def _update(self, i: int) -> None:
        for p in (0, 1):
            if not self._reach_ok[p]:
                self._forward(p, self._current, self.ranges[:, p])
                self._reach_ok[p] = True
        self._backward(i, self._current, "update")
        self._reach_ok[i] = False

    def _iteration(self) -> None:
        self._update(0)
        self._update(1)

    def _replay(self) -> None:
        """One iteration from the captured CUDA graph (capturing it first).
        The graph starts from player 0's reach being current."""
        if not self._reach_ok[0]:
            self._forward(0, self._current, self.ranges[:, 0])
            self._reach_ok[0] = True
        self._reach_ok[1] = False
        if self._graph is None:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self._iteration()
            self._graph = graph
        self._graph.replay()

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @torch.no_grad()
    def solve(self, iterations: int) -> dict:
        """Run ``iterations`` more iterations. On CUDA, iterations from the
        third on replay one captured CUDA graph (unless ``cuda_graph=False``)."""
        start = time.perf_counter()
        for _ in range(int(iterations)):
            self.t += 1
            self._set_scalars()
            if self._use_graph and self.t >= 3:
                self._replay()
            elif self._use_graph and self.t == 2:  # warm-up on a side stream
                side = torch.cuda.Stream(self.device)
                side.wait_stream(torch.cuda.current_stream(self.device))
                with torch.cuda.stream(side):
                    self._iteration()
                torch.cuda.current_stream(self.device).wait_stream(side)
            else:
                self._iteration()
        self._sync()
        dt = time.perf_counter() - start
        self.iterations_done += int(iterations)
        self.solve_time += dt
        return {"iterations": int(iterations), "seconds": dt}

    # -- results ------------------------------------------------------------

    def _to_full(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, 1081] -> [B, 1326]``, zero on board conflicts."""
        return x.new_zeros(*x.shape[:-1], C).scatter_(-1, self.cmap.expand_as(x), x)

    def _strategy_out(self, sigma: Strategy) -> torch.Tensor:
        """``[N, B, 1326]`` in BFS order, uniform for combos that conflict
        with the board, root row 1."""
        out = self.sigma.new_empty(self.N, self.B, C)
        out[0] = 1
        for g in self.segs:
            rows = sigma(g).reshape(g.e - g.s, self.B, CV)
            full = rows.new_full((g.e - g.s, self.B, C), 1.0 / g.k)
            full.scatter_(2, self.cmap.expand_as(rows), rows)
            out.index_copy_(0, g.orig, full)
        return out

    @torch.no_grad()
    def average_strategy(self) -> torch.Tensor:
        """``[N, B, 1326]``: entry ``n`` is the probability that the actor at
        ``parent(n)`` takes the action into ``n`` (root row 1), BFS node ids."""
        return self._strategy_out(self._average)

    @torch.no_grad()
    def current_strategy(self) -> torch.Tensor:
        return self._strategy_out(self._current)

    def _strategy_in(self, sigma: torch.Tensor | None) -> Strategy:
        if sigma is None:
            return self._average
        sigma = torch.as_tensor(sigma).to(self.device, self.dtype)
        if tuple(sigma.shape) != (self.N, self.B, C):
            raise ValueError(f"sigma must be [{self.N}, {self.B}, {C}]")
        x = sigma.index_select(0, self.perm).gather(2, self.cmap.expand(self.N, -1, -1))
        return lambda g: self._view(x, g)

    def _values(self, player: int, sigma: Strategy, root: torch.Tensor, mode: str):
        """Root values ``[B, 1081]`` of ``player`` (reach buffer overwritten)."""
        o = 1 - player
        self._reach_ok[o] = False
        self._forward(o, sigma, root[:, o])
        return self._backward(player, sigma, mode)[0]

    @torch.no_grad()
    def root_values(
        self,
        player: int,
        sigma: torch.Tensor | None = None,
        best_response: bool = False,
        ranges: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``[B, 1326]`` counterfactual values in chips at the root for
        ``player``: ``v(c) = sum_{c'} pi_opp(c') E[u_player | c, c']`` over
        ``c'`` disjoint from ``c`` and the board, against ``sigma`` (``[N, B,
        1326]`` as from :meth:`average_strategy`; default the average strategy),
        or a best response to it when ``best_response``. ``ranges [B, 2, 1326]``
        overrides the root reaches. Zero for combos that hit the board."""
        sig = self._strategy_in(sigma)
        root = self.ranges if ranges is None else self._compact_ranges(ranges)
        mode = "br" if best_response else "value"
        return self._to_full(self._values(player, sig, root, mode))

    @torch.no_grad()
    def exploitability(self, ranges: torch.Tensor | None = None) -> dict:
        """Per instance, against the average strategy: best-response values
        ``br [B, 2]`` and values ``ev [B, 2]`` of each player (range-weighted,
        divided by the pair mass), ``nashconv [B]`` and ``exploitability [B]``
        (half of it), all in chips, plus ``pot`` (``2c``)."""
        sigma = self._average
        root = self.ranges if ranges is None else self._compact_ranges(ranges)
        Z = (root[:, 0] * self.kern.fold(root[None, :, 1].clone())[0]).sum(-1)
        br = torch.zeros(self.B, 2, device=self.device, dtype=self.dtype)
        ev = torch.zeros_like(br)
        for p in (0, 1):
            br[:, p] = (root[:, p] * self._values(p, sigma, root, "br")).sum(-1)
            ev[:, p] = (root[:, p] * self._backward(p, sigma, "value")[0]).sum(-1)
        ok = Z[:, None] > 0
        Zs = Z.clamp(min=1e-30)[:, None]
        br = torch.where(ok, br / Zs, 0.0)
        ev = torch.where(ok, ev / Zs, 0.0)
        nashconv = br.sum(1)
        return {
            "br": br,
            "ev": ev,
            "nashconv": nashconv,
            "exploitability": nashconv / 2,
            "pot": self.pot,
        }


__all__ = ["BatchRiverSolver", "river_tree"]
