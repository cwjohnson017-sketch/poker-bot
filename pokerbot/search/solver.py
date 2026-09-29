"""Range-vs-range CFR (DCFR or CFR+) over a :class:`~pokerbot.search.tree.SubgameTree`.

Every player's range is a vector over the 1326 hole combos. Per decision
node, regrets, the current strategy and the strategy sum are ``[A, 1326]``
slices of ``[D, A, 1326]`` tensors (``D`` decision nodes, ``A`` the widest
action set; player 0's nodes first). One iteration updates player 0, then
player 1 (alternating updates); each update is

1. a forward pass computing both players' reach ``pi[2, N, 1326]`` level by
   level (a child's reach is the parent's reach times the actor's strategy
   for that action, or times the "combo avoids the dealt card" mask at a
   chance node);
2. terminal values for the updating player ``i`` against ``pi_-i`` (fold,
   showdown, all-in run-outs and depth-limit continuations);
3. a backward pass summing children into parents level by level
   (``sigma``-weighted at ``i``'s nodes, plain sums at the opponent's nodes,
   ``chance_weight`` and card masks at chance nodes), accumulating
   instantaneous regrets ``v(child) - v(node)`` at ``i``'s nodes;
4. strategy-sum accumulation weighted by ``pi_i`` and regret matching.

Counterfactual values are the usual vector form:
``v_i(n)(c) = sum_{c' disjoint from c} pi_-i(n)(c') * E[u_i | c, c', n]``.
A chance child's reach is masked by the dealt card and its value enters the
parent with weight ``1 / (52 - |board| - 4)`` (the number of cards that can
come given both hands), so card removal is exact.

DCFR (Brown & Sandholm 2019) with ``alpha = 1.5, beta = 0, gamma = 2``:
after iteration ``t`` positive regrets are multiplied by
``t^a / (t^a + 1)``, negative ones by ``t^b / (t^b + 1)`` and the strategy
sum by ``(t / (t + 1))^g``. CFR+: regrets floored at zero, strategy sum
weighted by ``t`` (linear averaging).

Terminal kernels:

* fold: ``+-stake * (opponent mass disjoint from c)`` (inclusion-exclusion);
* showdown on a full board: the sorted-strength trick of
  :mod:`pokerbot.search.showdown`, ``O(n)`` per node per iteration;
* all-in before the river: a dense ``[1326, 1326]`` equity-sign matrix per
  distinct board, averaged over enumerated (or sampled) run-outs and built
  once per solve, applied as one matmul for all nodes on that board;
* continuation (depth-limit leaf): blueprint rollouts, see
  :mod:`pokerbot.search.leaf`.

Safe resolving: when a ``gadget`` is given, its ``player`` starts at a
terminate/enter choice per combo (see :mod:`pokerbot.search.gadget`); their
reach at the root is ``prior * sigma_enter``.
"""

from __future__ import annotations

import itertools
import math
import random
import time
from dataclasses import dataclass
from typing import Any

import torch

from .combos import NUM_COMBOS, avoids_card, blocked_sum, conflict_matrix, valid_masks
from .leaf import RolloutSet
from .showdown import ShowdownTables, combo_strengths, fold_values
from .tree import CHANCE, CONTINUATION, DECISION, FOLD_NODE, LEAF, SHOWDOWN, SubgameTree

C = NUM_COMBOS


@dataclass
class SolverConfig:
    algorithm: str = "dcfr"  # "dcfr" | "cfr+"
    alpha: float = 1.5
    beta: float = 0.0
    gamma: float = 2.0
    iterations: int = 1000
    time_budget: float | None = None  # seconds for the iteration loop
    max_runouts: int = 48  # all-in run-outs per board before sampling
    allin_mode: str = "auto"  # "dense" | "runouts" | "auto"
    dense_min_nodes: int = 4  # auto: dense on CUDA, or on CPU for boards with this many nodes
    chunk_rows: int = 4096
    dtype: str = "float32"
    seed: int = 0


def runouts_for(
    board: tuple[int, ...], max_runouts: int, seed: int
) -> tuple[list[tuple[int, ...]], float]:
    """Run-outs of an incomplete board (all, or ``max_runouts`` sampled without
    replacement) and the scale ``|All| / (|S| * N_k)`` that makes their sum an
    unbiased estimate of the expectation over run-outs disjoint from both hands."""
    k = 5 - len(board)
    avail = [c for c in range(52) if c not in board]
    total = math.comb(len(avail), k)
    n_k = math.comb(len(avail) - 4, k)
    if total <= max_runouts:
        runouts = list(itertools.combinations(avail, k))
    else:
        rng = random.Random(hash((seed, board)) & 0xFFFFFFFF)
        seen: set = set()
        while len(seen) < max_runouts:
            seen.add(tuple(sorted(rng.sample(avail, k))))
        runouts = sorted(seen)
    return runouts, total / (len(runouts) * n_k)


def allin_matrix(
    board: tuple[int, ...], max_runouts: int, device: torch.device, dtype: torch.dtype, seed: int
) -> torch.Tensor:
    """``E[c, c']``: expected ``sign(s(c) - s(c'))`` over the run-outs of an
    incomplete board, zero for conflicting or blocked pairs, so that the
    all-in value of ``c`` is ``stake * (E @ pi)(c)``."""
    runouts, scale = runouts_for(board, max_runouts, seed)
    full = torch.tensor([list(board) + list(r) for r in runouts], device=device)
    S = combo_strengths(full)  # [R, C], -1 for blocked combos
    # G[c, c'] = number of run-outs where c beats c' (both unblocked); E = G - G^T.
    # Blocked combos have strength -1: requiring the loser to be unblocked is enough.
    G = torch.zeros(C, C, device=device, dtype=torch.int32)
    for s in range(0, S.shape[0], 16):
        st = S[s : s + 16]
        valid = st >= 0
        G += ((st[:, :, None] > st[:, None, :]) & valid[:, None, :]).sum(0, dtype=torch.int32)
    E = (G - G.t()).to(dtype)
    E *= ~conflict_matrix(device)
    return E * scale


class TerminalEvaluator:
    """Values of every terminal-like node for one player given the opponent's reach."""

    def __init__(self, tree: SubgameTree, cfg: SolverConfig, rollouts: RolloutSet | None):
        dev = tree.device
        self.dtype = getattr(torch, cfg.dtype)
        self.chunk = cfg.chunk_rows
        kind = tree.kind
        blen = torch.tensor([len(b) for b in tree.boards], device=dev)
        self.board_valid = valid_masks(tree.boards, dev)
        # folds
        self.fold_ids = (kind == FOLD_NODE).nonzero().flatten()
        self.fold_folder = tree.folder[self.fold_ids]
        self.fold_stake = (
            tree.contrib[self.fold_ids, self.fold_folder].to(self.dtype)
            if len(self.fold_ids)
            else torch.zeros(0, device=dev, dtype=self.dtype)
        )
        self.fold_bid = tree.board_id[self.fold_ids]
        # showdowns on a full board
        sd = kind == SHOWDOWN
        full = sd & (blen[tree.board_id] == 5)
        self.sd_ids = full.nonzero().flatten()
        sd_boards = sorted({int(b) for b in tree.board_id[self.sd_ids].tolist()})
        self.sd_tables = ShowdownTables([tree.boards[b] for b in sd_boards], dev)
        remap = {b: i for i, b in enumerate(sd_boards)}
        self.sd_tid = torch.tensor(
            [remap[int(b)] for b in tree.board_id[self.sd_ids].tolist()],
            dtype=torch.long,
            device=dev,
        )
        self.sd_stake = tree.contrib[self.sd_ids].min(1).values.to(self.dtype)
        # all-in run-outs: a dense matrix per board, or one kernel row per run-out
        part = sd & ~full
        mode = cfg.allin_mode
        self.allin: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        part_ids = part.nonzero().flatten()
        rows_node: list[int] = []
        rows_board: list[tuple] = []
        rows_coef: list[float] = []
        for b in sorted({int(x) for x in tree.board_id[part_ids].tolist()}):
            ids = part_ids[tree.board_id[part_ids] == b]
            stake = tree.contrib[ids].min(1).values.to(self.dtype)
            dense = mode == "dense" or (
                mode == "auto" and (dev.type == "cuda" or len(ids) >= cfg.dense_min_nodes)
            )
            if dense:
                E = allin_matrix(tree.boards[b], cfg.max_runouts, dev, self.dtype, cfg.seed)
                self.allin.append((ids, stake, E))
                continue
            runouts, scale = runouts_for(tree.boards[b], cfg.max_runouts, cfg.seed)
            for n, m in zip(ids.tolist(), stake.tolist(), strict=True):
                for r in runouts:
                    rows_node.append(n)
                    rows_board.append(tuple(tree.boards[b]) + tuple(r))
                    rows_coef.append(m * scale)
        uniq = sorted(set(rows_board))
        self.ro_tables = ShowdownTables(uniq, dev)
        pos = {bd: i for i, bd in enumerate(uniq)}
        self.ro_node = torch.tensor(rows_node, dtype=torch.long, device=dev)
        self.ro_tid = torch.tensor([pos[bd] for bd in rows_board], dtype=torch.long, device=dev)
        self.ro_coef = torch.tensor(rows_coef, dtype=self.dtype, device=dev)
        # continuations
        self.rollouts = rollouts
        conts = (kind == CONTINUATION).nonzero().flatten()
        if len(conts) and rollouts is None:
            raise ValueError("tree has depth-limit leaves but no rollouts were given")
        self.cont_ids = conts

    def evaluate(self, player: int, opp: torch.Tensor, v: torch.Tensor) -> None:
        """Write values of ``player`` into ``v[node]`` for all terminal-like
        nodes; ``opp`` is the opponent's reach ``[N, C]``."""
        if len(self.fold_ids):
            r = opp[self.fold_ids]
            coef = torch.where(self.fold_folder == player, -self.fold_stake, self.fold_stake)
            v[self.fold_ids] = fold_values(r, self.board_valid[self.fold_bid]) * coef[:, None]
        n = len(self.sd_ids)
        for s in range(0, n, self.chunk):
            ids = self.sd_ids[s : s + self.chunk]
            out = self.sd_tables.showdown(opp[ids], self.sd_tid[s : s + self.chunk])
            v[ids] = out * self.sd_stake[s : s + self.chunk, None]
        for ids, stake, E in self.allin:
            v[ids] = (opp[ids] @ E.t()) * stake[:, None]
        n = len(self.ro_node)
        for s in range(0, n, self.chunk):
            ids = self.ro_node[s : s + self.chunk]
            out = self.ro_tables.showdown(opp[ids], self.ro_tid[s : s + self.chunk])
            v.index_add_(0, ids, out * self.ro_coef[s : s + self.chunk, None])
        if len(self.cont_ids):
            v[self.cont_ids] = self.rollouts.values(player, opp[self.cont_ids], self.chunk).to(
                v.dtype
            )


class RangeSolver:
    def __init__(
        self,
        tree: SubgameTree,
        ranges: torch.Tensor,
        cfg: SolverConfig | None = None,
        rollouts: RolloutSet | None = None,
        gadget: Any = None,
        locked: dict[int, torch.Tensor] | None = None,
        terminals: TerminalEvaluator | None = None,
    ) -> None:
        self.tree = tree
        self.cfg = cfg or SolverConfig()
        dev = tree.device
        self.device = dev
        self.dtype = getattr(torch, self.cfg.dtype)
        N = tree.num_nodes
        self.N = N
        root_valid = self.board_valid_root = valid_masks([tree.boards[0]], dev)[0]
        self.ranges = ranges.to(dev, self.dtype) * root_valid
        # decision nodes, player 0 first
        is_dec = (tree.kind == DECISION) | (tree.kind == LEAF)
        d0 = (is_dec & (tree.actor == 0)).nonzero().flatten()
        d1 = (is_dec & (tree.actor == 1)).nonzero().flatten()
        self.dec_nodes = torch.cat([d0, d1])
        self.D0 = len(d0)
        self.Dn = len(self.dec_nodes)
        self.dec_index = torch.full((N,), -1, dtype=torch.long, device=dev)
        self.dec_index[self.dec_nodes] = torch.arange(self.Dn, device=dev)
        nch = tree.num_children[self.dec_nodes]
        self.A = max(1, int(nch.max())) if self.Dn else 1
        self.legal = torch.arange(self.A, device=dev)[None, :] < nch[:, None]  # [D, A]
        lf = self.legal.to(self.dtype)
        self.uniform = (lf / lf.sum(1, keepdim=True).clamp(min=1))[:, :, None].expand(-1, -1, C)
        self.regret = torch.zeros(self.Dn, self.A, C, device=dev, dtype=self.dtype)
        self.strat_sum = torch.zeros_like(self.regret)
        self.sigma = self.uniform.clone()
        self.locked = torch.zeros(self.Dn, dtype=torch.bool, device=dev)
        for node, strat in (locked or {}).items():
            d = int(self.dec_index[node])
            n = int(tree.num_children[node])
            s = torch.as_tensor(strat, dtype=self.dtype, device=dev)
            s = s[None].expand(C, n) if s.dim() == 1 else s
            s = s / s.sum(1, keepdim=True).clamp(min=1e-30)
            self.sigma[d] = 0
            self.sigma[d, :n] = s.t()
            self.locked[d] = True
        self.avoid = avoids_card(dev).to(self.dtype)
        self._levels()
        self.terminals = terminals or TerminalEvaluator(tree, self.cfg, rollouts)
        self.gadget = gadget
        if gadget is not None:
            self.g_prior = gadget.prior.to(dev, self.dtype) * root_valid
            self.g_term = gadget.terminate.to(dev, self.dtype)
            self.g_regret = torch.zeros(2, C, device=dev, dtype=self.dtype)
            self.g_enter = torch.full((C,), 0.5, device=dev, dtype=self.dtype)
        self.t = 0
        self.iterations_done = 0
        self.solve_time = 0.0

    # -- setup --------------------------------------------------------------

    def _levels(self) -> None:
        tree = self.tree
        self.levels = []
        ls = tree.level_start
        for d in range(1, len(ls) - 1):
            ids = torch.arange(ls[d], ls[d + 1], device=self.device)
            par = tree.parent[ids]
            pk = tree.kind[par]
            dm = (pk == DECISION) | (pk == LEAF)
            cm = pk == CHANCE
            L: dict[str, Any] = {}
            di = ids[dm]
            L["d_ids"] = di
            L["d_par"] = par[dm]
            L["d_pd"] = self.dec_index[par[dm]]
            L["d_slot"] = tree.slot[di]
            L["d_actor"] = tree.actor[par[dm]]
            ci = ids[cm]
            L["c_ids"] = ci
            L["c_par"] = par[cm]
            L["c_card"] = tree.deal_card[ci]
            L["c_w"] = tree.chance_weight[ci].to(self.dtype)
            for i in (0, 1):
                own = L["d_actor"] == i
                L[f"own{i}"] = own
                upd = own & ~self.locked[L["d_pd"]]
                L[f"upd{i}"] = upd.nonzero().flatten()
            self.levels.append(L)

    def _slice(self, i: int) -> slice:
        return slice(0, self.D0) if i == 0 else slice(self.D0, self.Dn)

    # -- passes -------------------------------------------------------------

    def root_reach(self, sigma_enter: torch.Tensor | None = None) -> torch.Tensor:
        r = self.ranges.clone()
        if self.gadget is not None:
            g = self.g_enter if sigma_enter is None else sigma_enter
            r[self.gadget.player] = self.g_prior * g
        return r

    def forward(self, sigma: torch.Tensor, root: torch.Tensor) -> torch.Tensor:
        reach = torch.empty(2, self.N, C, device=self.device, dtype=self.dtype)
        reach[:, 0] = root
        for L in self.levels:
            if len(L["d_ids"]):
                f = sigma[L["d_pd"], L["d_slot"]]
                r = reach[:, L["d_par"]]
                for p in (0, 1):
                    own = L[f"own{p}"][:, None]
                    reach[p, L["d_ids"]] = r[p] * torch.where(own, f, torch.ones_like(f))
            if len(L["c_ids"]):
                m = self.avoid[L["c_card"]]
                reach[:, L["c_ids"]] = reach[:, L["c_par"]] * m
        return reach

    def backward(
        self,
        i: int,
        sigma: torch.Tensor,
        opp: torch.Tensor,
        best_response: bool = False,
        regrets: bool = False,
    ) -> torch.Tensor:
        v = torch.zeros(self.N, C, device=self.device, dtype=self.dtype)
        self.terminals.evaluate(i, opp, v)
        for L in reversed(self.levels):
            ids = L["d_ids"]
            if len(ids):
                cv = v[ids]
                own = L[f"own{i}"]
                if best_response:
                    oi = own.nonzero().flatten()
                    ni = (~own).nonzero().flatten()
                    if len(oi):
                        idx = L["d_par"][oi][:, None].expand(-1, C)
                        v.scatter_reduce_(0, idx, cv[oi], "amax", include_self=False)
                    if len(ni):
                        v.index_add_(0, L["d_par"][ni], cv[ni])
                else:
                    f = sigma[L["d_pd"], L["d_slot"]]
                    w = torch.where(own[:, None], f * cv, cv)
                    v.index_add_(0, L["d_par"], w)
            if len(L["c_ids"]):
                m = self.avoid[L["c_card"]] * L["c_w"][:, None]
                v.index_add_(0, L["c_par"], v[L["c_ids"]] * m)
            if regrets:
                u = L[f"upd{i}"]
                if len(u):
                    cid = ids[u]
                    inst = v[cid] - v[L["d_par"][u]]
                    self.regret.index_put_((L["d_pd"][u], L["d_slot"][u]), inst, accumulate=True)
        return v

    # -- CFR ----------------------------------------------------------------

    def _discount(self, i: int) -> None:
        t = self.t
        if t < 2:
            return
        sl = self._slice(i)
        cfg = self.cfg
        if cfg.algorithm == "dcfr":
            tp = t - 1
            a = tp**cfg.alpha / (tp**cfg.alpha + 1)
            b = tp**cfg.beta / (tp**cfg.beta + 1)
            R = self.regret[sl]
            R.mul_(torch.where(R > 0, a, b))
            self.strat_sum[sl].mul_((tp / t) ** cfg.gamma)
            if self.gadget is not None and i == self.gadget.player:
                self.g_regret.mul_(torch.where(self.g_regret > 0, a, b))

    def _regret_match(self, i: int) -> None:
        sl = self._slice(i)
        pos = self.regret[sl].clamp(min=0) * self.legal[sl][:, :, None]
        s = pos.sum(1, keepdim=True)
        new = torch.where(s > 0, pos / s.clamp(min=1e-30), self.uniform[sl])
        keep = self.locked[sl][:, None, None]
        self.sigma[sl] = torch.where(keep, self.sigma[sl], new)

    def _update(self, i: int) -> None:
        self._discount(i)
        root = self.root_reach()
        reach = self.forward(self.sigma, root)
        v = self.backward(i, self.sigma, reach[1 - i], regrets=True)
        sl = self._slice(i)
        nodes = self.dec_nodes[sl]
        w = float(self.t) if self.cfg.algorithm == "cfr+" else 1.0
        self.strat_sum[sl] += w * reach[i, nodes][:, None, :] * self.sigma[sl]
        if self.cfg.algorithm == "cfr+":
            self.regret[sl].clamp_(min=0)
        self._regret_match(i)
        if self.gadget is not None and i == self.gadget.player:
            self._gadget_update(v[0])

    def _gadget_update(self, v_enter: torch.Tensor) -> None:
        e = self.g_enter
        vt = self.g_term
        vg = (1 - e) * vt + e * v_enter
        self.g_regret[0] += vt - vg
        self.g_regret[1] += v_enter - vg
        if self.cfg.algorithm == "cfr+":
            self.g_regret.clamp_(min=0)
        pos = self.g_regret.clamp(min=0)
        s = pos.sum(0)
        self.g_enter = torch.where(s > 0, pos[1] / s.clamp(min=1e-30), torch.full_like(s, 0.5))

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @torch.no_grad()
    def solve(self, iterations: int | None = None, time_budget: float | None = None) -> dict:
        iterations = self.cfg.iterations if iterations is None else iterations
        budget = self.cfg.time_budget if time_budget is None else time_budget
        start = time.perf_counter()
        n = 0
        while n < iterations:
            self.t += 1
            self._update(0)
            self._update(1)
            n += 1
            if budget is not None:
                self._sync()
                if time.perf_counter() - start >= budget:
                    break
        self._sync()
        dt = time.perf_counter() - start
        self.iterations_done += n
        self.solve_time += dt
        return {"iterations": n, "seconds": dt}

    # -- results ------------------------------------------------------------

    def average_strategy(self) -> torch.Tensor:
        s = self.strat_sum * self.legal[:, :, None]
        tot = s.sum(1, keepdim=True)
        avg = torch.where(tot > 0, s / tot.clamp(min=1e-30), self.sigma)
        keep = self.locked[:, None, None]
        return torch.where(keep, self.sigma, avg)

    def node_strategy(self, node: int, average: bool = True) -> torch.Tensor:
        """``[1326, n_children]`` strategy of the actor at ``node``."""
        d = int(self.dec_index[node])
        if d < 0:
            raise ValueError(f"node {node} is not a decision node")
        n = int(self.tree.num_children[node])
        s = self.average_strategy()[d] if average else self.sigma[d]
        return s[:n].t().contiguous()

    def root_strategy(self) -> torch.Tensor:
        return self.node_strategy(self.tree.current_node)

    @torch.no_grad()
    def values(
        self,
        player: int,
        sigma: torch.Tensor | None = None,
        best_response: bool = False,
        root: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Counterfactual values ``[N, C]`` of ``player`` (optionally a best
        response to the other player's strategy in ``sigma``) and the reach."""
        sigma = self.average_strategy() if sigma is None else sigma
        root = self.root_reach() if root is None else root
        reach = self.forward(sigma, root)
        return self.backward(player, sigma, reach[1 - player], best_response=best_response), reach

    def pair_mass(self, root: torch.Tensor | None = None) -> torch.Tensor:
        r = self.ranges if root is None else root
        return (r[0] * blocked_sum(r[1])).sum()

    @torch.no_grad()
    def exploitability(self) -> dict:
        """Exact best responses of both players against the average strategy
        on the plain ranges (no gadget), in chips per hand."""
        sigma = self.average_strategy()
        root = self.ranges
        Z = self.pair_mass(root)
        br, ev = [], []
        for p in (0, 1):
            vb, _ = self.values(p, sigma, True, root)
            va, _ = self.values(p, sigma, False, root)
            br.append(float((root[p] * vb[0]).sum() / Z))
            ev.append(float((root[p] * va[0]).sum() / Z))
        nashconv = br[0] + br[1]
        return {
            "br": br,
            "ev": ev,
            "nashconv": nashconv,
            "exploitability": nashconv / 2,
            "pot": int(self.tree.contrib[0].sum()),
        }
