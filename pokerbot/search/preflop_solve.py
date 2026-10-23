"""Solving the preflop against the postflop the search plays.

:mod:`.preflop_eval` measures a best response to a preflop strategy with flop-root
solves as leaves. This module changes the strategy: it alternates

1. **flop-root solves** for the current preflop ranges (as in
   :func:`.preflop_eval.solve_flop_root`), each followed by a **linearisation**
   (:func:`class_values`): with the postflop strategy and the turn-end leaves'
   per-hand values held fixed (:class:`FrozenLeaves`), a hand's flop value is
   linear in the opponent's reach, and the reach of a suit-symmetric preflop
   strategy is constant over each of the 169 hand classes. So ``W[p, k, h]``, the
   value of seat ``p``'s hand ``h`` against one unit of reach on every combo of
   class ``k``, gives the value against any symmetric preflop range;
2. **preflop CFR** (:func:`solve_preflop`) on the preflop tree with those values at
   the flop starts and exact values at folds and preflop all-ins.

Each round's flop solves see the ranges of the previous round's preflop solution,
so the frozen postflop plays catch up one round at a time.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from .combos import NUM_COMBOS, blocked_sum, combo_table, valid_mask
from .preflop_eval import PAIRS, Node, history_key, mbb
from .solver import RangeSolver
from .tree import VALUE, TreeBuilder

C = NUM_COMBOS
K = 169
# all flops / the flops disjoint from both hands
FLOP_SCALE_ALL = math.comb(52, 3) / math.comb(48, 3)


def hand_classes(device: Any = "cpu") -> torch.Tensor:
    """``[C]`` preflop class of every combo in ``0..168``: pairs by rank (0-12),
    then suited and offsuit by rank pair."""
    cards = combo_table("cpu")
    r = cards // 4
    s = cards % 4
    hi = torch.maximum(r[:, 0], r[:, 1])
    lo = torch.minimum(r[:, 0], r[:, 1])
    pair = hi == lo
    suited = s[:, 0] == s[:, 1]
    # 78 rank pairs hi > lo, indexed hi * (hi - 1) / 2 + lo
    tri = hi * (hi - 1) // 2 + lo
    cls = torch.where(pair, hi, torch.where(suited, 13 + tri, 13 + 78 + tri))
    return cls.to(device)


def class_matrix(device: Any = "cpu", dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """``[K, C]`` indicator of each class's combos."""
    m = torch.zeros(K, C, dtype=dtype, device=device)
    m[hand_classes(device), torch.arange(C, device=device)] = 1
    return m


class FrozenLeaves:
    """Value-net leaves frozen as a zero-sum payoff that is linear in both reaches.

    At each leaf, ``ev_p(h)`` is seat ``p``'s value per unit of opponent reach at
    the reach the leaves were frozen at. Holding ``ev_p`` fixed for each seat
    separately is not zero-sum once a range moves: each side would keep its old
    values. Instead every disjoint pair of hands gets the payoff

        ``U_0(h, o) = ev_0(h) - ev_1(o) + kappa = -U_1(o, h)``,

    with ``kappa`` the mean of ``ev_1`` over seat 1's reach minus the mean of
    ``ev_0`` over seat 0's (halved). Then
    ``v_p(h) = (ev_p(h) + kappa_p) * B(opp)(h) - B(opp * ev_-p)(h)``
    (``B`` = ``blocked_sum``, ``kappa_0 = kappa``, ``kappa_1 = -kappa``). It
    reproduces the net's values at the frozen reach up to card removal, and a
    hand's value falls as the opponent's range gets stronger. ``ev`` is
    ``[2, L, C]`` and ``kappa`` ``[L]``, in the provider's ``ids`` order."""

    graph_safe = False

    def __init__(self, ids: torch.Tensor, ev: torch.Tensor, kappa: torch.Tensor) -> None:
        self.ids = ids
        self.ev = ev
        self.kappa = kappa

    def values(self, player: int, reach: torch.Tensor, cached: bool = False) -> torch.Tensor:
        opp = reach[1 - player]
        k = (self.kappa if player == 0 else -self.kappa)[:, None]
        return (self.ev[player] + k) * blocked_sum(opp) - blocked_sum(opp * self.ev[1 - player])


@torch.no_grad()
def freeze_leaves(solver: RangeSolver) -> FrozenLeaves:
    """The solver's value-net leaves frozen at its average strategy's reach."""
    term = solver.terminals
    ids = term.value_ids
    sigma = solver.average_strategy()
    reach = solver.forward(sigma, solver.root_reach())
    r = reach[:, ids]
    evs = []
    for p in (0, 1):
        v = term.value_leaves.values(p, r, False).to(solver.dtype)
        m = blocked_sum(r[1 - p])
        evs.append(torch.where(m > 1e-12, v / m.clamp(min=1e-30), torch.zeros_like(v)))
    ev = torch.stack(evs)

    def mean(p: int) -> torch.Tensor:
        w = r[p]
        return (w * ev[p]).sum(-1) / w.sum(-1).clamp(min=1e-30)

    kappa = 0.5 * (mean(1) - mean(0))
    return FrozenLeaves(ids, ev, kappa)


@torch.no_grad()
def class_values(solver: RangeSolver, chunk: int = 0) -> torch.Tensor:
    """``[2, K, C]``: root values of each seat's hands against one unit of reach
    on every combo of each opponent class, with the postflop strategy (the
    average) and the value-net leaves frozen. Replaces the solver's leaf provider."""
    term = solver.terminals
    if len(term.value_ids):
        term.value_leaves = freeze_leaves(solver)
    sigma = solver.average_strategy()
    root = int(solver.tree.current_node)
    ok = solver.board_valid_root.to(solver.dtype)
    M = class_matrix(solver.device, solver.dtype) * ok
    W = torch.zeros(2, K, C, dtype=solver.dtype, device=solver.device)
    base = solver.ranges.clone()
    for p in (0, 1):
        for k in range(K):
            if not bool(M[k].any()):
                continue
            r = base.clone()
            r[1 - p] = M[k]
            v, _ = solver.values(p, sigma, False, r)
            W[p, k] = v[root]
    return W * ok


def solve_flop_linear(
    agent: Any,
    config: Any,
    engine: Any,
    history: list,
    flop: Sequence[int],
    reach: torch.Tensor,
    iterations: int,
) -> dict:
    """Solve the flop root (as :func:`.preflop_eval.solve_flop_root`) and linearise:
    both seats' root values at ``reach`` and the class values ``W``."""
    dev = agent.device
    t0 = time.perf_counter()
    tree = TreeBuilder(
        config, 0, list(flop), history, [], agent.cfg.tree, None, engine, dev
    ).build()
    leaves = agent.value_leaf_provider(tree) if bool((tree.kind == VALUE).any()) else None
    ranges = reach.to(dev) * valid_mask(list(flop), dev)
    solver = RangeSolver(tree, ranges, agent.cfg.solver, None, None, {}, value_leaves=leaves)
    solver.solve(iterations=iterations)
    root = int(tree.current_node)
    v = torch.stack([solver.values(p)[0][root] for p in (0, 1)]).float().cpu()
    t1 = time.perf_counter()
    W = class_values(solver).float().cpu()
    out = {
        "values": v,
        "W": W,
        "nodes": int(tree.num_nodes),
        "seconds": t1 - t0,
        "linear_seconds": time.perf_counter() - t1,
    }
    del solver, tree, leaves
    return out


class ClassValues:
    """Sums of ``W`` over sampled flops and their suit permutations, per flop start.

    :meth:`mean` divides by the number of samples, not per hand by the samples
    compatible with it (as :meth:`FlopValues.mean` does). Every sample is a
    zero-sum game, so the estimate is exactly zero-sum, which CFR needs: per-hand
    normalisation leaves a non-zero-sum residue that the solver exploits."""

    def __init__(self, perms: torch.Tensor) -> None:
        self.perms = perms
        self.sum = torch.zeros(2, K, C, dtype=torch.float64)
        self.cnt = torch.zeros(C, dtype=torch.float64)
        self.n = 0

    def add(self, flop: Sequence[int], W: torch.Tensor) -> None:
        ok = valid_mask(list(flop), "cpu").double()
        w = W.double() * ok
        for k in range(self.perms.shape[0]):
            idx = self.perms[k]
            self.sum.index_add_(2, idx, w)
            self.cnt.index_add_(0, idx, ok)
            self.n += 1

    def mean(self) -> torch.Tensor:
        """``[2, K, C]``: summed over all flops / C(48, 3), estimated as
        ``C(52, 3) / C(48, 3)`` times the mean over samples (zero where a sample's
        flop holds the hand)."""
        return FLOP_SCALE_ALL * self.sum / max(self.n, 1)

    def state_dict(self) -> dict:
        return {"sum": self.sum, "cnt": self.cnt, "n": self.n}

    def load(self, d: dict) -> None:
        self.sum = d["sum"]
        self.cnt = d["cnt"]
        self.n = int(d["n"])


# -- preflop CFR -----------------------------------------------------------------


class PreflopGame:
    """The preflop tree as a vectorised game for CFR: exact fold and all-in values,
    class-linear flop values. Strategies are ``{node: [C, A]}``."""

    def __init__(
        self,
        nodes: list[Node],
        W: dict[int, torch.Tensor],
        allin_E: torch.Tensor,
        device: Any = "cpu",
    ) -> None:
        self.nodes = nodes
        self.dev = torch.device(device)
        self.W = {k: w.to(self.dev, torch.float64) for k, w in W.items()}
        self.E = allin_E.to(self.dev, torch.float64)
        self.M = class_matrix(self.dev, torch.float64)
        self.csize = self.M.sum(1)
        self.dec = [i for i, n in enumerate(nodes) if n.kind == "dec"]

    def terminal(self, i: int, p: int, opp: torch.Tensor) -> torch.Tensor:
        """Seat ``p``'s values per combo at terminal ``i`` given the opponent's reach."""
        n = self.nodes[i]
        if n.kind == "fold":
            sign = -1.0 if p == n.player else 1.0
            return sign * n.contrib[n.player] * blocked_sum(opp)
        if n.kind == "allin":
            return min(n.contrib) * (self.E @ opp)
        rk = (self.M @ opp) / self.csize  # class reach (constant within a class)
        return rk @ self.W[n.flop_id][p]

    def values(
        self,
        p: int,
        sigma: dict[int, torch.Tensor],
        i: int = 0,
        opp: torch.Tensor | None = None,
        best_response: bool = False,
        regrets: dict[int, torch.Tensor] | None = None,
        own: torch.Tensor | None = None,
        weight: float = 1.0,
        strat_sum: dict | None = None,
    ) -> torch.Tensor:
        """Seat ``p``'s counterfactual values per combo below node ``i``. With
        ``regrets`` (and ``own``, ``p``'s reach), accumulates instantaneous regrets
        and the reach-weighted strategy (times ``weight``) into ``strat_sum``."""
        if opp is None:
            opp = torch.ones(C, dtype=torch.float64, device=self.dev)
            own = torch.ones(C, dtype=torch.float64, device=self.dev)
        n = self.nodes[i]
        if n.kind != "dec":
            return self.terminal(i, p, opp)
        s = sigma[i]
        if n.player != p:
            return sum(
                self.values(
                    p, sigma, j, opp * s[:, a], best_response, regrets, own, weight, strat_sum
                )
                for a, j in enumerate(n.children)
            )
        Q = torch.stack(
            [
                self.values(
                    p,
                    sigma,
                    j,
                    opp,
                    best_response,
                    regrets,
                    None if own is None else own * s[:, a],
                    weight,
                    strat_sum,
                )
                for a, j in enumerate(n.children)
            ],
            1,
        )
        if best_response:
            return Q.max(1).values
        v = (s * Q).sum(1)
        if regrets is not None:
            regrets[i] += Q - v[:, None]
            strat_sum[i] += weight * own[:, None] * s
        return v


def regret_match(R: torch.Tensor) -> torch.Tensor:
    pos = R.clamp(min=0)
    s = pos.sum(1, keepdim=True)
    uni = torch.full_like(R, 1.0 / R.shape[1])
    return torch.where(s > 0, pos / s.clamp(min=1e-300), uni)


def solve_preflop(
    game: PreflopGame,
    iterations: int,
    init: dict[int, torch.Tensor] | None = None,
    alpha: float = 1.5,
    beta: float = 0.0,
    gamma: float = 2.0,
    log_every: int = 0,
    log: Any = print,
) -> tuple[dict[int, torch.Tensor], list[dict]]:
    """DCFR on the preflop game (alternating updates); returns the average
    strategy and an exploitability trace (mbb/hand, from :func:`exploitability`)."""
    dev = game.dev
    R = {
        i: torch.zeros(C, len(game.nodes[i].children), dtype=torch.float64, device=dev)
        for i in game.dec
    }
    S = {i: torch.zeros_like(R[i]) for i in game.dec}
    sigma = {i: (init[i].to(dev, torch.float64) if init else regret_match(R[i])) for i in game.dec}
    trace = []
    for t in range(1, iterations + 1):
        for p in (0, 1):
            inst = {i: torch.zeros_like(R[i]) for i in game.dec}
            game.values(p, sigma, regrets=inst, weight=float(t) ** gamma, strat_sum=S)
            for i in game.dec:
                if game.nodes[i].player != p:
                    continue
                R[i] = R[i] + inst[i]
                # DCFR discounts: positive regrets by t^a / (t^a + 1), negative by t^b / (t^b + 1)
                R[i] = torch.where(
                    R[i] > 0, R[i] * (t**alpha / (t**alpha + 1)), R[i] * (t**beta / (t**beta + 1))
                )
                sigma[i] = regret_match(R[i])
        if log_every and (t % log_every == 0 or t == iterations):
            avg = average(S, sigma)
            ex = exploitability(game, avg)
            trace.append({"iteration": t, **ex})
            log(
                f"preflop CFR {t}: exploitability {ex['exploitability']:.2f} mbb/hand "
                f"(seat values {ex['values'][0]:+.1f} / {ex['values'][1]:+.1f})"
            )
    return average(S, sigma), trace


def average(
    S: dict[int, torch.Tensor], fallback: dict[int, torch.Tensor]
) -> dict[int, torch.Tensor]:
    out = {}
    for i, s in S.items():
        tot = s.sum(1, keepdim=True)
        out[i] = torch.where(tot > 0, s / tot.clamp(min=1e-300), fallback[i])
    return out


def exploitability(game: PreflopGame, sigma: dict[int, torch.Tensor], big_blind: int = 100) -> dict:
    """Values and best responses (mbb/hand) of ``sigma`` in the linearised game."""
    vals, brs = [], []
    for p in (0, 1):
        vals.append(mbb(game.values(p, sigma).sum(), big_blind))
        brs.append(mbb(game.values(p, sigma, best_response=True).sum(), big_blind))
    return {"values": vals, "br": brs, "exploitability": (brs[0] + brs[1]) / 2}


def node_policies(nodes: list[Node]) -> dict[int, torch.Tensor]:
    """The tree's own (blueprint) policies as a strategy."""
    return {i: n.policy.double().cpu() for i, n in enumerate(nodes) if n.kind == "dec"}


def with_strategy(nodes: list[Node], sigma: dict[int, torch.Tensor]) -> list[Node]:
    """A copy of the tree playing ``sigma``: policies replaced, reaches recomputed."""
    out = [
        replace(n, reach=n.reach.clone())
        if n.policy is None
        else replace(n, reach=n.reach.clone(), policy=sigma[i].to(n.policy.device, n.policy.dtype))
        for i, n in enumerate(nodes)
    ]
    out[0].reach = torch.ones_like(out[0].reach)

    def push(i: int) -> None:
        n = out[i]
        if n.kind != "dec":
            return
        for a, j in enumerate(n.children):
            r = n.reach.clone()
            r[n.player] = r[n.player] * n.policy[:, a].to(r.dtype)
            out[j].reach = r
            push(j)

    push(0)
    return out


# -- a preflop table in place of the blueprint's preflop ---------------------------


def table_from_strategy(nodes: list[Node], sigma: dict[int, torch.Tensor], width: int) -> dict:
    """``{history key: [C, width]}`` policies over the spec's action indices."""
    out = {}
    for i, n in enumerate(nodes):
        if n.kind != "dec":
            continue
        P = torch.zeros(C, width, dtype=torch.float32)
        P[:, n.indices] = sigma[i].float().cpu()
        out[n.key()] = P
    return out


def save_table(path: str | Path, base: str, table: dict, meta: dict | None = None) -> None:
    torch.save({"base": base, "table": table, "meta": meta or {}}, path)


class PreflopTable:
    """A blueprint whose preflop decisions on the tree come from a table.

    Preflop states whose history is a key of the table (the abstract preflop tree,
    button or not: keys are by seat, and the table was solved with button = seat 0,
    so seat numbers are mapped through the button) use the table; every other state
    (later streets, off-tree preflop histories) uses ``base``.
    """

    card_independent = False

    def __init__(self, base: Any, table: dict) -> None:
        self.base = base
        self.spec = base.spec
        self.table = table
        self._dev: dict = {}

    def _key(self, state: Any) -> tuple | None:
        if int(state.street) != 0:
            return None
        b = int(state.button)
        # the table's seat 0 is the button
        key = tuple(((p - b) % 2, k, a) for p, k, a in history_key(state.history))
        return key if key in self.table else None

    def on_table(self, state: Any) -> bool:
        """Whether ``state``'s decision is played from the table (not ``base``)."""
        return self._key(state) is not None

    def _rows(self, key: tuple, device: Any) -> torch.Tensor:
        d = (key, str(device))
        if d not in self._dev:
            self._dev[d] = self.table[key].to(device)
        return self._dev[d]

    def policy_combos(self, state: Any, player: int) -> torch.Tensor:
        key = self._key(state)
        if key is None:
            fn = getattr(self.base, "policy_combos", None)
            if fn is not None:
                return fn(state, player)
            from .blueprint import policy_matrix

            return policy_matrix(self.base, state, player)
        return self._rows(key, "cpu")

    def policy(self, state: Any, player: int) -> Any:
        key = self._key(state)
        if key is None:
            return self.base.policy(state, player)
        from .combos import combo_index

        a, b = state.hole_cards(player)
        return self._rows(key, "cpu")[combo_index(int(a), int(b))].numpy()


def local_base(spec: str, root: Path | None = None) -> str:
    """The table's base blueprint spec, usable on this machine. Tables name their base
    by absolute path on the machine that solved them (a Windows path); where that path
    does not exist (a cloud instance), the same ``runs/...`` path under the repository
    root is used when it exists. Anything else comes back unchanged."""
    prefix, sep, path = spec.partition(":")
    if not sep or not path or Path(path).exists():
        return spec
    p = path.replace("\\", "/")
    i = p.find("runs/")
    if i < 0:
        return spec
    if root is None:
        from ..config import REPO_ROOT

        root = REPO_ROOT
    local = Path(root) / p[i:]
    return f"{prefix}:{local.as_posix()}" if local.exists() else spec


def load_table(path: str | Path, **kwargs: Any) -> PreflopTable:
    """``preflop:<table.pt>``: the table's base blueprint with its preflop replaced."""
    from .blueprint import make_blueprint

    d = torch.load(path)
    return PreflopTable(make_blueprint(local_base(d["base"]), **kwargs), d["table"])


__all__ = [
    "ClassValues",
    "FrozenLeaves",
    "PAIRS",
    "PreflopGame",
    "PreflopTable",
    "load_table",
    "save_table",
    "table_from_strategy",
    "class_matrix",
    "class_values",
    "exploitability",
    "hand_classes",
    "node_policies",
    "solve_flop_linear",
    "solve_preflop",
    "with_strategy",
]
