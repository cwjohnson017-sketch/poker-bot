"""Preflop best response against the blueprint, with flop-root solves as leaves.

Production plays preflop from the blueprint and searches every later street.
This module measures how much a player could gain by changing only preflop
play, holding postflop play at the search's own solution for the blueprint's
flop-root ranges:

1. **The preflop tree.** :func:`preflop_tree` walks the blueprint's preflop
   abstraction (button = seat 0) and records every decision with the
   blueprint's per-combo policy, and every terminal: a fold, a preflop all-in,
   or a flop start. Each node carries both seats' reach under the blueprint.
2. **Leaf values.** At every flop start and every sampled flop,
   :func:`solve_flop_root` solves the flop-root subgame as production's flop
   search would (same tree, same leaf nets, a fixed iteration count) with the
   blueprint's ranges, and keeps both seats' counterfactual values at the root.
   A combo's value is the hand's net chips, summed over the opponent's reach.
3. **Suit symmetry.** Each flop's values also stand for the 23 suit
   permutations of the flop (:class:`FlopValues`), so a sample of K flops
   covers about 24 K flop-hand pairs.
4. **Best response.** :func:`evaluate` walks the preflop tree for each seat:
   the blueprint's preflop against a best response that picks, per combo, the
   best action at each of its own decisions.

Values per hand are estimates from sampled flops. A maximum over noisy values
overestimates the best response, so :func:`evaluate` also cross-fits: the
best response is chosen on one half of the flops and scored on the other,
which underestimates it. The truth (for this postflop model) lies between.

What it does not measure: when a player deviates preflop, the ranges reaching
the flop change and so would the opponent's postflop solution. The leaf
values are fixed at the blueprint's ranges. This is a local best response:
the gain from preflop changes alone, as a preflop re-solve with flop-root
values could capture.
"""

from __future__ import annotations

import itertools
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from .abstract import legal_options, make_state, to_action
from .blueprint import policy_matrix
from .combos import NUM_COMBOS, blocked_sum, combo_index, combo_table, valid_mask
from .solver import RangeSolver, allin_matrix
from .tree import VALUE, TreeBuilder

C = NUM_COMBOS
# pairs of disjoint hole-card combos: the chance normaliser of a deal
PAIRS = C * 1225
# a flop sample's mean value per hand -> the sum over flops / C(48, 3)
FLOP_SCALE = math.comb(50, 3) / math.comb(48, 3)


@dataclass
class Node:
    kind: str  # "dec", "fold", "allin", "flop"
    history: list  # (street, player, engine Action)
    reach: torch.Tensor  # [2, C] blueprint reach of both seats
    contrib: tuple[int, int]
    player: int = -1  # actor ("dec"), folder ("fold")
    labels: list[str] = field(default_factory=list)
    policy: torch.Tensor | None = None  # [C, A] blueprint policy over the children
    children: list[int] = field(default_factory=list)
    flop_id: int = -1  # index among the flop starts
    indices: list[int] = field(default_factory=list)  # spec index of each child's action

    def key(self) -> tuple:
        """The preflop history as ``((player, kind, amount), ...)``."""
        return history_key(self.history)

    def line(self) -> str:
        return " ".join(f"{p}:{_act(a)}" for _s, p, a in self.history) or "root"


def history_key(history: Sequence) -> tuple:
    """``((player, kind, amount), ...)`` of the preflop actions in ``history``."""
    return tuple((int(p), int(a.kind), int(a.amount)) for s, p, a in history if int(s) == 0)


def _act(a: Any) -> str:
    k = int(a.kind)
    return "f" if k == 0 else "c" if k == 1 else f"r{int(a.amount)}"


def preflop_tree(
    bp: Any, config: Any, engine: Any, device: Any = "cpu", symmetrize: bool = True
) -> list[Node]:
    """Every node of the blueprint's preflop tree with button 0, root first.

    ``symmetrize`` averages each policy over the 24 suit permutations. The
    neural blueprint is close to suit-symmetric but not exactly (about 1% of
    probability per action, reach-weighted), and the flop values rely on
    symmetry (:class:`FlopValues`)."""
    nodes: list[Node] = []
    perms = suit_perms(device) if symmetrize else None
    n_flops = 0

    def rec(hist: list, reach: torch.Tensor) -> int:
        nonlocal n_flops
        st = make_state(engine, config, 0, [], hist)
        contrib = tuple(int(config.stacks[p]) - int(st.stacks[p]) for p in range(2))
        i = len(nodes)
        if st.is_terminal:
            folded = [int(a.kind) == 0 for _s, _p, a in hist]
            if any(folded):
                nodes.append(Node("fold", hist, reach, contrib, player=int(hist[-1][1])))
            else:
                nodes.append(Node("allin", hist, reach, contrib))
            return i
        if int(st.street) > 0:
            nodes.append(Node("flop", hist, reach, contrib, flop_id=n_flops))
            n_flops += 1
            return i
        p = int(st.current_player)
        opts = legal_options(st, bp.spec)
        P = policy_matrix(bp, st, p, device=device)
        pol = torch.stack([P[:, o.index] for o in opts], 1).expand(C, -1)
        pol = pol / pol.sum(1, keepdim=True).clamp(min=1e-30)
        if perms is not None:
            pol = torch.stack([pol[perms[k]] for k in range(perms.shape[0])]).mean(0)
        node = Node("dec", hist, reach, contrib, player=p, policy=pol)
        node.labels = [o.label if o.kind != 2 else f"r{o.amount}" for o in opts]
        node.indices = [o.index for o in opts]
        nodes.append(node)
        for j, o in enumerate(opts):
            r = reach.clone()
            r[p] = r[p] * pol[:, j]
            node.children.append(rec(hist + [(0, p, to_action(engine, o.kind, o.amount))], r))
        return i

    rec([], torch.ones(2, C, device=device))
    return nodes


def sample_flops(k: int, seed: int) -> list[tuple[int, int, int]]:
    """``k`` distinct flops drawn uniformly at random (sorted cards)."""
    g = torch.Generator().manual_seed(int(seed))
    seen: list[tuple[int, int, int]] = []
    have: set = set()
    while len(seen) < k:
        f = tuple(sorted(torch.randperm(52, generator=g)[:3].tolist()))
        if f not in have:
            have.add(f)
            seen.append(f)  # type: ignore[arg-type]
    return seen


def solve_flop_root(
    agent: Any,
    config: Any,
    engine: Any,
    history: list,
    flop: Sequence[int],
    reach: torch.Tensor,
    iterations: int,
    exploitability: bool = False,
) -> dict:
    """Solve the flop-root subgame after the preflop ``history`` on ``flop`` with
    ``agent``'s tree and leaf settings; both seats' root counterfactual values."""
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
    out = {
        "values": v,
        "nodes": int(tree.num_nodes),
        "iterations": int(solver.iterations_done),
        "seconds": time.perf_counter() - t0,
    }
    if exploitability:
        ex = solver.exploitability()
        out["exploitability"] = ex["exploitability"]
        out["pot"] = ex["pot"]
    del solver, tree, leaves
    return out


def suit_perms(device: Any = "cpu") -> torch.Tensor:
    """``[24, C]``: combo ``c`` maps to ``perm[k, c]`` under the k-th suit permutation."""
    cards = combo_table("cpu")
    out = []
    for perm in itertools.permutations(range(4)):
        m = [(c // 4) * 4 + perm[c % 4] for c in range(52)]
        out.append([combo_index(m[int(a)], m[int(b)]) for a, b in cards.tolist()])
    return torch.tensor(out, dtype=torch.long, device=device)


class FlopValues:
    """Per-hand flop values of one flop start: sums and counts over sampled flops
    and their suit permutations, in two halves for cross-fitting."""

    def __init__(self, perms: torch.Tensor) -> None:
        self.perms = perms
        self.sum = torch.zeros(2, 2, C, dtype=torch.float64)  # [half, seat, C]
        self.cnt = torch.zeros(2, C, dtype=torch.float64)  # [half, C]

    def add(self, half: int, flop: Sequence[int], values: torch.Tensor) -> None:
        ok = valid_mask(list(flop), "cpu").double()
        v = values.double() * ok
        for k in range(self.perms.shape[0]):
            idx = self.perms[k]
            self.sum[half].index_add_(1, idx, v)
            self.cnt[half].index_add_(0, idx, ok)

    def mean(self, halves: Sequence[int]) -> torch.Tensor:
        """``[2, C]``: each seat's value per hand, summed over flops / C(48, 3)."""
        s = self.sum[list(halves)].sum(0)
        n = self.cnt[list(halves)].sum(0)
        return FLOP_SCALE * s / n.clamp(min=1)


def terminal_values(
    nodes: list[Node],
    flop_values: dict[int, torch.Tensor],
    allin_E: torch.Tensor | None,
) -> dict[int, torch.Tensor]:
    """``{node: [2, C]}`` counterfactual values of both seats at every terminal:
    net chips summed over the opponent's blueprint reach."""
    out: dict[int, torch.Tensor] = {}
    for i, n in enumerate(nodes):
        if n.kind == "dec":
            continue
        v = torch.zeros(2, C, dtype=torch.float64)
        r = n.reach.double().cpu()
        if n.kind == "fold":
            stake = n.contrib[n.player]
            for p in (0, 1):
                sign = -1.0 if p == n.player else 1.0
                v[p] = sign * stake * blocked_sum(r[1 - p])
        elif n.kind == "allin":
            stake = min(n.contrib)
            for p in (0, 1):
                v[p] = stake * (allin_E @ r[1 - p])
        else:
            v = flop_values[n.flop_id].double()
        out[i] = v
    return out


def _walk(
    nodes: list[Node],
    i: int,
    p: int,
    tv_choose: dict[int, torch.Tensor],
    tv_eval: dict[int, torch.Tensor],
    mode: str,
    gains: dict[int, dict] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Seat ``p``'s values per combo below node ``i``: ``(choose, eval)``.

    ``mode`` "bp": ``p`` plays the blueprint. "br": at each of ``p``'s decisions,
    the action with the highest ``choose`` value, scored with ``eval``."""
    n = nodes[i]
    if n.kind != "dec":
        return tv_choose[i][p], tv_eval[i][p]
    kids = [_walk(nodes, j, p, tv_choose, tv_eval, mode, gains) for j in n.children]
    Qc = torch.stack([k[0] for k in kids], 1)  # [C, A]
    Qe = torch.stack([k[1] for k in kids], 1)
    if n.player != p:
        return Qc.sum(1), Qe.sum(1)
    pol = n.policy.double().cpu()
    if gains is not None:
        own = n.reach[p].double().cpu()
        bp_c = (pol * Qc).sum(1)
        best = Qc.argmax(1)
        gains[i] = {
            "gain": own * (Qc.max(1).values - bp_c),
            "best": best,
            "own": own,
        }
    if mode == "bp":
        return (pol * Qc).sum(1), (pol * Qe).sum(1)
    a = Qc.argmax(1, keepdim=True)
    return Qc.gather(1, a)[:, 0], Qe.gather(1, a)[:, 0]


def mbb(chips_sum: torch.Tensor | float, big_blind: int) -> float:
    """A sum over deals of chips -> mbb/hand."""
    return float(chips_sum) / PAIRS / big_blind * 1000.0


def evaluate(
    nodes: list[Node],
    fv: dict[int, FlopValues],
    allin_E: torch.Tensor | None,
    big_blind: int,
) -> dict:
    """Blueprint values, best responses (in-sample and cross-fitted) and the
    one-step gains per decision node, in mbb/hand per seat (button = seat 0)."""
    full = terminal_values(nodes, {k: f.mean((0, 1)) for k, f in fv.items()}, allin_E)
    half = [
        terminal_values(nodes, {k: f.mean((h,)) for k, f in fv.items()}, allin_E) for h in (0, 1)
    ]
    res: dict[str, Any] = {"seats": []}
    for p in (0, 1):
        gains: dict[int, dict] = {}
        bp = _walk(nodes, 0, p, full, full, "bp", gains)[0].sum()
        br = _walk(nodes, 0, p, full, full, "br", None)[0].sum()
        x01 = _walk(nodes, 0, p, half[0], half[1], "br", None)[1].sum()
        x10 = _walk(nodes, 0, p, half[1], half[0], "br", None)[1].sum()
        # cross-fitted blueprint values, on the same halves as the scores above
        bp01 = _walk(nodes, 0, p, half[1], half[1], "bp", None)[0].sum()
        bp10 = _walk(nodes, 0, p, half[0], half[0], "bp", None)[0].sum()
        seat = {
            "blueprint": mbb(bp, big_blind),
            "br_in_sample": mbb(br, big_blind),
            "br_cross_fit": mbb((x01 + x10) / 2, big_blind),
            "blueprint_halves": mbb((bp01 + bp10) / 2, big_blind),
            "nodes": [],
        }
        for i, g in sorted(gains.items(), key=lambda kv: -float(kv[1]["gain"].sum())):
            n = nodes[i]
            own = g["own"]
            m = float(own.sum())
            bp_freq = (own[:, None] * n.policy.double().cpu()).sum(0) / max(m, 1e-30)
            br_freq = torch.zeros(len(n.children), dtype=torch.float64)
            br_freq.index_add_(0, g["best"], own)
            br_freq /= max(m, 1e-30)
            seat["nodes"].append(
                {
                    "node": i,
                    "line": n.line(),
                    "gain": mbb(g["gain"].sum(), big_blind),
                    "labels": n.labels,
                    "blueprint_freq": [round(float(x), 4) for x in bp_freq],
                    "best_freq": [round(float(x), 4) for x in br_freq],
                }
            )
        res["seats"].append(seat)
    s = res["seats"]
    res["zero_sum_check"] = s[0]["blueprint"] + s[1]["blueprint"]
    res["exploitability_in_sample"] = (s[0]["br_in_sample"] + s[1]["br_in_sample"]) / 2
    res["exploitability_cross_fit"] = (s[0]["br_cross_fit"] + s[1]["br_cross_fit"]) / 2
    return res


def preflop_allin_matrix(max_runouts: int, seed: int = 0) -> torch.Tensor:
    """``E[c, c']`` over five-card boards (sampled), for preflop all-ins."""
    return allin_matrix((), max_runouts, torch.device("cpu"), torch.float64, seed)
