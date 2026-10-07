"""Exact exploitability of a flop/turn trunk strategy with the river solved exactly.

A depth-limited flop solve (``depth_streets: 1``) decides flop and turn play;
its ``VALUE`` leaves sit at the end of turn betting. To score a trunk profile
``sigma_T`` (both players' flop and turn strategies) without trusting any
leaf model, the river is played by an exact equilibrium:

1. a forward pass of ``sigma_T`` gives both players' reaches at every leaf;
2. every (leaf, river card) river subgame is solved with
   :class:`~pokerbot.search.batch_solver.BatchRiverSolver` on both reaches,
   each normalised and mixed with ``mix`` uniform over the valid combos (so a
   defender has a defined river strategy against ranges that reach a leaf
   with zero probability under ``sigma_T``);
3. per player ``p``, best-response values at every river root against the
   other player's river strategy, with the opponent's *actual* reach, are
   chance-averaged to the leaf (weight ``1 / 44`` per river card);
4. those leaf values are backed up through the trunk with ``p`` maximising and
   the opponent playing ``sigma_T``.

``(BR_0 + BR_1) / 2`` is then the exact exploitability, in the full
flop-to-showdown game of the tree's abstraction, of ``sigma_T`` followed by
that river equilibrium. River subgames where both reaches are negligible are
skipped; ``skip_bound`` bounds what they could have added.

:func:`map_sigma` moves a solver's average strategy onto another tree of the
same trunk (matched by observed history, board and child actions), so the
strategies of rollout search, value-net search and the blueprint are scored
on one tree.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

import torch

from .combos import NUM_COMBOS, avoids_card, valid_masks
from .solver import RangeSolver, SolverConfig
from .tree import DECISION, VALUE, SubgameTree

C = NUM_COMBOS


def node_key(tree: SubgameTree, node: int) -> tuple:
    """Identity of a node across trees of the same trunk: action history and board."""
    return (tree.histories[node], tree.boards[int(tree.board_id[node])])


def map_sigma(
    src: RangeSolver,
    dst: RangeSolver,
    src_sigma: torch.Tensor | None = None,
    strict: bool = True,
) -> tuple[torch.Tensor, int]:
    """``src``'s strategy (average by default) on ``dst``'s decision nodes.

    Nodes are matched by :func:`node_key` and children by their concrete
    actions. Returns the ``[D, A, C]`` sigma of ``dst`` and the number of
    ``dst`` decision nodes without a match (uniform there); ``strict`` raises
    instead."""
    sig = src.average_strategy() if src_sigma is None else src_sigma
    st, dt = src.tree, dst.tree
    index = {}
    for d, n in enumerate(src.dec_nodes.tolist()):
        if int(st.kind[n]) == DECISION:
            index[node_key(st, n)] = (d, n)
    out = dst.uniform.clone().to(sig.dtype)
    missing = 0
    for d, n in enumerate(dst.dec_nodes.tolist()):
        if int(dt.kind[n]) != DECISION:
            continue
        hit = index.get(node_key(dt, n))
        if hit is None:
            if strict:
                raise KeyError(f"no source node for {node_key(dt, n)}")
            missing += 1
            continue
        ds, ns = hit
        s_acts = st.child_actions(ns)
        d_acts = dt.child_actions(n)
        if sorted(s_acts) != sorted(d_acts):
            if strict:
                raise KeyError(f"child actions differ at {node_key(dt, n)}: {s_acts} vs {d_acts}")
            missing += 1
            continue
        out[d] = 0
        for j, a in enumerate(d_acts):
            out[d, j] = sig[ds, s_acts.index(a)].to(out.device)
    return out, missing


def _leaf_info(tree: SubgameTree) -> list[dict]:
    info = []
    for n in (tree.kind == VALUE).nonzero().flatten().tolist():
        board = tuple(tree.boards[int(tree.board_id[n])])
        if len(board) != 4:
            raise ValueError("exact evaluation needs VALUE leaves at the end of the turn")
        c0, c1 = (int(x) for x in tree.contrib[n].tolist())
        if c0 != c1:
            raise ValueError("a VALUE leaf with unequal contributions")
        info.append({"node": n, "board": board, "c": c0, "state": tree.states[n]})
    return info


@torch.no_grad()
def trunk_exploitability(
    solver: RangeSolver,
    sigma: torch.Tensor,
    river_spec: Any,
    game_config: Any,
    ranges: torch.Tensor | None = None,
    iterations: int = 400,
    mix: float = 0.05,
    min_mass: float = 1e-7,
    batch: int = 512,
    river_cfg: SolverConfig | None = None,
    log: Any = None,
    keep_river: list | None = None,
    root_values: bool = False,
) -> dict:
    """Exploitability of ``sigma`` (``[D, A, C]`` on ``solver``'s tree, whose
    depth-limit leaves are ``VALUE`` nodes at the end of the turn) with an
    exact river; see the module docstring. ``ranges`` defaults to the solver's
    plain root ranges. Values in chips per hand. ``keep_river`` (a list) receives
    ``(leaf history, 5-card board, river tree, [N, C] river strategy)`` per
    solved river subgame, for tests. ``root_values`` adds ``"root_values"``:
    each player's per-combo best-response counterfactual values at the root,
    ``[2, C]`` (weighted by the other player's root reach)."""
    from .batch_solver import BatchRiverSolver, river_tree
    from .value_leaf import FixedLeafValues

    t0 = time.perf_counter()
    tree = solver.tree
    dev = solver.device
    dtype = solver.dtype
    root = solver.ranges if ranges is None else ranges.to(dev, dtype)
    Z = float(solver.pair_mass(root))
    leaves = _leaf_info(tree)
    L = len(leaves)
    leaf_vals = torch.zeros(2, L, C, device=dev, dtype=dtype)
    stats = {"leaves": L, "instances": 0, "skipped": 0, "river_exploit_max": 0.0}
    skip_bound = 0.0
    root_mass = root.sum(1)  # [2]
    if L:
        reach = solver.forward(sigma, root)
        ids = torch.tensor([lf["node"] for lf in leaves], device=dev)
        lr = reach[:, ids]  # [2, L, C]
        avoid = avoids_card(dev).to(dtype)
        button = int(leaves[0]["state"].button)
        stack_total = int(game_config.stacks[0])
        # relative reach mass of both players at every (leaf, river card)
        mass = (lr @ avoid.t()) / root_mass.clamp(min=1e-30)[:, None, None]  # [2, L, 52]
        top = mass.max(0).values.cpu()  # [L, 52]
        # instances grouped by contribution (the river tree depends only on it)
        groups: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for j, lf in enumerate(leaves):
            for x in range(52):
                if x in lf["board"]:
                    continue
                if float(top[j, x]) < min_mass:
                    # either BR gains at most the opponent's mass times the largest payoff
                    stats["skipped"] += 1
                    skip_bound += float(top[j, x]) * stack_total / 44.0
                    continue
                groups[lf["c"]].append((j, x))
        w = 1.0 / 44.0
        for c, insts in sorted(groups.items()):
            rt = river_tree(game_config, c, river_spec, button=button, device=dev)
            for s in range(0, len(insts), batch):
                chunk = insts[s : s + batch]
                jj = torch.tensor([j for j, _ in chunk], device=dev)
                xx = torch.tensor([x for _, x in chunk], device=dev)
                boards = torch.tensor(
                    [list(leaves[j]["board"]) + [x] for j, x in chunk], device=dev
                )
                actual = lr[:, jj].permute(1, 0, 2) * avoid[xx][:, None, :]  # [B, 2, C]
                valid = valid_masks(boards.tolist(), dev).to(dtype)  # [B, C]
                uni = valid / valid.sum(1, keepdim=True)
                tot = actual.sum(2, keepdim=True)
                norm = torch.where(tot > 0, actual / tot.clamp(min=1e-30), uni[:, None, :])
                mixed = (1 - mix) * norm + mix * uni[:, None, :]
                bs = BatchRiverSolver(rt, boards, mixed, river_cfg, device=dev)
                bs.solve(iterations)
                ex = bs.exploitability()
                rel = ex["exploitability"] / float(2 * c)
                stats["river_exploit_max"] = max(stats["river_exploit_max"], float(rel.max()))
                sig_r = bs.average_strategy()
                if keep_river is not None:
                    for b, (j, x) in enumerate(chunk):
                        hist = tree.histories[leaves[j]["node"]]
                        keep_river.append((hist, (*leaves[j]["board"], x), rt, sig_r[:, b].cpu()))
                for p in (0, 1):
                    v = bs.root_values(p, sig_r, best_response=True, ranges=actual)  # [B, C]
                    leaf_vals[p].index_add_(0, jj, (w * v).to(dtype))
                stats["instances"] += len(chunk)
                del bs, sig_r, ex, v, actual, mixed, norm  # keep one batch alive at a time
            if log:
                log(f"#   c={c}: {len(insts)} river subgames, {time.perf_counter() - t0:.0f}s")
    t_river = time.perf_counter() - t0
    # back the leaf values up through the trunk with the solver's own terminal
    # evaluator (its all-in matrices), the leaf provider swapped for fixed values
    term = solver.terminals
    if L:
        row = {lf["node"]: i for i, lf in enumerate(leaves)}
        perm = torch.tensor([row[int(n)] for n in term.value_ids.tolist()], device=dev)
        fixed = FixedLeafValues(leaf_vals[:, perm])
    else:
        fixed = FixedLeafValues(leaf_vals)
    old = term.value_leaves
    term.value_leaves = fixed
    br = []
    rv = []
    try:
        for p in (0, 1):
            vb, _ = solver.values(p, sigma, True, root)
            br.append(float((root[p] * vb[0]).sum() / Z))
            rv.append(vb[0].clone())
    finally:
        term.value_leaves = old
    extra = {"root_values": torch.stack(rv)} if root_values else {}
    return {
        **extra,
        "br": br,
        "exploitability": (br[0] + br[1]) / 2,
        "skip_bound": skip_bound * float(root_mass[0] * root_mass[1]) / Z if L else 0.0,
        "river_seconds": t_river,
        "seconds": time.perf_counter() - t0,
        **stats,
    }


__all__ = ["map_sigma", "node_key", "trunk_exploitability"]
