"""The blueprint's strategy on every decision node of a search tree.

:func:`blueprint_profile` puts the blueprint on a solver's tree as a
``[D, A, C]`` strategy: the exact-evaluation baseline
(:mod:`~pokerbot.search.spot_eval`) and the gadget's terminate values computed
in the search tree (``gadget.terminate: blueprint | blueprint_br``,
:mod:`~pokerbot.search.gadget`).

Each node is queried on its own board. ``tree.states`` below a chance node
hold the chance template's engine state, whose board has whatever card the
builder dealt (only its betting is right), so the board comes from
``tree.boards``. (``spot_eval.blueprint_profile`` before 2026-10-06 passed the
stored state as it was: every turn decision but one turn card's got the
blueprint's strategy for the template's turn card.)

Children are matched to the blueprint's legal options at the node's state by
concrete action. Mass on blueprint actions the tree lacks (sizes the node
budget dropped) is renormalised over the tree's children (uniform where none
match) and reported as ``dropped``. A ``LEAF`` node of a rollout tree plays
the plain blueprint continuation (index 0).

**Speed.** A flop tree has thousands of decision nodes (2,572 at the
production 6,000-node budget), and querying a neural blueprint once per node
costs about 5 ms, mostly Python and small kernels (14 s per tree). For a
single-net blueprint without own-reach weighting (a distilled Deep CFR run,
:func:`~pokerbot.search.vec_rollouts.supports`), :func:`node_policies` instead
replays every node's state into one ``VecNLHE`` slot and evaluates all 1326
combos of many slots per network call
(:func:`~pokerbot.search.vec_rollouts.slot_policies`). Other blueprints fall
back to :func:`~pokerbot.search.blueprint.policy_matrix` per node. The child
matching is computed once per betting history, since the turn subtrees repeat
it for every turn card.
"""

from __future__ import annotations

from typing import Any

import torch

from .abstract import RAISE, CardView, legal_options
from .blueprint import policy_matrix
from .combos import NUM_COMBOS
from .tree import DECISION, LEAF, SubgameTree

C = NUM_COMBOS


def _padded_board(board: tuple[int, ...] | list[int]) -> list[int]:
    """``board`` completed to 5 cards with the lowest free cards (the features
    only see the dealt prefix)."""
    b = [int(c) for c in board]
    free = (c for c in range(52) if c not in set(b))
    return b + [next(free) for _ in range(5 - len(b))]


@torch.no_grad()
def node_policies(
    tree: SubgameTree,
    nodes: list[int],
    bp: Any,
    game_config: Any = None,
    device: torch.device | None = None,
    slots_per_call: int = 48,
) -> torch.Tensor:
    """``[n, 1326, A]`` blueprint policy (``bp.spec`` indices) of every combo for
    the actor at each decision node in ``nodes``, normalised over the legal
    actions. Rows of combos that share a card with the node's board are
    unspecified (they have zero reach). ``game_config`` defaults to the first
    node's ``state.config``."""
    from . import vec_rollouts as vr

    dev = tree.device if device is None else torch.device(device)
    A = bp.spec.num_actions
    if not nodes:
        return torch.zeros(0, C, A, device=dev)
    if not vr.supports(bp):
        out = torch.empty(len(nodes), C, A, device=dev)
        for i, n in enumerate(nodes):
            # the stored state of a node below a chance node is its template's: its board
            # has whatever card the builder dealt, so show the node's own board
            st = CardView(tree.states[n], tree.boards[int(tree.board_id[n])])
            out[i] = policy_matrix(bp, st, int(tree.actor[n])).to(dev).expand(C, -1)
        return out
    from ..blueprint.deepcfr.features import features_from_obs
    from ..blueprint.deepcfr.strength import load_strength

    agent = bp.agent
    if game_config is None:
        game_config = tree.states[nodes[0]].config
    ros = [
        vr.Rollout(
            tree.states[n], _padded_board(tree.boards[int(tree.board_id[n])]), int(tree.actor[n])
        )
        for n in nodes
    ]
    env = vr.make_env(ros, bp.spec, game_config, dev)
    actors = torch.tensor([int(tree.actor[n]) for n in nodes], device=dev)
    if not torch.equal(env.actor.to(dev), actors):
        raise RuntimeError("replayed blueprint states disagree with the tree's actors")
    feats = features_from_obs(env.obs(**agent.features.obs_kwargs()))
    st = agent.features.strength_tables
    cache = vr.StrengthCache(load_strength(st), dev) if st else None
    boards = torch.tensor([ro.board for ro in ros], dtype=torch.long, device=dev)
    out = torch.empty(len(nodes), C, A, device=dev)
    for lo in range(0, len(nodes), slots_per_call):
        idx = torch.arange(lo, min(len(nodes), lo + slots_per_call), device=dev)
        out[idx] = vr.slot_policies(agent, env, idx, feats, boards, cache)
    return out


@torch.no_grad()
def blueprint_profile(
    solver: Any, bp: Any, game_config: Any = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """The blueprint's strategy on every decision node of ``solver``'s tree,
    ``[D, A, C]``, and the blueprint mass on actions the tree lacks, ``[D, C]``
    (see the module docstring)."""
    tree = solver.tree
    dev, dt = solver.device, solver.dtype
    sigma = solver.uniform.clone()
    dropped = torch.zeros(solver.Dn, C, device=dev, dtype=dt)
    dec = solver.dec_nodes.tolist()
    kinds = tree.kind[solver.dec_nodes].tolist()
    for d, k in enumerate(kinds):
        if k == LEAF:
            sigma[d] = 0
            sigma[d, 0] = 1
    dd = [d for d, k in enumerate(kinds) if k == DECISION]
    if not dd:
        return sigma, dropped
    nodes = [dec[d] for d in dd]
    P = node_policies(tree, nodes, bp, game_config, dev)  # [n, C, A_bp]
    # tree child j -> blueprint index (-1: the blueprint has no such action), per history
    colmap = torch.full((len(nodes), solver.A), -1, dtype=torch.long)
    memo: dict[tuple, list[int]] = {}
    for i, n in enumerate(nodes):
        acts = tree.child_actions(n)
        key = (tree.histories[n], tuple(acts))
        cm = memo.get(key)
        if cm is None:
            col = {(o.kind, o.amount): o.index for o in legal_options(tree.states[n], bp.spec)}
            cm = [col.get((int(k), int(a) if int(k) == RAISE else 0), -1) for k, a in acts]
            memo[key] = cm
        colmap[i, : len(cm)] = torch.tensor(cm, dtype=torch.long)
    colmap = colmap.to(dev)
    have = colmap >= 0  # [n, A]
    ddt = torch.tensor(dd, device=dev)
    S = P.gather(2, colmap.clamp(min=0)[:, None, :].expand(-1, C, -1)).to(dt)  # [n, C, A]
    S = S * have[:, None, :]
    tot = S.sum(2, keepdim=True)
    legal = solver.legal[ddt].to(dt)  # [n, A]
    uni = (legal / legal.sum(1, keepdim=True).clamp(min=1))[:, None, :]
    S = torch.where(tot > 0, S / tot.clamp(min=1e-30), uni)
    sigma[ddt] = S.transpose(1, 2)
    dropped[ddt] = (1 - tot[..., 0]).clamp(min=0)
    return sigma, dropped


def blueprint_sigma(solver: Any, bp: Any, game_config: Any = None) -> torch.Tensor:
    """The blueprint's ``[D, A, C]`` strategy on ``solver``'s tree (see
    :func:`blueprint_profile`)."""
    return blueprint_profile(solver, bp, game_config)[0]


__all__ = ["blueprint_profile", "blueprint_sigma", "node_policies"]
