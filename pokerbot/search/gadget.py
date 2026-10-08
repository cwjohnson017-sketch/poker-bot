"""Safe resolving gadget and the continual-resolving cache.

Resolve gadget (Burch, Johanson & Bowling 2014, CFR-D; used by DeepStack for
continual re-solving). The opponent of the searcher enters the subgame
through an extra decision per hole combo ``c'``:

* **terminate**: receive ``T(c')``, the counterfactual value the opponent
  could already secure against the strategy we played so far (from the
  previous solve, or the blueprint when there is none);
* **enter**: play the subgame; their reach at the subgame root becomes
  ``prior(c') * sigma_enter(c')``.

``sigma_enter`` is learned by regret matching inside the solver, alongside
the rest of the tree. At a solution of the gadget game the opponent's
best-response value from entering satisfies
``sum_c' prior(c') * max(0, BR(c') - T(c')) <= eps``: the re-solved strategy
gives no combo more than it could already get, so it is no more exploitable
than the strategy it refines (Burch et al., Theorem 1). With ``safe=False``
(unsafe resolving) the opponent's root range is simply the prior.

Units: ``T`` and the subgame's counterfactual values are both weighted by the
searcher's root reach, so they must be computed with the same range vector.
:class:`ContinualCache` stores, after each solve, the searcher's reach, the
opponent's reach and the opponent's best-response values at every chance
child the game can still reach; the next street's solve starts from exactly
those vectors (continual resolving).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from .combos import NUM_COMBOS
from .leaf import build_root_rollouts
from .tree import CHANCE, SubgameTree


@dataclass
class Gadget:
    player: int  # the opponent, who may terminate
    prior: torch.Tensor  # [C] opponent weights at the gadget chance node
    terminate: torch.Tensor  # [C] terminate values T(c')


def mixed_prior(opp_range: torch.Tensor, valid: torch.Tensor, mix: float) -> torch.Tensor:
    """Opponent range normalised to 1, mixed with ``mix`` uniform over ``valid``."""
    r = opp_range * valid
    s = r.sum()
    r = r / s if float(s) > 0 else valid.to(r.dtype) / valid.sum()
    u = valid.to(r.dtype) / valid.sum().clamp(min=1)
    return (1 - mix) * r + mix * u


def blueprint_terminate_values(
    state: Any,
    board: list[int],
    bp: Any,
    agent: int,
    agent_range: torch.Tensor,
    game_config: Any,
    rollouts: int,
    engine: Any,
    device: torch.device,
    seed: int = 0,
) -> torch.Tensor:
    """Opponent counterfactual values at ``state`` when both players follow the
    blueprint, estimated by rollouts, weighted by ``agent_range``."""
    rs = build_root_rollouts(state, board, bp, game_config, rollouts, engine, device, seed)
    opp = 1 - agent
    return rs.values(opp, agent_range.to(device)[None].float())[0]


@torch.no_grad()
def tree_terminate_values(
    solver: Any, sigma: torch.Tensor, player: int, best_response: bool = True
) -> torch.Tensor:
    """Terminate values ``[C]`` for the gadget ``player`` (the opponent) computed
    in ``solver``'s own tree: their counterfactual values at the root when the
    searcher plays ``sigma`` (``[D, A, C]``, e.g. the blueprint's strategy from
    :func:`~pokerbot.search.tree_policy.blueprint_profile`), on the plain root
    ranges.

    With ``best_response`` the opponent best-responds to ``sigma`` (the CFR-D
    value they could already secure against the strategy being refined);
    otherwise they also play ``sigma``. The tree's own leaf values are used
    (with value-net leaves, the net on both players' reaches under ``sigma``),
    so ``T`` and the gadget's entry values come from the same game and are
    weighted by the same searcher range, without rollout noise."""
    v, _ = solver.values(player, sigma, best_response=best_response, root=solver.ranges)
    return v[0]


def gadget_entry_values(solver: Any, sigma: torch.Tensor | None = None) -> torch.Tensor:
    """The opponent's best-response values for entering the subgame (``[C]``)
    against the searcher's average strategy, on the searcher's root range."""
    g = solver.gadget
    root = solver.ranges.clone()
    root[g.player] = g.prior.to(root)
    v, _ = solver.values(g.player, sigma, best_response=True, root=root)
    return v[0]


def gadget_violation(solver: Any) -> float:
    """``sum_c' prior(c') * max(0, BR_enter(c') - T(c'))`` for the current average."""
    g = solver.gadget
    e = gadget_entry_values(solver)
    return float((g.prior.to(e) * (e - g.terminate.to(e)).clamp(min=0)).sum())


@dataclass
class CacheEntry:
    agent_reach: torch.Tensor  # [C]
    opp_reach: torch.Tensor  # [C]
    opp_values: torch.Tensor  # [C] opponent best-response cfvs, weighted by agent_reach


def history_key(history) -> tuple:
    """Hashable form of a history of ``(street, player, action)`` tuples."""
    out = []
    for h in history:
        if len(h) == 4:
            out.append(tuple(int(x) for x in h))
        else:
            s, p, a = h
            k = int(a.kind)
            out.append((int(s), int(p), k, int(a.amount) if k == 2 else 0))
    return tuple(out)


@dataclass
class ContinualCache:
    entries: dict = field(default_factory=dict)

    def clear(self) -> None:
        self.entries.clear()

    def get(self, key: tuple) -> CacheEntry | None:
        return self.entries.get(key)

    @torch.no_grad()
    def store(self, solver: Any, tree: SubgameTree, agent: int, prefix: tuple) -> int:
        """Store every chance child whose street history starts with ``prefix``
        (the observed path plus the action we are about to take)."""
        chance_children = (
            (tree.parent >= 0) & (tree.kind[tree.parent.clamp(min=0)] == CHANCE)
        ).nonzero()
        nodes = [
            int(n)
            for n in chance_children.flatten().tolist()
            if tree.histories[int(n)][: len(prefix)] == prefix
        ]
        if not nodes:
            return 0
        opp = 1 - agent
        sigma = solver.average_strategy()
        root = solver.root_reach()
        v, reach = solver.values(opp, sigma, best_response=True, root=root)
        base = history_key(tree.root_history)
        idx = torch.tensor(nodes, device=v.device)
        a_r, o_r, o_v = reach[agent, idx], reach[opp, idx], v[idx]
        for j, n in enumerate(nodes):
            key = (base + tree.histories[n], tuple(tree.boards[int(tree.board_id[n])]))
            self.entries[key] = CacheEntry(a_r[j].clone(), o_r[j].clone(), o_v[j].clone())
        return len(nodes)


def normalise_entry(e: CacheEntry) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Agent range scaled to sum 1 and the opponent values scaled alike."""
    s = float(e.agent_reach.sum())
    if s <= 0:
        return e.agent_reach, e.opp_reach, e.opp_values
    so = float(e.opp_reach.sum())
    opp = e.opp_reach / so if so > 0 else e.opp_reach
    return e.agent_reach / s, opp, e.opp_values / s


__all__ = [
    "NUM_COMBOS",
    "CacheEntry",
    "ContinualCache",
    "Gadget",
    "blueprint_terminate_values",
    "gadget_entry_values",
    "gadget_violation",
    "history_key",
    "mixed_prior",
    "normalise_entry",
    "tree_terminate_values",
]
