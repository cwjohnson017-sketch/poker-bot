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

A turn solve that stops at ``VALUE`` leaves at the end of turn betting
(``tree.depth_streets_turn: 0``) has no chance children. When its leaves are
valued by a river net averaged over the river cards (a provider with
``card_values``, :class:`~pokerbot.search.value_leaf.ValueLeafEvaluator`), the
cache stores the river roots below them instead, under the keys a chance
child would have, with the terms of that chance average as the opponent's
values (:func:`card_value_provider`, :meth:`ContinualCache.store`). A turn-end net
(:class:`~pokerbot.search.value_leaf.TurnEndLeafEvaluator`) has no per-card
values, and flop-end leaves (a flop solve with ``depth_streets: 0``) are not
handled: nothing is stored for them, and the next street starts from the
blueprint ranges with ``gadget.terminate``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from .combos import NUM_COMBOS, avoids_card
from .leaf import build_root_rollouts
from .tree import CHANCE, VALUE, SubgameTree

TURN = 2  # street of the trees whose VALUE leaves give river roots
TURN_BOARD = 4


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


def street_end_leaves(tree: SubgameTree, prefix: tuple = ()) -> list[int]:
    """The ``VALUE`` nodes of ``tree`` at the end of its root street (on the root's
    board: no chance node above them) whose history starts with ``prefix``."""
    root_len = len(tree.boards[int(tree.board_id[0])])
    return [
        n
        for n in (tree.kind == VALUE).nonzero().flatten().tolist()
        if len(tree.boards[int(tree.board_id[n])]) == root_len
        and tree.histories[n][: len(prefix)] == prefix
    ]


def card_value_provider(solver: Any, tree: SubgameTree) -> Any:
    """``solver``'s leaf-value provider when it gives the values at the river roots
    below ``tree``'s turn-end leaves, else ``None``.

    That needs a tree rooted on the turn (its ``VALUE`` leaves end turn betting,
    on 4-card boards) and a provider with ``card_values`` over 4-card boards: a
    river net averaged over the river cards
    (:class:`~pokerbot.search.value_leaf.ValueLeafEvaluator`, any river
    predictor). A turn-end net
    (:class:`~pokerbot.search.value_leaf.TurnEndLeafEvaluator`) predicts only the
    average over the river cards, so it gives ``None``; so do flop-end leaves
    (:class:`~pokerbot.search.value_leaf.FlopEndLeafEvaluator`, not handled)."""
    if int(tree.root_street) != TURN:
        return None
    provider = getattr(getattr(solver, "terminals", None), "value_leaves", None)
    if not callable(getattr(provider, "card_values", None)):
        return None
    if getattr(provider, "board_len", None) != TURN_BOARD:
        return None
    return provider


@dataclass
class ContinualCache:
    entries: dict = field(default_factory=dict)
    # VALUE leaves under the prefix of the last store whose next-street roots could
    # not be stored (a turn-end net, or flop-end leaves): continual resolving stops there
    skipped_leaves: int = 0

    def clear(self) -> None:
        self.entries.clear()

    def get(self, key: tuple) -> CacheEntry | None:
        return self.entries.get(key)

    @torch.no_grad()
    def store(self, solver: Any, tree: SubgameTree, agent: int, prefix: tuple) -> int:
        """Store every next-street root below ``prefix`` (the observed path plus
        the action we are about to take); returns the number of entries.

        * Every chance child whose history starts with ``prefix``.
        * On a turn tree ending at ``VALUE`` leaves valued by a river net
          averaged over the river cards (:func:`card_value_provider`), every
          river root below a leaf whose history starts with ``prefix``: one entry
          per river card ``x`` not on the board, under the key a chance child
          dealing ``x`` would have, with the terms of the chance average
          (:meth:`_store_river_roots`).

        Leaves with any other provider (a turn-end net) store nothing; their
        number is in :attr:`skipped_leaves`."""
        chance_children = (
            (tree.parent >= 0) & (tree.kind[tree.parent.clamp(min=0)] == CHANCE)
        ).nonzero()
        nodes = [
            int(n)
            for n in chance_children.flatten().tolist()
            if tree.histories[int(n)][: len(prefix)] == prefix
        ]
        leaves = street_end_leaves(tree, prefix)
        provider = card_value_provider(solver, tree) if leaves else None
        self.skipped_leaves = 0 if provider is not None else len(leaves)
        if provider is None:
            leaves = []
        if not nodes and not leaves:
            return 0
        opp = 1 - agent
        sigma = solver.average_strategy()
        root = solver.root_reach()
        base = history_key(tree.root_history)
        if not nodes:
            # the reach is the forward pass solver.values would run; skip its value
            # pass, which would run the net on every leaf
            reach = solver.forward(sigma, root)
            return self._store_river_roots(provider, solver, tree, agent, leaves, reach, base)
        v, reach = solver.values(opp, sigma, best_response=True, root=root)
        idx = torch.tensor(nodes, device=v.device)
        a_r, o_r, o_v = reach[agent, idx], reach[opp, idx], v[idx]
        for j, n in enumerate(nodes):
            key = (base + tree.histories[n], tuple(tree.boards[int(tree.board_id[n])]))
            self.entries[key] = CacheEntry(a_r[j].clone(), o_r[j].clone(), o_v[j].clone())
        stored = len(nodes)
        if leaves:
            stored += self._store_river_roots(provider, solver, tree, agent, leaves, reach, base)
        return stored

    def _store_river_roots(
        self,
        provider: Any,
        solver: Any,
        tree: SubgameTree,
        agent: int,
        leaves: list[int],
        reach: torch.Tensor,
        base: tuple,
    ) -> int:
        """The river roots below the turn-end ``leaves``, from both players'
        reaches ``reach [2, N, C]`` (the forward pass of the stored profile).

        For a leaf ``n`` on board ``b4`` and a river card ``x`` not on it, the key
        is ``(base + histories[n], b4 + (x,))``, the key of a chance child dealing
        ``x`` below ``n`` in a tree solved to showdown (a chance child's history is
        its parent's betting history). The entry holds both reaches at ``n``
        masked by ``[avoids x]`` and the opponent's value there in chips, the
        solver's convention for a chance child (weighted by our reach, not
        multiplied by the chance weight ``1 / 44``):

            opp_values(c) = [c avoids x] * m^x_agent(c) * pot * ev^x_opp(c)

        with ``m^x_agent = blocked_sum(pi_agent(n) * [avoids x])``, ``pot = 2 c``
        and ``ev^x_opp`` the river net's output for the opponent on ``b4 + x``
        (``RiverAverage.card_values``). These are the terms of the leaf value
        the solve used: they average to it with weight ``1 / 44``."""
        opp = 1 - agent
        term = solver.terminals
        pos = {n: i for i, n in enumerate(term.value_ids.tolist())}
        cards, v = provider.card_values(opp, reach[:, term.value_ids], [pos[n] for n in leaves])
        avoid = avoids_card(reach.device).to(reach.dtype)[cards]  # [n, 48, C]
        ids = torch.tensor(leaves, dtype=torch.long, device=reach.device)
        a_r = reach[agent, ids][:, None, :] * avoid
        o_r = reach[opp, ids][:, None, :] * avoid
        for j, (n, xs) in enumerate(zip(leaves, cards.tolist(), strict=True)):
            hist = base + tree.histories[n]
            board = tuple(tree.boards[int(tree.board_id[n])])
            for k, x in enumerate(xs):
                self.entries[(hist, (*board, int(x)))] = CacheEntry(a_r[j, k], o_r[j, k], v[j, k])
        return len(leaves) * int(cards.shape[1])


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
    "card_value_provider",
    "gadget_entry_values",
    "gadget_violation",
    "history_key",
    "mixed_prior",
    "normalise_entry",
    "street_end_leaves",
    "tree_terminate_values",
]
