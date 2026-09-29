"""``SearchAgent``: blueprint preflop, real-time depth-limited re-solving after.

At every postflop decision:

1. **Ranges at the street root.** From the continual-resolving cache when the
   previous solve reached this street (agent reach, opponent reach and the
   opponent's best-response values there), otherwise from the blueprint
   reach along the observed history (:func:`~pokerbot.search.blueprint.range_reach`),
   with our own hole cards removed from the opponent's range.
2. **Tree.** Rooted at the start of the current street, with this street's
   observed actions forced in (off-tree sizes become extra branches where they
   happened). Our own earlier actions on this street are locked to the
   strategy we actually played (cached from that solve).
3. **Solve** with DCFR within the street's time budget (tree building and
   rollouts count against it; at least ``min_iterations`` are run), with the
   safe-resolving gadget when ``gadget.safe``.
4. **Act.** Read the average strategy of our actual combo at the current node,
   sample a child, and play its concrete action (sizes computed exactly as
   :mod:`pokerbot.env.actions` does). Cache the played strategy (for locking)
   and the values at the next street's roots (continual resolving).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..agents.base import BaseAgent
from .abstract import legal_options, map_concrete, to_action
from .blueprint import make_blueprint, policy_matrix, policy_vector, range_reach
from .combos import NUM_COMBOS, combo_index, valid_mask
from .config import SearchConfig, search_config
from .gadget import (
    ContinualCache,
    Gadget,
    blueprint_terminate_values,
    history_key,
    mixed_prior,
    normalise_entry,
)
from .leaf import build_leaf_rollouts
from .solver import RangeSolver
from .tree import LEAF, TreeBuilder


def _engine_config(engine: Any, config: Any) -> Any:
    if isinstance(config, engine.GameConfig):
        return config
    return engine.GameConfig(
        num_players=int(config.num_players),
        stacks=[int(s) for s in config.stacks],
        small_blind=int(config.small_blind),
        big_blind=int(config.big_blind),
        ante=int(getattr(config, "ante", 0)),
    )


class SearchAgent(BaseAgent):
    name = "search"

    def __init__(
        self,
        blueprint: Any,
        config: SearchConfig | dict | str | Path | None = None,
        name: str | None = None,
        **overrides: Any,
    ) -> None:
        super().__init__(name)
        self.blueprint = blueprint
        if isinstance(config, SearchConfig):
            self.cfg = config
        else:
            self.cfg = search_config(config, **overrides)
        self.device = self.cfg.torch_device()
        self.cache = ContinualCache()
        self._roots: dict = {}
        self._played: dict = {}
        self.last_stats: dict = {}
        self.stats: list[dict] = []

    # -- Agent protocol -----------------------------------------------------

    def new_hand(self, seat: int, config: Any) -> None:
        super().new_hand(seat, config)
        self.cache.clear()
        self._roots.clear()
        self._played.clear()

    def observe_end(self, state: Any) -> None:
        self.cache.clear()
        self._roots.clear()
        self._played.clear()

    def act(self, state: Any, seat: int, rng: np.random.Generator) -> Any:
        if state.street == 0:
            return self._blueprint_action(state, seat, rng)
        try:
            return self._search_action(state, seat, rng)
        except Exception:
            if not self.cfg.fallback_on_error:
                raise
            self.last_stats = {"fallback": True, "street": int(state.street)}
            return self._blueprint_action(state, seat, rng)

    # -- helpers ------------------------------------------------------------

    def _blueprint_action(self, state: Any, seat: int, rng: np.random.Generator) -> Any:
        bp = self.blueprint
        probs = np.asarray(policy_vector(bp, state, seat), dtype=np.float64)
        idx = int(rng.choice(len(probs), p=probs / probs.sum()))
        opt = next(o for o in legal_options(state, bp.spec) if o.index == idx)
        return to_action(self._engine, opt.kind, opt.amount)

    def _root_info(self, key, config, state, seat, pre, board) -> dict:
        info = self._roots.get(key)
        if info is not None:
            return info
        opp = 1 - seat
        dev = self.device
        hole = list(state.hole_cards(seat))
        exclude = {opp: hole} if self.cfg.remove_own_blockers else None
        entry = self.cache.get(key)
        bp_r = range_reach(
            self.blueprint, config, state.button, board, pre, self._engine, dev, exclude
        )
        ranges = bp_r.clone()
        terminate = None
        if entry is not None:
            a, o, t = normalise_entry(entry)
            ranges[seat] = a.to(dev)
            if not self.cfg.gadget.safe and float(o.sum()) > 0:
                ranges[opp] = o.to(dev)
                if exclude:
                    ranges[opp] *= valid_mask(hole, dev)
            terminate = t.to(dev)
        ranges = ranges / ranges.sum(1, keepdim=True).clamp(min=1e-30)
        info = {"ranges": ranges, "terminate": terminate, "cached": entry is not None}
        self._roots[key] = info
        return info

    def _locks(self, tree: Any, key: tuple, seat: int) -> dict[int, torch.Tensor]:
        locks: dict[int, torch.Tensor] = {}
        for node, _slot in tree.path_nodes:
            if int(tree.actor[node]) != seat:
                continue
            acts = tree.child_actions(node)
            played = self._played.get((key, tree.histories[node]))
            strat = torch.zeros(NUM_COMBOS, len(acts))
            if played is not None:
                old_acts, old = played
                for j, a in enumerate(acts):
                    if a in old_acts:
                        strat[:, j] = old[:, old_acts.index(a)]
            if played is None or float(strat.sum()) <= 0:
                st = tree.states[node]
                P = policy_matrix(self.blueprint, st, seat).expand(NUM_COMBOS, -1)
                opts = legal_options(st, self.blueprint.spec)
                for j, (k, amt) in enumerate(acts):
                    for i, w in map_concrete(st, self.blueprint.spec, k, amt, opts).items():
                        strat[:, j] += w * P[:, i]
            locks[node] = strat
        return locks

    def _search_action(self, state: Any, seat: int, rng: np.random.Generator) -> Any:
        t0 = time.perf_counter()
        cfg = self.cfg
        engine = self._engine
        config = _engine_config(engine, self.config if self.config is not None else state.config)
        street = int(state.street)
        board = list(state.board)
        hist = list(state.history)
        pre = [h for h in hist if int(h[0]) < street]
        cur = [h for h in hist if int(h[0]) == street]
        key = (history_key(pre), tuple(board))
        info = self._root_info(key, config, state, seat, pre, board)
        tree = TreeBuilder(
            config, state.button, board, pre, cur, cfg.tree, seat, engine, self.device
        ).build()
        t_tree = time.perf_counter()
        gadget = None
        if cfg.gadget.safe:
            if info["terminate"] is None:
                info["terminate"] = blueprint_terminate_values(
                    tree.states[0],
                    board,
                    self.blueprint,
                    seat,
                    info["ranges"][seat],
                    config,
                    cfg.gadget.rollouts,
                    engine,
                    self.device,
                    cfg.seed,
                )
            prior = mixed_prior(
                info["ranges"][1 - seat], valid_mask(board, self.device), cfg.gadget.prior_mix
            )
            if cfg.remove_own_blockers:
                prior = prior * valid_mask(state.hole_cards(seat), self.device)
            gadget = Gadget(1 - seat, prior, info["terminate"])
        rollouts = None
        if bool((tree.kind == LEAF).any()):
            rollouts = build_leaf_rollouts(
                tree, self.blueprint, config, cfg.leaf, engine, self.device
            )
        locks = self._locks(tree, key, seat)
        solver = RangeSolver(tree, info["ranges"], cfg.solver, rollouts, gadget, locks)
        t_setup = time.perf_counter()
        solver.solve(iterations=min(cfg.min_iterations, cfg.solver.iterations))
        left = cfg.budget(street) - (time.perf_counter() - t0)
        remaining_iters = cfg.solver.iterations - solver.iterations_done
        if left > 0 and remaining_iters > 0:
            solver.solve(iterations=remaining_iters, time_budget=left)
        node = tree.current_node
        strat = solver.node_strategy(node).cpu()
        hole = list(state.hole_cards(seat))
        probs = strat[combo_index(hole[0], hole[1])].double().numpy()
        if not np.isfinite(probs).all() or probs.sum() <= 0:
            probs = np.ones(len(probs))
        slot = int(rng.choice(len(probs), p=probs / probs.sum()))
        acts = tree.child_actions(node)
        kind, amount = acts[slot]
        self._played[(key, tree.histories[node])] = (acts, strat)
        prefix = (*tree.histories[node], (street, seat, kind, amount))
        stored = self.cache.store(solver, tree, seat, prefix)
        t_end = time.perf_counter()
        self.last_stats = {
            "street": street,
            "nodes": tree.num_nodes,
            "iterations": solver.iterations_done,
            "tree_seconds": t_tree - t0,
            "setup_seconds": t_setup - t_tree,
            "solve_seconds": solver.solve_time,
            "total_seconds": t_end - t0,
            "cached_root": info["cached"],
            "cache_entries": stored,
        }
        self.stats.append(self.last_stats)
        return to_action(engine, kind, amount)


def make_search_agent(blueprint_spec: str = "uniform", **kwargs: Any) -> SearchAgent:
    """Factory behind ``search:<blueprint spec>`` in the match runner:
    ``search:uniform``, ``search:blueprint:<strategy file>`` (tabular MCCFR),
    ``search:neural:<checkpoint dir>`` (Deep CFR), or a registered prefix.

    ``kwargs`` may hold ``config`` (a YAML path or mapping), ``blueprint``
    (keyword arguments for the blueprint factory, e.g. ``{last_n: 8}`` for a
    neural blueprint) and overrides of the ``search:`` section
    (``time_budget``, ``tree``, ``solver``, ...).
    """
    config = kwargs.pop("config", None)
    bp_kwargs = dict(kwargs.pop("blueprint", None) or {})
    name = kwargs.pop("name", None) or f"search:{blueprint_spec or 'uniform'}"
    bp = make_blueprint(blueprint_spec or "uniform", **bp_kwargs)
    return SearchAgent(bp, config, name=name, **kwargs)
