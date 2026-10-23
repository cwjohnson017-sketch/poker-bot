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
   safe-resolving gadget when ``gadget.safe``. Depth-limit leaves are valued
   by blueprint rollouts (``leaf.mode: rollouts``) or by a value net on both
   players' current reaches (``leaf.mode: value_net``, see
   :mod:`pokerbot.search.value_leaf`): a river net averaged over the river
   cards, or a turn-end net (one row per leaf), picked by the checkpoint's kind.
   A turn-start net in ``leaf.net`` values the flop-end leaves of flop solves
   with ``tree.depth_streets: 0`` (averaged over the turn cards); trees rooted
   on the turn then use ``leaf.turn_net`` (a turn-end net) when it is set, for
   turn solves with ``tree.depth_streets_turn: 0``.
4. **Act.** Read the average strategy of our actual combo at the current node,
   sample a child, and play its concrete action (sizes computed exactly as
   :mod:`pokerbot.env.actions` does). Cache the played strategy (for locking)
   and the values at the next street's roots (continual resolving): the chance
   children under our action, or, for a turn tree ending at turn-end leaves
   valued by a river net, the river roots below those leaves
   (:meth:`~pokerbot.search.gadget.ContinualCache.store`).

With ``record=True`` every decision of the current hand is also kept in
``decisions`` (reset by ``new_hand``): the strategy of all 1326 combos over the
concrete actions available, both players' reach at the decision, the action
taken and, for searched decisions, the expected chips of every action for
every combo and, for our own combo, against every opponent combo
(:meth:`SearchAgent._search_record`, :func:`pair_values`). Recording costs two
extra value passes per searched decision and never changes the action played.
"""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ..agents.base import BaseAgent
from .abstract import legal_options, map_concrete, to_action
from .blueprint import make_blueprint, policy_matrix, policy_vector, range_reach
from .combos import NUM_COMBOS, blocked_sum, combo_index, valid_mask
from .config import SearchConfig, search_config
from .gadget import (
    ContinualCache,
    Gadget,
    blueprint_terminate_values,
    history_key,
    mixed_prior,
    normalise_entry,
    tree_terminate_values,
)
from .leaf import build_leaf_rollouts
from .solver import RangeSolver
from .tree import LEAF, VALUE, TreeBuilder
from .tree_policy import blueprint_profile
from .value_leaf import FixedLeafValues, make_leaf_evaluator, predictor_kind


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


def played_lock(
    acts: list[tuple[int, int]], slot: int, old_acts: list[tuple[int, int]], old: torch.Tensor
) -> torch.Tensor:
    """``[C, len(acts)]`` lock of one of our earlier decisions on this street: the
    strategy ``old`` we played there (over ``old_acts``) on the new tree's
    children ``acts``, of which ``slot`` is the action we took.

    The taken action keeps its exact probability, so our range below it is the
    one we really have. When the new tree lacks some sizes we could have played
    (a forced off-tree branch can push the budget into dropping them), their
    mass goes to the other children in proportion to what they already have
    (uniformly where they have none), never to the taken action."""
    C = old.shape[0]
    strat = torch.zeros(C, len(acts), dtype=old.dtype)
    for j, a in enumerate(acts):
        if a in old_acts:
            strat[:, j] = old[:, old_acts.index(a)]
    missing = (old.sum(1) - strat.sum(1)).clamp(min=0)  # [C]
    others = [j for j in range(len(acts)) if j != slot]
    if not others or float(missing.max()) <= 0:
        return strat
    rest = strat[:, others]
    tot = rest.sum(1, keepdim=True)
    share = torch.where(tot > 0, rest / tot.clamp(min=1e-30), 1.0 / len(others))
    strat[:, others] = rest + missing[:, None] * share
    return strat


@torch.no_grad()
def pair_values(solver: RangeSolver, tree: Any, seat: int, combo: int) -> torch.Tensor:
    """``[1326, n]``: our expected chips (net for the hand) holding ``combo``
    after each of the ``n`` actions at the current node, against every opponent
    combo, under the solved average strategies. NaN on opponent combos that
    share a card with ``combo``.

    The opponent's values are computed with our root reach one-hot at
    ``combo`` (and our strategy at the current node set to 1 on every action,
    so each child is reached): the opponent's counterfactual value of combo
    ``o`` at child ``a`` is then ``pi(combo) * u_opp(o, combo | a)``, and ours
    is its negative (terminal payoffs are zero-sum). Value-net leaves see only
    ranges: their values come from the net on the real reaches (as in
    :meth:`SearchAgent._search_record`), with the opponent's per-combo value
    re-weighted from our real range to our one combo. With sampled chance cards
    (``tree.chance_cards``) a pair's live cards carry the sample's weights, which
    are exact only on average over pairs; with every card (production) the
    values are exact."""
    node = tree.current_node
    first, n = int(tree.first_child[node]), int(tree.num_children[node])
    opp = 1 - seat
    sigma = solver.average_strategy()
    root = solver.root_reach()
    hot_root = root.clone()
    hot_root[seat] = 0
    hot_root[seat, combo] = 1
    hot_sigma = sigma.clone()
    hot_sigma[int(solver.dec_index[node]), :n, combo] = 1
    hot = solver.forward(hot_sigma, hot_root)
    term = solver.terminals
    saved = term.value_leaves
    if len(term.value_ids):
        ids = term.value_ids
        real = solver.forward(sigma, root)[:, ids]
        v_real = saved.values(opp, real)  # [L, C]: ev_opp(o) * m_real(o)
        m_real = blocked_sum(real[seat])
        scale = torch.where(m_real > 0, blocked_sum(hot[seat, ids]) / m_real.clamp(min=1e-30), 0)
        fixed = torch.zeros(2, *v_real.shape, dtype=v_real.dtype, device=v_real.device)
        fixed[opp] = v_real * scale
        term.value_leaves = FixedLeafValues(fixed)
    try:
        v = solver.backward(opp, hot_sigma, hot[seat], reach=hot)
    finally:
        term.value_leaves = saved
    pi = hot[seat, node, combo]
    out = -v[first : first + n].t() / pi.clamp(min=1e-30)  # [C, n]
    ok = blocked_sum(F.one_hot(torch.tensor(combo), NUM_COMBOS).to(v))[:, None] > 0.5
    return torch.where(ok & (pi > 0), out, torch.full_like(out, float("nan")))


class SearchAgent(BaseAgent):
    name = "search"

    def __init__(
        self,
        blueprint: Any,
        config: SearchConfig | dict | str | Path | None = None,
        name: str | None = None,
        value_predictor: Any = None,
        turn_value_predictor: Any = None,
        record: bool = False,
        **overrides: Any,
    ) -> None:
        super().__init__(name)
        self.blueprint = blueprint
        self.record = bool(record)
        self.decisions: list[dict] = []
        if isinstance(config, SearchConfig):
            self.cfg = config
        else:
            self.cfg = search_config(config, **overrides)
        if self.cfg.tree.spec is None:  # tree.actions: blueprint
            self.cfg.tree = replace(self.cfg.tree, spec=blueprint.spec)
        if self.cfg.tree.leaf_mode != self.cfg.leaf.mode:  # leaf.mode decides
            self.cfg.tree = replace(self.cfg.tree, leaf_mode=self.cfg.leaf.mode)
        self.device = self.cfg.torch_device()
        # leaf.mode value_net: a river or turn-end value net (loaded lazily from leaf.net)
        self.value_predictor = value_predictor
        if self.value_net and value_predictor is None and not self.cfg.leaf.net:
            raise ValueError(
                "leaf.mode is value_net but no value_predictor was given and leaf.net "
                "(the value-net checkpoint path) is not set"
            )
        # trees rooted on the turn: a turn-end (or river) net from leaf.turn_net, if any
        self.turn_value_predictor = turn_value_predictor
        self._leaf_nets_checked = False
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
        self.decisions = []  # a new list: a caller may still hold the last hand's

    def observe_end(self, state: Any) -> None:
        self.cache.clear()
        self._roots.clear()
        self._played.clear()

    def policy_agent(self, **kwargs: Any) -> Any:
        """This agent as a range policy for evaluation tools (local best response):
        a :class:`~pokerbot.search.policy.SearchPolicyAgent` that plays through it
        and answers what it plays with every hand, re-searching at decisions it
        never reached. ``kwargs`` go to its constructor."""
        from .policy import SearchPolicyAgent

        return SearchPolicyAgent(self, **kwargs)

    @property
    def value_net(self) -> bool:
        return self.cfg.leaf.mode == "value_net"

    def get_value_predictor(self) -> Any:
        """The value net (``leaf.mode: value_net``), loaded once from ``leaf.net``
        unless one was passed to the constructor: a river
        :class:`~.value_net.ValueNetPredictor` or a turn-end
        :class:`~.turn_net.TurnEndPredictor`, by the checkpoint's ``meta`` kind."""
        if self.value_predictor is None:
            path = self.cfg.leaf.net
            if not path:
                raise ValueError("leaf.mode is value_net but leaf.net is not set")
            from .turn_net import load_leaf_predictor

            self.value_predictor = load_leaf_predictor(path, self.device)
        return self.value_predictor

    def get_turn_value_predictor(self) -> Any:
        """The value net of trees rooted on the turn: the constructor's
        ``turn_value_predictor``, else loaded once from ``leaf.turn_net``;
        ``None`` when neither is set (``leaf.net`` then serves every tree)."""
        if self.turn_value_predictor is None and self.cfg.leaf.turn_net:
            from .turn_net import load_leaf_predictor

            self.turn_value_predictor = load_leaf_predictor(self.cfg.leaf.turn_net, self.device)
        return self.turn_value_predictor

    def _check_leaf_nets(self) -> None:
        """Config errors of the turn-start options, raised before any search
        (so they are not hidden by ``fallback_on_error``): a turn-start
        ``leaf.net`` needs ``depth_streets: 0`` and, when turn solves have
        leaves, a turn-end ``leaf.turn_net``. Other setups are not checked."""
        if self._leaf_nets_checked:
            return
        tc = self.cfg.tree
        main = predictor_kind(self.get_value_predictor())
        turn = self.get_turn_value_predictor()
        if turn is not None and predictor_kind(turn) not in ("turn_end", "river"):
            raise ValueError(
                f"leaf.turn_net must be a turn-end or river net, got {predictor_kind(turn)}"
            )
        if main == "turn_start":
            if int(tc.depth_streets) != 0:
                raise ValueError(
                    "leaf.net is a turn-start net, which values flop-end leaves: set "
                    "tree.depth_streets: 0"
                )
            turn_depth = (
                tc.depth_streets if tc.depth_streets_turn is None else tc.depth_streets_turn
            )
            if int(turn_depth) == 0 and turn is None:
                raise ValueError(
                    "turn solves end at turn-end leaves (tree.depth_streets_turn 0) but "
                    "leaf.turn_net (a turn-end net) is not set; set it, or "
                    "tree.depth_streets_turn: 1 to solve the turn to showdown"
                )
        self._leaf_nets_checked = True

    def value_leaf_provider(self, tree: Any) -> Any:
        """The leaf-value provider of ``tree``'s ``VALUE`` nodes for the value net:
        :class:`~.value_leaf.TurnEndLeafEvaluator` (one net row per leaf) for a
        turn-end net, :class:`~.value_leaf.FlopEndLeafEvaluator` (49 rows per
        leaf) for a turn-start net, else :class:`~.value_leaf.ValueLeafEvaluator`
        (the river net averaged over the 48 river cards). Trees rooted on the
        turn use the turn net (``leaf.turn_net``) when one is set."""
        pred = self.get_turn_value_predictor() if tree.root_street == 2 else None
        if pred is None:
            pred = self.get_value_predictor()
        return make_leaf_evaluator(tree, pred, every=self.cfg.leaf.net_every)

    def act(self, state: Any, seat: int, rng: np.random.Generator) -> Any:
        if state.street == 0:
            return self._blueprint_action(state, seat, rng)
        if self.value_net:
            self.get_value_predictor()  # load errors are config errors: no fallback
            self._check_leaf_nets()
        try:
            return self._search_action(state, seat, rng)
        except Exception:
            if not self.cfg.fallback_on_error:
                raise
            self.last_stats = {"fallback": True, "street": int(state.street)}
            self.stats.append(self.last_stats)
            return self._blueprint_action(state, seat, rng, source="fallback")

    # -- helpers ------------------------------------------------------------

    def _blueprint_action(
        self, state: Any, seat: int, rng: np.random.Generator, source: str = "blueprint"
    ) -> Any:
        t0 = time.perf_counter()
        bp = self.blueprint
        probs = np.asarray(policy_vector(bp, state, seat), dtype=np.float64)
        idx = int(rng.choice(len(probs), p=probs / probs.sum()))
        opts = legal_options(state, bp.spec)
        opt = next(o for o in opts if o.index == idx)
        if self.record:
            self.decisions.append(
                self._blueprint_record(state, seat, opts, opt, source, time.perf_counter() - t0)
            )
        return to_action(self._engine, opt.kind, opt.amount)

    def _blueprint_record(
        self, state: Any, seat: int, opts: list, taken: Any, source: str, seconds: float
    ) -> dict:
        """A decision played from the blueprint (preflop, or a search fallback): its
        policy for every combo over the legal options, and both players' blueprint
        reach along the observed history. No action values."""
        t0 = time.perf_counter()
        bp = self.blueprint
        config = _engine_config(
            self._engine, self.config if self.config is not None else state.config
        )
        P = policy_matrix(bp, state, seat).expand(NUM_COMBOS, -1)
        strategy = P[:, [o.index for o in opts]].float().cpu().numpy()
        board = list(state.board)
        reach = range_reach(bp, config, state.button, board, list(state.history), self._engine)
        return {
            "street": int(state.street),
            "source": source,
            "actions": [(int(o.kind), int(o.amount)) for o in opts],
            "taken": next(j for j, o in enumerate(opts) if o.index == taken.index),
            "strategy": strategy,
            "values": None,
            "values_vs": None,
            "reach": reach.float().cpu().numpy(),
            "seconds": seconds,
            "record_seconds": time.perf_counter() - t0,
            "stats": {},
        }

    def _search_record(
        self,
        solver: RangeSolver,
        tree: Any,
        seat: int,
        strat: torch.Tensor,
        acts: list,
        slot: int,
        seconds: float,
        combo: int,
    ) -> dict:
        """A searched decision: the solved average strategy of every combo at the
        current node, both players' reach there, and ``values[c, a]``, our
        expected chips (net for the hand) for combo ``c`` after action ``a``
        under the solved strategies: the counterfactual value of the child divided
        by the opponent's reach disjoint from ``c``. NaN where that reach is 0.
        ``values_vs[o, a]``: the same for our own ``combo`` against opponent
        combo ``o`` (:func:`pair_values`), for AIVAT's action terms."""
        t0 = time.perf_counter()
        node = tree.current_node
        v, reach = solver.values(seat)
        first, n = int(tree.first_child[node]), int(tree.num_children[node])
        mass = blocked_sum(reach[1 - seat, node])  # [C]
        ev = v[first : first + n].t() / mass[:, None].clamp(min=1e-30)
        ev = torch.where(mass[:, None] > 0, ev, torch.full_like(ev, float("nan")))
        rec = {
            "street": int(tree.root_street),
            "source": "search",
            "actions": [(int(k), int(a)) for k, a in acts],
            "taken": int(slot),
            "strategy": strat.float().cpu().numpy(),
            "values": ev.float().cpu().numpy(),
            "values_vs": pair_values(solver, tree, seat, combo).float().cpu().numpy(),
            "reach": reach[:, node].float().cpu().numpy(),
            "seconds": seconds,
            "record_seconds": time.perf_counter() - t0,
            "stats": dict(self.last_stats),
        }
        del v, reach
        return rec

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
                strat = played_lock(acts, _slot, *played)
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
        # safe resolving: terminate values from the previous street's solve (continual
        # resolving), or computed once per street root as gadget.terminate says
        mode = "off"  # stats: where this decision's terminate values come from
        compute = None
        t_term = 0.0
        if cfg.gadget.safe:
            if info["terminate"] is None and info.get("terminate_source") is None:
                compute = cfg.gadget.terminate
            mode = compute or info.get("terminate_source") or "cache"
            if compute == "unsafe":
                info["terminate_source"] = compute
            if compute == "rollouts":
                t_term = time.perf_counter()
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
                info["terminate_source"] = compute
                t_term = time.perf_counter() - t_term
        t_leaf = time.perf_counter()
        rollouts = None
        value_leaves = None
        if bool((tree.kind == LEAF).any()):
            rollouts = build_leaf_rollouts(
                tree, self.blueprint, config, cfg.leaf, engine, self.device
            )
        if bool((tree.kind == VALUE).any()):
            value_leaves = self.value_leaf_provider(tree)
        t_leaf = time.perf_counter() - t_leaf
        locks = self._locks(tree, key, seat)
        solver = RangeSolver(
            tree, info["ranges"], cfg.solver, rollouts, None, locks, value_leaves=value_leaves
        )
        if compute in ("blueprint", "blueprint_br"):
            # terminate values in this tree's own game: the opponent against the blueprint
            t0_term = time.perf_counter()
            sigma_bp = blueprint_profile(solver, self.blueprint, config)[0]
            info["terminate"] = tree_terminate_values(
                solver, sigma_bp, 1 - seat, best_response=compute == "blueprint_br"
            )
            info["terminate_source"] = compute
            del sigma_bp
            t_term = time.perf_counter() - t0_term
        if info["terminate"] is not None and cfg.gadget.safe:  # "unsafe": no gadget here
            prior = mixed_prior(
                info["ranges"][1 - seat], valid_mask(board, self.device), cfg.gadget.prior_mix
            )
            if cfg.remove_own_blockers:
                prior = prior * valid_mask(state.hole_cards(seat), self.device)
            solver.set_gadget(Gadget(1 - seat, prior, info["terminate"]))
        t_setup = time.perf_counter()
        cap = cfg.max_iterations(street)
        solver.solve(iterations=min(cfg.min_iterations, cap))
        left = cfg.budget(street) - (time.perf_counter() - t0)
        remaining_iters = cap - solver.iterations_done
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
        net_rows = 0 if value_leaves is None else value_leaves.net_rows
        t_cache = time.perf_counter()
        stored = self.cache.store(solver, tree, seat, prefix)
        t_end = time.perf_counter()
        self.last_stats = {
            "street": street,
            "nodes": tree.num_nodes,
            "iterations": solver.iterations_done,
            "tree_seconds": t_tree - t0,
            "setup_seconds": t_setup - t_tree,
            "leaf_seconds": t_leaf,
            "value_leaves": 0 if value_leaves is None else value_leaves.num_leaves,
            "value_provider": None if value_leaves is None else type(value_leaves).__name__,
            "value_net_rows": 0 if value_leaves is None else value_leaves.net_rows,
            "solve_seconds": solver.solve_time,
            "total_seconds": t_end - t0,
            "cached_root": info["cached"],
            "cache_entries": stored,
            # the cache store's share of value_net_rows, and its time
            "cache_net_rows": 0 if value_leaves is None else value_leaves.net_rows - net_rows,
            "cache_seconds": t_end - t_cache,
            # leaves under our action whose next-street roots were not stored (turn-end net)
            "cache_skipped_leaves": self.cache.skipped_leaves,
            "gadget": mode,
            "terminate_seconds": t_term,
        }
        self.stats.append(self.last_stats)
        if self.record:
            self.decisions.append(
                self._search_record(
                    solver, tree, seat, strat, acts, slot, t_end - t0, combo_index(hole[0], hole[1])
                )
            )
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
