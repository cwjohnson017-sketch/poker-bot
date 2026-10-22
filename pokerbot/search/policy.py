"""The search as a range policy: what ``SearchAgent`` plays with every hand.

Evaluation tools that track the bot's range (local best response,
:mod:`pokerbot.eval.lbr`) need the probability of every action for each of
the 1,326 hands the bot could hold, at the decisions it really made and at
decisions it never reached (after a raise the evaluator is only considering).
The search solves for its whole range at every decision, so both exist:

* **Real decisions.** :class:`SearchPolicyAgent` plays the hand through the
  wrapped agent with ``record=True`` and keeps each decision's full-range
  strategy (``SearchAgent.decisions[k]["strategy"]``, ``[1326, n]`` over the
  concrete actions ``decisions[k]["actions"]``) under the decision's public
  state: the strategy the bot really played.
* **Hypothetical decisions.** A query at a public state the bot has not
  played runs the search there, on a *fork* of the agent: a shallow copy with
  its own copies of the per-hand containers (the continual-resolving cache
  ``cache``, the street roots ``_roots``, the played strategies ``_played``),
  so the fork resolves exactly as the real agent would on reaching that state
  (same cached ranges and gadget values from its earlier streets, same locks)
  and the real hand's caches never see it. The nets, the blueprint and the
  config are shared. Results are cached by public state for the hand.

The fork plays with the bot's real hole cards (remembered from its last
``act``) and never sees the evaluator's: with ``remove_own_blockers`` the
search's view of the opponent's range depends on its own cards, and the
fork does what the real bot would do. Preflop the agent plays from its
blueprint (``search:preflop:<table>``: the solved table on the tree, its base
blueprint through the history off it), so preflop "searches" are table lookups.

The strategy rows of hands other than the bot's actual one come from a solve
that removed the bot's actual cards from its model of the opponent's range
(``remove_own_blockers``); they are what the search computed, not what it
would compute holding that hand. The difference is a few blocker combos.

``range_strategy(state, seat)`` is the interface: ``(actions, strategy)``
with ``actions`` the ``(kind, amount)`` children and ``strategy`` a
``[1326, len(actions)]`` array, rows summing to one. ``policy`` and
``policy_batch`` map it onto ``spec`` (the blueprint's action spec) with
:func:`~pokerbot.agents.policy.nearest_abstract_index`, for tools that work in
abstract indices.
"""

from __future__ import annotations

import copy
import time
from dataclasses import replace
from typing import Any

import numpy as np

from ..agents.base import BaseAgent
from ..agents.policy import RAISE, hypothetical_state, legal_vector, nearest_abstract_index
from .combos import NUM_COMBOS, combo_index
from .gadget import ContinualCache, history_key


class _Act:
    """A concrete action as ``nearest_abstract_index`` reads it."""

    __slots__ = ("amount", "kind")

    def __init__(self, kind: int, amount: int) -> None:
        self.kind = kind
        self.amount = amount


def public_key(state: Any) -> tuple:
    """Hashable public state: button, board and betting history."""
    return (int(state.button), tuple(int(c) for c in state.board), history_key(state.history))


def normalise_strategy(strategy: Any) -> np.ndarray:
    """``[1326, n]`` float64 rows summing to one (uniform where a row has no
    finite positive mass, e.g. hands that conflict with the board)."""
    s = np.nan_to_num(np.asarray(strategy, dtype=np.float64), nan=0.0, posinf=0.0)
    s = np.clip(s, 0.0, None)
    tot = s.sum(1, keepdims=True)
    n = s.shape[1]
    return np.where(tot > 0, s / np.where(tot > 0, tot, 1.0), 1.0 / max(1, n))


class SearchPolicyAgent(BaseAgent):
    """Plays as ``agent`` (a :class:`~pokerbot.search.agent.SearchAgent`) and
    answers range queries for it (see the module docstring).

    ``hypothetical_budget`` scales the time budget of hypothetical searches
    (1.0: the agent's own budget, the faithful setting). ``hypo_stats`` keeps
    one entry per hypothetical search (street, seconds) for cost accounting."""

    def __init__(self, agent: Any, hypothetical_budget: float = 1.0) -> None:
        super().__init__(getattr(agent, "name", "search"))
        self.agent = agent
        agent.record = True
        self.spec = agent.blueprint.spec
        self.hypothetical_budget = float(hypothetical_budget)
        self._hypo_cfg = agent.cfg
        if self.hypothetical_budget != 1.0:
            tb = {k: v * self.hypothetical_budget for k, v in agent.cfg.time_budget.items()}
            self._hypo_cfg = replace(agent.cfg, time_budget=tb)
        self._real: dict[tuple, tuple[list, np.ndarray]] = {}
        self._hypo: dict[tuple, tuple[list, np.ndarray]] = {}
        self._hole: list[int] | None = None
        self.hypo_stats: list[dict] = []
        self.real_stats: list[dict] = []

    # -- Agent protocol -----------------------------------------------------

    def new_hand(self, seat: int, config: Any) -> None:
        super().new_hand(seat, config)
        self.agent.new_hand(seat, config)
        self._real.clear()
        self._hypo.clear()
        self._hole = None

    def observe_end(self, state: Any) -> None:
        self.agent.observe_end(state)

    def act(self, state: Any, seat: int, rng: np.random.Generator) -> Any:
        self._hole = [int(c) for c in state.hole_cards(seat)]
        key = public_key(state)
        t0 = time.perf_counter()
        n = len(self.agent.decisions)
        action = self.agent.act(state, seat, rng)
        rec = self.agent.decisions[n]
        self._real[key] = ([tuple(a) for a in rec["actions"]], normalise_strategy(rec["strategy"]))
        self.real_stats.append(
            {
                "street": int(state.street),
                "source": rec["source"],
                "seconds": time.perf_counter() - t0,
            }
        )
        return action

    def cost(self) -> dict[str, list[float]]:
        """``{"<real|hypothetical>/<street>": [decisions, seconds]}`` since
        construction (preflop decisions are blueprint lookups), plus
        ``"<real|hypothetical>/fallback"`` for postflop searches that failed and
        played the blueprint (``fallback_on_error``)."""
        names = ("preflop", "flop", "turn", "river")
        out: dict[str, list[float]] = {}
        for kind, stats in (("real", self.real_stats), ("hypothetical", self.hypo_stats)):
            for st in stats:
                keys = [f"{kind}/{names[min(st['street'], 3)]}"]
                if st["source"] == "fallback":
                    keys.append(f"{kind}/fallback")
                for key in keys:
                    c = out.setdefault(key, [0, 0.0])
                    c[0] += 1
                    c[1] += st["seconds"]
        return out

    # -- range queries ------------------------------------------------------

    def range_strategy(self, state: Any, seat: int) -> tuple[list[tuple[int, int]], np.ndarray]:
        """``(actions, strategy)`` of the bot in ``seat`` (the player to act at
        ``state``, whose public information alone is read): the recorded
        strategy if the bot really decided here this hand, else a hypothetical
        search (cached)."""
        if int(state.current_player) != int(seat):
            raise ValueError(f"seat {seat} is not to act at this decision point")
        key = public_key(state)
        hit = self._real.get(key) or self._hypo.get(key)
        if hit is None:
            hit = self._hypothetical(state, seat)
            self._hypo[key] = hit
        return hit

    def _fork(self) -> Any:
        """A copy of the agent that shares its nets, blueprint and config but has
        its own per-hand containers (the real agent's contents at this moment)."""
        a = self.agent
        if a.value_net:  # load the nets once, on the real agent, so forks share them
            a.get_value_predictor()
            a.get_turn_value_predictor()
        f = copy.copy(a)
        f.cfg = self._hypo_cfg
        f.cache = ContinualCache(dict(a.cache.entries), a.cache.skipped_leaves)
        f._roots = {k: dict(v) for k, v in a._roots.items()}
        f._played = dict(a._played)
        f.decisions = []
        f.stats = []
        f.record = True
        return f

    def _hypothetical(self, state: Any, seat: int) -> tuple[list[tuple[int, int]], np.ndarray]:
        from ..eval.masking import MaskedState

        config = self.config if self.config is not None else state.config
        board = [int(c) for c in state.board]
        hole = self._hole
        if hole is None or set(hole) & set(board):
            hole = [c for c in range(52) if c not in board][:2]  # preflop: cards unused
        full = hypothetical_state(state, seat, hole, config, None, self._engine)
        view = MaskedState(full, seat, config, np.random.default_rng(0))
        fork = self._fork()
        t0 = time.perf_counter()
        fork.act(view, seat, np.random.default_rng(0))
        rec = fork.decisions[-1]
        self.hypo_stats.append(
            {
                "street": int(state.street),
                "source": rec["source"],
                "seconds": time.perf_counter() - t0,
            }
        )
        return [tuple(a) for a in rec["actions"]], normalise_strategy(rec["strategy"])

    # -- abstract-index views (PolicyAgent) -----------------------------------

    def policy_batch(self, state: Any, seat: int, holes: np.ndarray) -> np.ndarray:
        """``[K, A]`` over ``spec`` indices for the hands ``holes`` (``[K, 2]``);
        each concrete action's mass goes to its nearest abstract index."""
        acts, strat = self.range_strategy(state, seat)
        A = self.spec.num_actions
        M = np.zeros((len(acts), A))
        for j, (k, amt) in enumerate(acts):
            M[j, nearest_abstract_index(state, _Act(k, amt), self.spec)] = 1.0
        holes = np.asarray(holes, dtype=np.int64).reshape(-1, 2)
        rows = np.array([combo_index(int(a), int(b)) for a, b in holes], dtype=np.int64)
        out = strat[rows] @ M
        return out * legal_vector(state, self.spec)[None, :]

    def policy(self, state: Any, seat: int) -> np.ndarray:
        return self.policy_batch(state, seat, np.array([state.hole_cards(seat)]))[0]


def action_probs(
    actions: list[tuple[int, int]], strategy: np.ndarray, kind: int, amount: int = 0
) -> np.ndarray:
    """``[1326]`` probability of the concrete action ``(kind, amount)`` under a
    ``range_strategy`` result: its own column when present; otherwise (a size
    the strategy lacks) the column of the nearest raise size, or zeros."""
    kind = int(kind)
    amount = int(amount) if kind == RAISE else 0
    if (kind, amount) in actions:
        return strategy[:, actions.index((kind, amount))]
    if kind == RAISE:
        raises = [j for j, (k, _a) in enumerate(actions) if k == RAISE]
        if raises:
            j = min(raises, key=lambda j: abs(actions[j][1] - amount))
            return strategy[:, j]
    return np.zeros(NUM_COMBOS)
