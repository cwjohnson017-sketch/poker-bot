"""``BlueprintAgent``: plays an exported MCCFR strategy.

The agent keeps two shadow states of the current hand, both
``poker_engine.GameState`` built on a dummy deck (cards never matter for
betting):

* ``real``: the actual betting, replayed action by action from the history
  of the state the agent is shown;
* ``abs``: the same hand in the abstract game the blueprint was trained on
  (the training stacks and blinds), where every action is one of the
  abstract actions.

Opponent actions are mapped into ``abs`` with the pseudo-harmonic mapping
(randomized by default, Ganzfried & Sandholm 2013) computed from the pot
fraction of the bet in ``real``; the agent's own actions are the abstract
actions it chose. At a decision the agent looks up the infoset of ``abs``
(street, its bucket from the real hole cards and board, abstract betting
sequence), samples an abstract action, and converts it to a concrete action
sized against the real pot. When ``abs`` no longer tracks the hand (the
opponent raised past the abstraction's raise cap, or a bet mapped to an
all-in that was not one), the agent falls back to check/call.

The agent is also a :class:`~pokerbot.agents.policy.PolicyAgent`:
``policy(state, seat)`` and the batched ``policy_batch(state, seat, holes)``
give the blueprint's probabilities over the indices of ``agent.spec`` (the
training action list) from the public state alone, through
:class:`~.policy.TabularPolicy`. That replay maps every action with the
deterministic pseudo-harmonic split (``u = 0.5``), so after an off-tree
opponent size the policy is the one of the more likely mapping, while
``act`` with ``mapping="randomized"`` samples the mapping per hand.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from ...agents.base import BaseAgent
from .policy import TabularPolicy


class BlueprintAgent(BaseAgent):
    name = "blueprint"

    def __init__(
        self,
        path: str | Path,
        name: str | None = None,
        mapping: str = "randomized",
        greedy: bool = False,
    ) -> None:
        super().__init__(name)
        import poker_engine as pe

        if mapping not in ("randomized", "deterministic"):
            raise ValueError("mapping must be 'randomized' or 'deterministic'")
        self._pe = pe
        self.path = str(path)
        self.strategy = pe.BlueprintStrategy(self.path)
        self.abstraction = self.strategy.action_abstraction()
        self.solver_config = json.loads(self.strategy.config_json)
        # ``game:`` section of the training game (used by play_match when the
        # match config does not name one).
        self.default_game = dict(self.solver_config["game"])
        self.mapping = mapping
        self.greedy = greedy
        self.counters: Counter[str] = Counter()
        self.table = TabularPolicy(self.strategy)
        self.spec = self.table.spec
        self._reset()

    # -- hand bookkeeping ---------------------------------------------------

    def _reset(self) -> None:
        self._real = None
        self._abs = None
        self._seen = 0
        self._pending: list[int] = []
        self._off_tree = False

    def new_hand(self, seat: int, config: Any) -> None:
        super().new_hand(seat, config)
        self._reset()

    def _start(self, state: Any) -> None:
        pe = self._pe
        cfg = getattr(state, "config", None) or self.config
        real_cfg = pe.GameConfig(
            num_players=int(cfg.num_players),
            stacks=[int(s) for s in cfg.stacks],
            small_blind=int(cfg.small_blind),
            big_blind=int(cfg.big_blind),
            ante=int(cfg.ante),
        )
        deck = list(range(52))
        button = int(state.button)
        self._real = pe.GameState.new_hand(real_cfg, button, deck)
        self._abs = pe.GameState.new_hand(self.strategy.game_config, button, deck)

    def _concrete(self, action: Any) -> Any:
        return self._pe.action_from(int(action.kind), int(action.amount))

    def _sync(self, state: Any, seat: int, rng: np.random.Generator) -> None:
        history = state.history
        for _street, player, action in history[self._seen :]:
            a = self._concrete(action)
            if not self._off_tree:
                idx = None
                if player == seat and self._pending:
                    idx = self._pending.pop(0)
                else:
                    u = float(rng.random()) if self.mapping == "randomized" else 0.5
                    idx = self.abstraction.translate(self._abs, self._real, a, u)
                    if idx is not None and player != seat:
                        legal = dict(self.abstraction.legal(self._real))
                        if idx not in legal or legal[idx] != a:
                            self.counters["translated"] += 1
                abs_a = None if idx is None else self.abstraction.to_concrete(self._abs, idx)
                if abs_a is None:
                    self._off_tree = True
                    self.counters["off_tree"] += 1
                else:
                    self._abs.apply(abs_a)
            self._real.apply(a)
        self._seen = len(history)

    # -- PolicyAgent ----------------------------------------------------------

    def policy(self, state: Any, seat: int) -> np.ndarray:
        """``[A]`` probabilities over ``self.spec`` for ``seat`` holding
        ``state.hole_cards(seat)`` (stateless; see the module docstring)."""
        return self.table.probs(state, seat)

    def policy_batch(self, state: Any, seat: int, holes: Any) -> np.ndarray:
        """``[K, A]`` probabilities for ``K`` hypothetical hole-card pairs."""
        return self.table.probs_batch(state, seat, holes)

    # -- decisions ----------------------------------------------------------

    def act(self, state: Any, seat: int, rng: np.random.Generator) -> Any:
        if self._real is None:
            self._start(state)
        self._sync(state, seat, rng)
        self.counters["decisions"] += 1
        abs_s = self._abs
        tracked = (
            not self._off_tree
            and not abs_s.is_terminal
            and abs_s.current_player == seat
            and abs_s.street == state.street
        )
        if tracked:
            hole = [int(c) for c in state.hole_cards(seat)]
            board = [int(c) for c in state.board]
            idxs, _acts, probs, found = self.strategy.action_probs(abs_s, hole, board)
            if not found:
                self.counters["missing_infoset"] += 1
            p = np.asarray(probs, dtype=np.float64)
            p = p / p.sum()
            k = int(np.argmax(p)) if self.greedy else int(rng.choice(len(p), p=p))
            idx = int(idxs[k])
            concrete = self.abstraction.to_concrete(self._real, idx)
            if concrete is not None:
                self._pending.append(idx)
                return concrete
            # The abstract action does not exist in the real state (e.g. no
            # raise left); play the closest legal action and let _sync map it.
            self.counters["unavailable"] += 1
        else:
            self.counters["fallback"] += 1
        return self._pe.Action.check_call()
