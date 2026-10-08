"""Search :class:`~pokerbot.search.blueprint.Blueprint` adapters for the trained blueprints.

* :class:`TabularBlueprint` wraps an exported MCCFR strategy
  (``poker_engine.BlueprintStrategy``) through
  :class:`~pokerbot.blueprint.mccfr.policy.TabularPolicy`. ``policy_combos``
  buckets all 1326 combos of the board with one
  ``CardAbstraction.buckets_batch`` call (cached per board) and looks up each
  distinct bucket's infoset once.
* :class:`NeuralBlueprint` wraps the Deep CFR ``NeuralBlueprintAgent`` (one
  ``SDCFRPolicy`` per seat) through
  :class:`~pokerbot.blueprint.deepcfr.range_policy.NeuralRangePolicy`.
  ``policy_combos`` encodes the public state once, writes every combo's hole
  cards into the batch and runs one forward pass per net (plus one per
  earlier own decision for the reach weights, cached per history prefix).

Both use the blueprint's own action list as ``spec``, so the search maps the
observed betting onto it and plays rollouts with it. Rows of combos that
share a card with the board are zero. Registered as ``search:blueprint:<path>``
and ``search:neural:<dir>`` (see :mod:`pokerbot.search.blueprint`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch


class TabularBlueprint:
    """Search blueprint over an exported MCCFR strategy file."""

    card_independent = False

    def __init__(self, source: Any, mapping_u: float = 0.5) -> None:
        from ..blueprint.mccfr.policy import TabularPolicy

        if isinstance(source, TabularPolicy):
            self.table = source
        elif hasattr(source, "table") and isinstance(source.table, TabularPolicy):
            self.table = source.table  # a BlueprintAgent
        else:
            self.table = TabularPolicy.load(str(source), mapping_u=mapping_u)
        self.spec = self.table.spec
        self.game = dict(self.table.solver_config["game"])

    def policy(self, state: Any, player: int) -> np.ndarray:
        return self.table.probs(state, player)

    def policy_combos(self, state: Any, player: int) -> torch.Tensor:
        return torch.from_numpy(self.table.probs_batch(state, player)).float()


class NeuralBlueprint:
    """Search blueprint over a Deep CFR checkpoint directory (SD-CFR average)."""

    card_independent = False

    def __init__(self, agent: Any) -> None:
        self.agent = agent
        self.spec = agent.spec
        self.game = agent.trained_game

    @classmethod
    def from_dir(cls, path: str | Path, **kwargs: Any) -> NeuralBlueprint:
        from ..blueprint.deepcfr.agent import NeuralBlueprintAgent

        return cls(NeuralBlueprintAgent.from_dir(path, **kwargs))

    def policy(self, state: Any, player: int) -> np.ndarray:
        return self.agent.policy(state, player)

    def policy_combos(self, state: Any, player: int) -> torch.Tensor:
        return self.agent.policy_all(state, player)

    # incremental own reach for rollouts (see leaf._rollout)
    def log_reach_combos(self, state: Any, player: int) -> torch.Tensor | None:
        rp = self.agent.range_policy
        return rp.log_reach_all(state, player, self.agent._query_config(state))

    def policy_combos_nets(
        self, state: Any, player: int, log_reach: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rp = self.agent.range_policy
        return rp.probs_all_nets(state, player, self.agent._query_config(state), log_reach)


def tabular_blueprint(arg: str, **kwargs: Any) -> TabularBlueprint:
    if not arg:
        raise ValueError("search:blueprint needs a strategy file: search:blueprint:<path>")
    return TabularBlueprint(arg, **kwargs)


def neural_blueprint(arg: str, **kwargs: Any) -> NeuralBlueprint:
    if not arg:
        raise ValueError("search:neural needs a checkpoint directory: search:neural:<dir>")
    return NeuralBlueprint.from_dir(arg, **kwargs)
