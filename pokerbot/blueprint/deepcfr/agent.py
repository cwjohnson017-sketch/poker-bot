"""``NeuralBlueprintAgent``: plays the SD-CFR average strategy through the scalar engine.

At each decision the agent encodes its view of the ``GameState`` (either
engine, or the match runner's masked view) with the scalar mirror of the env
encoder (:mod:`.scalar`), asks its seat's :class:`~.policy.SDCFRPolicy` for
the average policy over the abstract actions, samples one (or takes the most
likely with ``greedy``), and converts it to a concrete action with the env's
sizing rule.

Opponent raises off the abstraction are mapped to an abstract index when the
history is encoded. With ``offtree="harmonic"`` (the default) that is the
pseudo-harmonic mapping of :func:`pokerbot.abstraction.actions.map_offtree`,
randomized with the ``rng`` passed to ``act`` and drawn once per opponent
action per hand (kept until ``new_hand``). ``offtree="nearest"`` records the
nearest legal size, exactly the index ``VecNLHE.step_concrete`` records.

Registered in the agent factory as ``neural:<checkpoint dir>``.

The agent is also a :class:`~pokerbot.agents.policy.PolicyAgent`:
``policy(state, seat)`` and ``policy_batch(state, seat, holes)`` return the
same average policy as a function of the public state and a hand (own reach
recomputed from the history, :class:`~.range_policy.NeuralRangePolicy`; with
``offtree="harmonic"`` it maps off-tree opponent raises deterministically,
``u = 0.5``, as the tabular policy does), and
``vec_policy(device)`` returns a :class:`~.vec_policy.NeuralVecPolicy` that
acts on a ``VecNLHE`` batch directly (for the approximate best response).
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ...agents.base import BaseAgent
from ...env.actions import CHECK_CALL, FOLD, ActionSpec
from .checkpoint import read_meta
from .config import spec_from_dict
from .features import FeatureConfig
from .policy import SDCFRPolicy
from .range_policy import NeuralRangePolicy
from .scalar import ScalarSpec, check_offtree, encode_state


class NeuralBlueprintAgent(BaseAgent):
    name = "neural"

    def __init__(
        self,
        policies: list[SDCFRPolicy],
        spec: ActionSpec,
        features: FeatureConfig | None = None,
        greedy: bool = False,
        name: str | None = None,
        offtree: str = "harmonic",
    ) -> None:
        super().__init__(name)
        if len(policies) != 2:
            raise ValueError("need one SDCFRPolicy per seat")
        self.policies = policies
        self.spec = spec
        self.sp = ScalarSpec.build(spec)
        self.features = features or FeatureConfig()
        self.greedy = greedy
        self.offtree = check_offtree(offtree)
        self.trained_game: dict[str, Any] | None = None
        self._warned = False
        self.last_probs: np.ndarray | None = None
        # opponent off-tree raise -> harmonic index drawn this hand (per seat)
        self._offtree_memo: dict[tuple, int] = {}
        self.offtree_mapped = 0  # opponent raises mapped with the harmonic rule
        self.range_policy = NeuralRangePolicy(
            policies, spec, self.features, self.sp, offtree=self.offtree
        )

    @classmethod
    def from_checkpoint(cls, path: str | Path, **kwargs: Any) -> NeuralBlueprintAgent:
        """Registry entry point (``neural:<run dir>``); same as :meth:`from_dir`."""
        return cls.from_dir(path, **kwargs)

    @classmethod
    def from_dir(
        cls,
        path: str | Path,
        last_n: int | None = None,
        max_iter: int | None = None,
        device: str = "cpu",
        reach_weighted: bool = True,
        greedy: bool = False,
        name: str | None = None,
        offtree: str = "harmonic",
    ) -> NeuralBlueprintAgent:
        meta = read_meta(path)
        kw = dict(
            last_n=last_n,
            max_iter=max_iter,
            device=device,
            reach_weighted=reach_weighted,
            fallback=meta.get("fallback", "uniform"),
        )
        pols = [SDCFRPolicy.from_dir(path, p, **kw) for p in (0, 1)]
        agent = cls(
            pols,
            spec_from_dict(meta.get("spec")),
            FeatureConfig.from_dict(meta.get("features")),
            greedy=greedy,
            name=name,
            offtree=offtree,
        )
        agent.trained_game = meta.get("game")
        return agent

    def new_hand(self, seat: int, config: Any) -> None:
        super().new_hand(seat, config)
        self.policies[seat].new_hand()
        for k in [k for k in self._offtree_memo if k[0] == seat]:
            del self._offtree_memo[k]
        g = self.trained_game
        if g and not self._warned and config is not None:
            now = (list(config.stacks), config.small_blind, config.big_blind)
            if now != (list(g["stacks"]), g["small_blind"], g["big_blind"]):
                warnings.warn(
                    f"{self.name}: trained with stacks {g['stacks']} and blinds "
                    f"{g['small_blind']}/{g['big_blind']}, playing {now}",
                    stacklevel=2,
                )
                self._warned = True

    # -- PolicyAgent ----------------------------------------------------------

    def _query_config(self, state: Any) -> Any:
        cfg = getattr(state, "config", None)
        return cfg if cfg is not None else self.config

    def policy(self, state: Any, seat: int) -> np.ndarray:
        """``[A]`` average policy over ``self.spec`` for ``seat`` holding
        ``state.hole_cards(seat)`` (stateless: own reach from the history)."""
        return self.range_policy.probs(state, seat, self._query_config(state))

    def policy_batch(self, state: Any, seat: int, holes: Any) -> np.ndarray:
        """``[K, A]`` average policy for ``K`` hypothetical hole-card pairs."""
        return self.range_policy.probs_batch(state, seat, holes, self._query_config(state))

    def policy_all(self, state: Any, seat: int) -> torch.Tensor:
        """``[1326, A]`` for every combo (canonical order), zero rows for
        combos that share a card with the board."""
        return self.range_policy.probs_all(state, seat, self._query_config(state))

    def vec_policy(self, device: torch.device | str = "cpu", seed: int = 0) -> Any:
        """A ``VecPolicy`` (see :mod:`pokerbot.eval.abr`) playing this blueprint
        on a ``VecNLHE`` built with ``spec=self.spec``."""
        from .vec_policy import NeuralVecPolicy

        return NeuralVecPolicy(
            self.policies, self.spec, self.features, device, seed, self.greedy, self.name
        )

    def act(self, state: Any, seat: int, rng: np.random.Generator) -> Any:
        config = self.config if self.config is not None else state.config
        gen = None
        if self.features.equity_samples > 0 or self.features.hist_runouts > 0:
            gen = torch.Generator().manual_seed(int(rng.integers(2**62)))
        n_memo = len(self._offtree_memo)
        feats, info = encode_state(
            state,
            seat,
            config,
            self.spec,
            self.features,
            self.features.history_len,
            generator=gen,
            sp=self.sp,
            offtree=self.offtree,
            rng=rng,
            memo=self._offtree_memo,
        )
        self.offtree_mapped += len(self._offtree_memo) - n_memo
        pol = self.policies[seat]
        probs = pol.act_probs(feats)[0].double().cpu().numpy()
        probs = np.where(np.asarray(info.legal), np.clip(probs, 0.0, None), 0.0)
        probs = probs / probs.sum()
        self.last_probs = probs
        a = int(np.argmax(probs)) if self.greedy else int(rng.choice(len(probs), p=probs))
        pol.observe(a)
        kind = self.sp.concrete_kind(info.street, a)
        if kind == FOLD:
            return self.fold()
        if kind == CHECK_CALL:
            return self.check_call()
        return self.raise_to(int(info.targets[a]))
