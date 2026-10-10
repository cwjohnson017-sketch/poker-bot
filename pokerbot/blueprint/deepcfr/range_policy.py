"""Stateless SD-CFR average policy for one hand or for every hand at once.

:class:`~.policy.SDCFRPolicy` tracks the player's own reach through a hand
as the agent acts. Tools that query the blueprint for hands it is not
holding (the search's ranges and rollouts, LBR, the match runner's masked
views) need the same average policy as a function of the public state and a
hypothetical hand. :class:`NeuralRangePolicy` recomputes it:

* **Features.** The public part of the features (history tokens, amounts,
  pot and stack fractions, legal mask) does not depend on the hand, so the
  batched path encodes the decision once with a placeholder hand
  (:func:`~.scalar.encode_state`), repeats the row and writes each hand's
  two cards into ``cards[:, :2]``. Optional equity features are recomputed
  for the batch (Monte Carlo, with a fixed seed per call).
* **Off-tree opponent raises.** With ``offtree="harmonic"`` the history
  encoding maps them with the deterministic pseudo-harmonic split
  (``u = 0.5``, :func:`~.scalar.harmonic_abstract`), so the encoding stays a
  function of the public state and the per-prefix cache stays valid; with
  ``"nearest"`` it records the nearest legal size (the env's rule).
* **Own reach.** With ``reach_weighted`` the SD-CFR average weights net
  ``t`` by ``t`` times the hand's own reach under net ``t``: the product,
  over the player's earlier decisions in the history, of that net's
  probability of the action taken (mapped to the abstract index the env
  records, :func:`~.scalar.nearest_abstract`). The batched path evaluates
  every net once per earlier own decision for all 1326 hands and caches the
  cumulative log reach per (history prefix, board prefix), so later queries
  on the same betting line reuse it.

The single-hand path, :meth:`NeuralRangePolicy.probs`, encodes the actual
state (and each earlier decision) exactly as
:class:`~.agent.NeuralBlueprintAgent` does when it acts, so the two paths check
each other and :meth:`probs` equals the agent's own distribution (with
``offtree="harmonic"``, whenever the agent's randomized mapping of each
off-tree opponent raise fell on the ``u = 0.5`` side).
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from itertools import combinations
from typing import Any

import numpy as np
import torch

from ...engine_select import to_engine_action
from ...env.actions import ActionSpec
from ...env.cards import NO_CARD
from ...env.obs import NUM_SCALARS
from .features import FeatureConfig
from .policy import SDCFRPolicy
from .scalar import (
    ScalarSpec,
    check_offtree,
    encode_state,
    engine_config,
    engine_for,
    nearest_abstract,
)
from .strength import add_strength, load_strength

RAISE = 2
BOARD_LEN = (0, 3, 4, 5)
NUM_COMBOS = 1326
ALL_COMBOS = torch.tensor(list(combinations(range(52), 2)), dtype=torch.long)  # [1326, 2]


def combo_rows(holes: np.ndarray) -> np.ndarray:
    """Canonical combo index (``itertools.combinations`` order) of ``[K, 2]`` hands."""
    a = np.minimum(holes[:, 0], holes[:, 1])
    b = np.maximum(holes[:, 0], holes[:, 1])
    return a * (103 - a) // 2 + (b - a - 1)


def board_conflicts(board: Sequence[int]) -> torch.Tensor:
    """``[1326]`` bool: combos sharing a card with ``board``."""
    cards = torch.tensor([int(c) for c in board], dtype=torch.long)
    return torch.isin(ALL_COMBOS, cards).any(1)


class NeuralRangePolicy:
    def __init__(
        self,
        policies: Sequence[SDCFRPolicy],
        spec: ActionSpec,
        features: FeatureConfig | None = None,
        sp: ScalarSpec | None = None,
        seed: int = 0,
        cache_bytes: float = 64e6,
        offtree: str = "harmonic",
    ) -> None:
        self.policies = list(policies)
        self.spec = spec
        self.sp = sp or ScalarSpec.build(spec)
        self.features = features or FeatureConfig()
        self.seed = int(seed)
        self.offtree = check_offtree(offtree)
        T = max(len(p) for p in self.policies)
        self.cache_entries = max(8, int(cache_bytes // (T * NUM_COMBOS * 8)))
        self._reach: OrderedDict[tuple, torch.Tensor] = OrderedDict()

    # ------------------------------------------------------------ helpers
    @property
    def _equity(self) -> bool:
        return self.features.equity_samples > 0 or self.features.hist_runouts > 0

    def _gen(self) -> torch.Generator | None:
        return torch.Generator().manual_seed(self.seed) if self._equity else None

    def _encode(self, state: Any, seat: int, config: Any) -> tuple[dict, Any]:
        return encode_state(
            state,
            seat,
            config,
            self.spec,
            self.features,
            self.features.history_len,
            generator=self._gen(),
            sp=self.sp,
            offtree=self.offtree,
        )

    @staticmethod
    def _state(
        config: Any,
        button: int,
        board: Sequence[int],
        history: Sequence,
        seat: int,
        hole: Sequence[int],
    ) -> Any:
        """Engine state after ``history`` with ``seat`` holding ``hole`` and
        ``board`` dealt as far as the history goes (other cards arbitrary)."""
        engine = engine_for(config)
        cfg = engine_config(engine, config)
        board = [int(c) for c in board]
        hole = [int(c) for c in hole]
        used = set(hole) | set(board)
        rest = iter(c for c in range(52) if c not in used)
        deck: list[int] = [0] * 9
        deck[2 * seat : 2 * seat + 2] = hole
        deck[2 * (1 - seat) : 2 * (1 - seat) + 2] = [next(rest), next(rest)]
        for j in range(5):
            deck[4 + j] = board[j] if j < len(board) else next(rest)
        deck += list(rest)
        s = engine.GameState.new_hand(cfg, int(button), deck)
        for _st, _p, a in history:
            s.apply(to_engine_action(engine, a))
        return s

    @staticmethod
    def _placeholder(board: Sequence[int]) -> list[int]:
        free = [c for c in range(52) if c not in set(int(x) for x in board)]
        return free[:2]

    def _batch(self, state: Any, seat: int, config: Any, holes: torch.Tensor) -> tuple[dict, Any]:
        """Features ``[K, ...]`` of ``holes`` at the decision ``state`` (an
        engine state in which ``seat`` holds a placeholder hand)."""
        feats, info = self._encode(state, seat, config)
        K = holes.shape[0]
        out = {k: v.expand(K, *v.shape[1:]).clone() for k, v in feats.items()}
        out["cards"][:, :2] = holes
        f = self.features
        if self._equity or f.strength_tables:
            # the hand-dependent columns are recomputed for every hand. Hands
            # sharing a card with the board (zero rows in probs_all) get a valid
            # placeholder hand for this, since the lookups reject duplicate cards.
            b = out["cards"][:, 2:]
            clash = (holes[:, :, None] == b[:, None, :]).any(2).any(1)
            if bool(clash.any()):
                board = [int(c) for c in b[0].tolist() if int(c) != NO_CARD]
                holes = holes.clone()
                holes[clash] = torch.tensor(self._placeholder(board), dtype=holes.dtype)
                out["cards"][:, :2] = holes
            extra = [out["scalars"][:, :NUM_SCALARS]]
            if self._equity:
                from ...env.equity import equity_histogram, equity_vs_random

                gen = self._gen()
                if f.equity_samples > 0:
                    extra.append(equity_vs_random(holes, b, f.equity_samples, gen)[:, None].float())
                if f.hist_runouts > 0:
                    extra.append(
                        equity_histogram(
                            holes, b, f.hist_runouts, f.hist_bins, f.hist_opp_samples, gen
                        ).float()
                    )
            out["scalars"] = torch.cat(extra, 1)
            if f.strength_tables:
                out = add_strength(out, load_strength(f.strength_tables))
        return out, info

    @staticmethod
    def _action_key(a: Any) -> tuple[int, int]:
        kind = int(a.kind)
        return kind, (int(a.amount) if kind == RAISE else 0)

    # ------------------------------------------------------------ own reach
    def _log_reach_one(self, state: Any, seat: int, config: Any) -> torch.Tensor:
        """``[T, 1]`` own log reach of the hand ``state.hole_cards(seat)``."""
        pol = self.policies[seat]
        lr = torch.zeros(len(pol), 1, dtype=torch.float64)
        hist = list(state.history)
        hole = list(state.hole_cards(seat))
        board = list(state.board)
        for k, (st, p, a) in enumerate(hist):
            if int(p) != seat:
                continue
            pre = self._state(config, state.button, board, hist[:k], seat, hole)
            feats, info = self._encode(pre, seat, config)
            kind, amount = self._action_key(a)
            idx = nearest_abstract(self.sp, int(st), kind, amount, info.targets, info.legal)
            P = pol.net_policies(feats)[:, 0, idx].double().cpu()
            lr = lr + torch.log(P.clamp(min=0))[:, None]
        return lr

    def _log_reach_all(self, state: Any, seat: int, config: Any) -> torch.Tensor:
        """``[T, 1326]`` own log reach of every combo (cached per prefix)."""
        pol = self.policies[seat]
        hist = list(state.history)
        board = [int(c) for c in state.board]
        base = (
            seat,
            int(state.button),
            tuple(int(s) for s in config.stacks),
            int(config.small_blind),
            int(config.big_blind),
        )
        cum = torch.zeros(len(pol), NUM_COMBOS, dtype=torch.float64)
        pub: list[tuple] = []
        for k, (st, p, a) in enumerate(hist):
            kind, amount = self._action_key(a)
            pub.append((int(p), kind, amount))
            if int(p) != seat:
                continue
            bpre = tuple(board[: BOARD_LEN[int(st)]])
            key = (base, tuple(pub), bpre)
            hit = self._reach.get(key)
            if hit is not None:
                self._reach.move_to_end(key)
                cum = hit
                continue
            pre = self._state(config, state.button, board, hist[:k], seat, self._placeholder(board))
            feats, info = self._batch(pre, seat, config, ALL_COMBOS)
            idx = nearest_abstract(self.sp, int(st), kind, amount, info.targets, info.legal)
            P = pol.net_policies(feats)[:, :, idx].double().cpu()
            cum = cum + torch.log(P.clamp(min=0))
            self._reach[key] = cum
            if len(self._reach) > self.cache_entries:
                self._reach.popitem(last=False)
        return cum

    # ------------------------------------------------------------ queries
    def probs(self, state: Any, seat: int, config: Any) -> np.ndarray:
        """``[A]`` average policy of ``seat`` holding ``state.hole_cards(seat)``."""
        pol = self.policies[seat]
        feats, info = self._encode(state, seat, config)
        lr = self._log_reach_one(state, seat, config) if pol.reach_weighted else None
        avg, _ = pol.average(feats, lr)
        p = avg[0].double().cpu().numpy()
        p = np.where(np.asarray(info.legal), np.clip(p, 0.0, None), 0.0)
        return p / p.sum()

    @torch.no_grad()
    def probs_all(self, state: Any, seat: int, config: Any) -> torch.Tensor:
        """``[1326, A]`` average policy of every combo (canonical order); rows
        of combos sharing a card with the board are zero."""
        return self.probs_all_nets(state, seat, config)[0]

    def log_reach_all(self, state: Any, seat: int, config: Any) -> torch.Tensor | None:
        """``[T, 1326]`` own log reach of every combo at ``state`` (None when
        the policy is not reach-weighted)."""
        if not self.policies[seat].reach_weighted:
            return None
        return self._log_reach_all(state, seat, config)

    @torch.no_grad()
    def probs_all_nets(
        self, state: Any, seat: int, config: Any, log_reach: Any = "history"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """:meth:`probs_all` plus every net's policy ``[T, 1326, A]`` (float64,
        CPU). ``log_reach`` (``[T, 1326]`` or None) supplies the own reach
        instead of replaying the history: after an own action ``a`` the reach
        at the next decision is ``log_reach + log(nets[:, :, a])``, exactly what
        the replay computes, so rollouts can carry it forward."""
        pol = self.policies[seat]
        board = [int(c) for c in state.board]
        hist = list(state.history)
        cur = self._state(config, state.button, board, hist, seat, self._placeholder(board))
        feats, info = self._batch(cur, seat, config, ALL_COMBOS)
        if isinstance(log_reach, str):
            log_reach = self.log_reach_all(state, seat, config)
        lr = log_reach if pol.reach_weighted else None
        avg, P = pol.average(feats, lr)
        avg = avg.float().cpu() * torch.tensor(info.legal, dtype=torch.float32)
        avg = avg / avg.sum(1, keepdim=True).clamp(min=1e-30)
        avg[board_conflicts(board)] = 0.0
        return avg, P.double().cpu()

    def probs_batch(self, state: Any, seat: int, holes: Any, config: Any) -> np.ndarray:
        """``[K, A]`` for the hands ``holes`` (rows of :meth:`probs_all`)."""
        holes = np.asarray(holes, dtype=np.int64).reshape(-1, 2)
        P = self.probs_all(state, seat, config).double().numpy()
        return P[combo_rows(holes)]
