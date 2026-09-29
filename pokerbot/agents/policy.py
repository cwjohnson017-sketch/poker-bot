"""Policy-exposing agents and the scalar mapping to abstract actions.

Evaluation tools that reason about an opponent's strategy (local best
response, the scalar adapter of the approximate best response) need more
than ``act``: they need the probability of every abstract action for any
hand the opponent might hold. This module defines that interface.

``PolicyAgent``
    An :class:`~pokerbot.agents.base.Agent` with
    ``policy(state, seat) -> dict[abstract_index, prob] | array[A]`` giving
    the action distribution of ``seat`` holding ``state.hole_cards(seat)``.
    Abstract indices refer to an :class:`~pokerbot.env.actions.ActionSpec`
    (the agent's ``spec`` attribute, ``DEFAULT_SPEC`` when absent). An agent
    may also provide ``policy_batch(state, seat, holes) -> array[K, A]``
    for ``K`` hypothetical hole-card pairs at once; tools use it when
    present and otherwise call ``policy`` once per hand.

The scalar helpers here reproduce the torch environment's abstract action
semantics (:mod:`pokerbot.env.actions`) exactly on a contract ``GameState``:
the same raise-to amounts, the same legality mask and dedupe rule.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Protocol, runtime_checkable

import numpy as np

from ..engine_select import get_engine, to_engine_action
from .base import BaseAgent

FOLD, CHECK_CALL, RAISE = 0, 1, 2


def default_spec() -> Any:
    from ..env.actions import DEFAULT_SPEC  # torch import deferred until needed

    return DEFAULT_SPEC


@runtime_checkable
class PolicyAgent(Protocol):
    name: str

    def new_hand(self, seat: int, config: Any) -> None: ...

    def act(self, state: Any, seat: int, rng: np.random.Generator) -> Any: ...

    def observe_end(self, state: Any) -> None: ...

    def policy(self, state: Any, seat: int) -> Mapping[int, float] | np.ndarray: ...


# ------------------------------------------------------------------ scalar abstract actions


@dataclass(frozen=True)
class AbstractChoice:
    """A legal abstract action in a concrete state."""

    index: int
    kind: int  # FOLD / CHECK_CALL / RAISE
    amount: int  # raise-to amount (0 unless RAISE)
    label: str

    def action(self, engine: ModuleType | None = None) -> Any:
        engine = engine or get_engine()
        if self.kind == FOLD:
            return engine.Action.fold()
        if self.kind == CHECK_CALL:
            return engine.Action.check_call()
        return engine.Action.raise_to(int(self.amount))


def raises_this_street(state: Any) -> int:
    st = state.street
    return sum(1 for s, _p, a in state.history if s == st and int(a.kind) == RAISE)


def abstract_targets(state: Any, spec: Any = None) -> list[int]:
    """Concrete raise-to amount of every abstract entry of the current street
    (entries that are not raises get the clamp bound), as in ``raise_targets``."""
    spec = spec or default_spec()
    legal = state.legal_actions()
    street = min(int(state.street), 3)
    seat = state.current_player
    bets = list(state.street_bets)
    max_bet = max(bets)
    to_call = max_bet - bets[seat]
    pot = int(state.pot)
    lo, hi = int(legal.min_raise_to), int(legal.max_raise_to)
    out = []
    for a in spec.streets[street]:
        if a[0] == "raise":
            raw = max_bet + (int(round(float(a[1]) * 1000)) * (pot + to_call) + 500) // 1000
        elif a[0] == "raise_x":
            raw = (int(round(float(a[1]) * 1000)) * max_bet + 500) // 1000
        else:
            raw = hi
        out.append(min(max(raw, lo), hi))
    return out


def abstract_actions(state: Any, spec: Any = None) -> list[AbstractChoice]:
    """Legal abstract actions of the player to act, in index order.

    Mirrors :func:`pokerbot.env.actions.legal_mask`: fold only when facing a
    bet, check/call always, raises need a legal raise and fewer than
    ``spec.max_raises`` raises this street, sized raises that would be all-in
    are left to ``allin``, and duplicates of an earlier sized raise are
    dropped when ``spec.dedupe``.
    """
    spec = spec or default_spec()
    if state.is_terminal:
        return []
    legal = state.legal_actions()
    street = min(int(state.street), 3)
    seat = state.current_player
    bets = list(state.street_bets)
    to_call = max(bets) - bets[seat]
    hi = int(legal.max_raise_to)
    can_raise = legal.min_raise_to > 0 and raises_this_street(state) < spec.max_raises
    targets = abstract_targets(state, spec)
    acts = spec.streets[street]
    sized_ok = [
        a[0] in ("raise", "raise_x") and can_raise and targets[i] < hi for i, a in enumerate(acts)
    ]
    out: list[AbstractChoice] = []
    for i, a in enumerate(acts):
        kind = a[0]
        label = kind if len(a) == 1 else f"{kind} {a[1]:g}"
        if kind == "fold":
            if to_call > 0:
                out.append(AbstractChoice(i, FOLD, 0, label))
        elif kind == "check_call":
            out.append(AbstractChoice(i, CHECK_CALL, 0, label))
        elif kind == "allin":
            if can_raise:
                out.append(AbstractChoice(i, RAISE, hi, label))
        elif sized_ok[i]:
            if spec.dedupe and any(sized_ok[j] and targets[j] == targets[i] for j in range(i)):
                continue
            out.append(AbstractChoice(i, RAISE, targets[i], label))
    return out


def legal_vector(state: Any, spec: Any = None) -> np.ndarray:
    spec = spec or default_spec()
    m = np.zeros(spec.num_actions, dtype=bool)
    for c in abstract_actions(state, spec):
        m[c.index] = True
    return m


def nearest_abstract_index(state: Any, action: Any, spec: Any = None) -> int:
    """Abstract index of a concrete ``action`` taken in ``state``.

    Fold and check/call map to their entries; a raise maps to the legal
    raise entry with the closest raise-to amount (ties to the lower index).
    """
    spec = spec or default_spec()
    choices = abstract_actions(state, spec)
    kind = int(action.kind)
    for c in choices:
        if kind != RAISE and c.kind == kind:
            return c.index
    raises = [c for c in choices if c.kind == RAISE]
    if kind == RAISE and raises:
        amt = int(action.amount)
        return min(raises, key=lambda c: (abs(c.amount - amt), c.index)).index
    # fold when checking is free, or a raise with no legal abstract raise: call
    return next(c.index for c in choices if c.kind == CHECK_CALL)


def policy_vector(
    pol: Mapping[int, float] | Sequence[float] | Any, legal: np.ndarray
) -> np.ndarray:
    """Normalize a policy output (dict, list, numpy or torch) to a probability
    vector over ``len(legal)`` abstract actions, zero on illegal entries.
    A policy with no mass on legal actions becomes uniform over them."""
    A = len(legal)
    v = np.zeros(A, dtype=np.float64)
    if isinstance(pol, Mapping):
        for k, p in pol.items():
            if 0 <= int(k) < A:
                v[int(k)] = float(p)
    else:
        if hasattr(pol, "detach"):
            pol = pol.detach().cpu().numpy()
        arr = np.asarray(pol, dtype=np.float64).reshape(-1)
        v[: min(A, arr.size)] = arr[:A]
    v = np.where(legal, np.clip(v, 0.0, None), 0.0)
    s = v.sum()
    if s <= 0:
        return legal.astype(np.float64) / max(1, legal.sum())
    return v / s


def normalize_rows(probs: np.ndarray, legal: np.ndarray) -> np.ndarray:
    """Row-wise :func:`policy_vector` for a ``[K, A']`` array."""
    A = len(legal)
    v = np.zeros((probs.shape[0], A))
    w = min(A, probs.shape[1])
    v[:, :w] = probs[:, :w]
    v = np.where(legal[None, :], np.clip(v, 0.0, None), 0.0)
    s = v.sum(1, keepdims=True)
    uniform = legal.astype(np.float64) / max(1, legal.sum())
    return np.where(s > 0, v / np.where(s > 0, s, 1.0), uniform[None, :])


def sample_abstract(
    state: Any, probs: np.ndarray, rng: np.random.Generator, spec: Any = None
) -> AbstractChoice:
    choices = {c.index: c for c in abstract_actions(state, spec)}
    idx = sorted(choices)
    p = np.array([probs[i] for i in idx], dtype=np.float64)
    p = p / p.sum() if p.sum() > 0 else np.full(len(idx), 1.0 / len(idx))
    return choices[idx[int(rng.choice(len(idx), p=p))]]


# --------------------------------------------------------------------------- hypothetical states


def hypothetical_state(
    state: Any,
    seat: int,
    hole: Sequence[int],
    config: Any,
    known: Mapping[int, Sequence[int]] | None = None,
    engine: ModuleType | None = None,
    upto: int | None = None,
) -> Any:
    """Full ``GameState`` equal to ``state``'s public history (optionally only
    its first ``upto`` actions) with ``seat`` holding ``hole``.

    ``known`` gives hole cards of other seats (e.g. the evaluating agent's
    own cards); every other unseen card is filled deterministically. The
    dealt board is kept."""
    engine = engine or get_engine()
    n = state.num_players
    deck: list[int | None] = [None] * 52
    holes = dict(known or {})
    holes[seat] = list(hole)
    for p, cards in holes.items():
        deck[2 * p], deck[2 * p + 1] = int(cards[0]), int(cards[1])
    for j, c in enumerate(state.board):
        deck[2 * n + j] = int(c)
    used = {c for c in deck if c is not None}
    if len(used) != sum(c is not None for c in deck):
        raise ValueError("hypothetical hole cards collide with known cards")
    fill = iter(c for c in range(52) if c not in used)
    full = [c if c is not None else next(fill) for c in deck]
    new = engine.GameState.new_hand(config, state.button, full)
    hist = state.history if upto is None else state.history[:upto]
    for _s, _p, a in hist:
        new.apply(to_engine_action(engine, a))
    return new


def range_policy(
    agent: Any,
    state: Any,
    seat: int,
    holes: np.ndarray,
    config: Any,
    known: Mapping[int, Sequence[int]] | None = None,
    engine: ModuleType | None = None,
    spec: Any = None,
    upto: int | None = None,
) -> np.ndarray:
    """``[K, A]`` action probabilities of ``agent`` in ``seat`` for each of the
    ``K`` hypothetical hole-card pairs ``holes`` (a ``[K, 2]`` array), at the
    decision point described by ``state`` (whose current player is ``seat``),
    or by its first ``upto`` actions.

    Uses ``agent.policy_batch`` when available, else one ``policy`` call per
    hand on a masked hypothetical state."""
    from ..eval.masking import MaskedState

    spec = spec or getattr(agent, "spec", None) or default_spec()
    holes = np.asarray(holes, dtype=np.int64).reshape(-1, 2)
    K = holes.shape[0]
    A = spec.num_actions
    if K == 0:
        return np.zeros((0, A))
    engine = engine or get_engine()
    base = hypothetical_state(state, seat, holes[0], config, known, engine, upto)
    if base.current_player != seat:
        raise ValueError(f"seat {seat} is not to act at this decision point")
    legal = legal_vector(base, spec)
    if hasattr(agent, "policy_batch"):
        view = MaskedState(base, seat, config, np.random.default_rng(0))
        out = agent.policy_batch(view, seat, holes)
        if hasattr(out, "detach"):
            out = out.detach().cpu().numpy()
        return normalize_rows(np.asarray(out, dtype=np.float64).reshape(K, -1), legal)
    rows = np.empty((K, A))
    for k in range(K):
        st = (
            base
            if k == 0
            else hypothetical_state(state, seat, holes[k], config, known, engine, upto)
        )
        view = MaskedState(st, seat, config, np.random.default_rng(k))
        rows[k] = policy_vector(agent.policy(view, seat), legal)
    return rows


# --------------------------------------------------------------------------- policy agents


class PolicyAgentBase(BaseAgent):
    """``act`` samples an abstract action from ``policy``."""

    def __init__(self, spec: Any = None, name: str | None = None) -> None:
        super().__init__(name)
        self.spec = spec or default_spec()

    def policy(self, state: Any, seat: int) -> Mapping[int, float] | np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def act(self, state: Any, seat: int, rng: np.random.Generator) -> Any:
        probs = policy_vector(self.policy(state, seat), legal_vector(state, self.spec))
        return sample_abstract(state, probs, rng, self.spec).action(self._engine)


class UniformPolicyAgent(PolicyAgentBase):
    """Uniform over the legal abstract actions (the scalar twin of the
    vectorized ``UniformRandomVecPolicy``)."""

    name = "uniform"

    def policy(self, state: Any, seat: int) -> np.ndarray:
        m = legal_vector(state, self.spec).astype(np.float64)
        return m / m.sum()

    def policy_batch(self, state: Any, seat: int, holes: np.ndarray) -> np.ndarray:
        return np.tile(self.policy(state, seat), (len(holes), 1))


class FixedPolicyAgent(PolicyAgentBase):
    """Always the same abstract action kind (``"check_call"``, ``"fold"``,
    ``"allin"``), falling back to check/call when it is illegal. Its policy is
    known exactly, so it needs no sampling."""

    def __init__(self, kind: str = "check_call", spec: Any = None, name: str | None = None) -> None:
        super().__init__(spec, name or f"fixed_{kind}")
        self.kind = kind

    def policy(self, state: Any, seat: int) -> dict[int, float]:
        street = min(int(state.street), 3)
        legal = legal_vector(state, self.spec)
        for i, a in enumerate(self.spec.streets[street]):
            if a[0] == self.kind and legal[i]:
                return {i: 1.0}
        call = next(i for i, a in enumerate(self.spec.streets[street]) if a[0] == "check_call")
        return {call: 1.0}

    def policy_batch(self, state: Any, seat: int, holes: np.ndarray) -> np.ndarray:
        v = policy_vector(self.policy(state, seat), legal_vector(state, self.spec))
        return np.tile(v, (len(holes), 1))


class SampledPolicyAgent(PolicyAgentBase):
    """Wraps an act-only agent; its policy is estimated from ``samples`` calls
    to ``act`` (each mapped to the nearest abstract action). Estimates are
    noisy and costly, so this warns once on construction."""

    def __init__(
        self, agent: Any, samples: int = 16, spec: Any = None, warn: bool = True, seed: int = 0
    ) -> None:
        super().__init__(spec, getattr(agent, "name", type(agent).__name__))
        self.agent = agent
        self.samples = int(samples)
        self._rng = np.random.default_rng(seed)
        if warn:
            warnings.warn(
                f"agent {self.name!r} exposes no policy(); estimating it from {self.samples} "
                "sampled act() calls per query (slow and noisy)",
                stacklevel=2,
            )

    def new_hand(self, seat: int, config: Any) -> None:
        super().new_hand(seat, config)
        self.agent.new_hand(seat, config)

    def observe_end(self, state: Any) -> None:
        self.agent.observe_end(state)

    def act(self, state: Any, seat: int, rng: np.random.Generator) -> Any:
        return self.agent.act(state, seat, rng)

    def policy(self, state: Any, seat: int) -> np.ndarray:
        counts = np.zeros(self.spec.num_actions)
        for _ in range(self.samples):
            rng = np.random.default_rng(int(self._rng.integers(2**63)))
            a = self.agent.act(state, seat, rng)
            counts[nearest_abstract_index(state, a, self.spec)] += 1
        return counts / counts.sum()


def as_policy_agent(agent: Any, samples: int = 16, spec: Any = None) -> Any:
    """Return ``agent`` itself when it has ``policy``; an exact policy for the
    always-call baseline; otherwise a :class:`SampledPolicyAgent` (warns)."""
    if hasattr(agent, "policy"):
        return agent
    from .baselines import AlwaysCallAgent

    if isinstance(agent, AlwaysCallAgent):
        return FixedPolicyAgent("check_call", spec, name=agent.name)
    return SampledPolicyAgent(agent, samples, spec)
