"""Blueprint protocol used by the search, two simple implementations, and
range reconstruction from the observed betting.

A blueprint is anything with

* ``spec``: the :class:`~pokerbot.env.actions.ActionSpec` its abstract
  action indices refer to, and
* ``policy(state, player)``: the probability of every abstract action for
  ``player`` (who must be the actor) at the scalar-engine ``GameState``
  ``state``, using ``state.hole_cards(player)``. It returns a mapping
  ``{abstract_index: prob}``, a sequence of length ``spec.num_actions``, or a
  1-D tensor. Illegal entries are ignored and the rest renormalised.

Optional, for speed:

* ``policy_combos(state, player)``: a ``[1326, A]`` (or broadcastable
  ``[1, A]``) tensor with the policy of every hole combo at once. Rows of
  combos that conflict with the board are ignored. Without it the search
  calls ``policy`` once per combo on a :class:`~pokerbot.search.abstract.CardView`
  of the state with the hole cards replaced.
* ``card_independent = True`` when the policy never looks at any card (the
  uniform blueprint); rollouts then share work across boards.

The states handed to a blueprint may be :class:`CardView` wrappers: they
support the read-only part of the ``GameState`` contract (``board``,
``hole_cards``, ``history``, ``street``, ``pot``, ``street_bets``,
``stacks``, ``legal_actions()``, ``public_key()``, ``infoset_key()``, ...).

The real blueprints are adapted in :mod:`pokerbot.search.adapters`
(``TabularBlueprint``, ``NeuralBlueprint``, both with a vectorised
``policy_combos``) and registered here as ``search:blueprint:<strategy file>``
and ``search:neural:<checkpoint dir>``. Other blueprints: register a factory
with :func:`register_blueprint` so ``search:<prefix>:<path>`` works from the
match runner.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from types import ModuleType
from typing import Any, Protocol, runtime_checkable

import torch

from ..engine_select import get_engine
from ..env.actions import DEFAULT_SPEC, ActionSpec
from .abstract import CardView, legal_options, make_state, map_concrete, to_action
from .combos import NUM_COMBOS, combo_cards, combo_table, valid_mask


@runtime_checkable
class Blueprint(Protocol):
    spec: ActionSpec

    def policy(self, state: Any, player: int) -> Any: ...


def _normalise(raw: Any, legal: list[int], width: int) -> list[float]:
    vec = [0.0] * width
    if isinstance(raw, Mapping):
        for k, v in raw.items():
            if 0 <= int(k) < width:
                vec[int(k)] = float(v)
    else:
        vals = raw.tolist() if isinstance(raw, torch.Tensor) else list(raw)
        for i, v in enumerate(vals[:width]):
            vec[i] = float(v)
    legal_set = set(legal)
    vec = [max(0.0, v) if i in legal_set else 0.0 for i, v in enumerate(vec)]
    s = sum(vec)
    if s <= 0:
        return [1.0 / len(legal) if i in legal_set else 0.0 for i in range(width)]
    return [v / s for v in vec]


def policy_vector(bp: Any, state: Any, player: int) -> list[float]:
    """Normalised policy over ``bp.spec`` indices for the state's own hole cards."""
    legal = [o.index for o in legal_options(state, bp.spec)]
    return _normalise(bp.policy(state, player), legal, bp.spec.num_actions)


def normalise_combos(P: Any, legal_t: torch.Tensor, device: Any = "cpu") -> torch.Tensor:
    """Rows of a ``policy_combos`` result normalised over the legal actions
    (uniform where a row has no legal mass)."""
    P = torch.as_tensor(P, dtype=torch.float32, device=device)
    if P.dim() == 1:
        P = P[None]
    P = P.clamp(min=0) * legal_t
    s = P.sum(-1, keepdim=True)
    uni = legal_t / legal_t.sum()
    return torch.where(s > 0, P / s.clamp(min=1e-30), uni)


def policy_matrix(
    bp: Any,
    state: Any,
    player: int,
    board: Sequence[int] | None = None,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Blueprint policy of ``player`` for every hole combo: ``[1326, A]`` or ``[1, A]``.

    ``board`` optionally overrides the state's board (see :class:`CardView`).
    Rows are normalised over the legal abstract actions; rows of combos that
    conflict with the board are uniform over legal actions.
    """
    A = bp.spec.num_actions
    legal = [o.index for o in legal_options(state, bp.spec)]
    view = state if board is None and isinstance(state, CardView) else CardView(state, board)
    legal_t = torch.zeros(A, device=device)
    legal_t[legal] = 1.0
    fn = getattr(bp, "policy_combos", None)
    if fn is not None:
        return normalise_combos(fn(view, player), legal_t, device)
    if getattr(bp, "card_independent", False):
        row = _normalise(bp.policy(view, player), legal, A)
        return torch.tensor([row], dtype=torch.float32, device=device)
    P = (legal_t / legal_t.sum()).repeat(NUM_COMBOS, 1)
    ok = valid_mask(view.board)
    rows = []
    idx = ok.nonzero().flatten().tolist()
    for c in idx:
        rows.append(_normalise(bp.policy(view.with_hole(player, combo_cards(c)), player), legal, A))
    if idx:
        P[torch.tensor(idx)] = torch.tensor(rows, dtype=torch.float32)
    return P.to(device)


class UniformBlueprint:
    """Uniform over the legal abstract actions; ignores all cards."""

    card_independent = True

    def __init__(self, spec: ActionSpec = DEFAULT_SPEC) -> None:
        self.spec = spec

    def policy(self, state: Any, player: int) -> dict[int, float]:
        opts = legal_options(state, self.spec)
        return {o.index: 1.0 / len(opts) for o in opts}

    def policy_combos(self, state: Any, player: int) -> torch.Tensor:
        v = torch.zeros(1, self.spec.num_actions)
        opts = legal_options(state, self.spec)
        for o in opts:
            v[0, o.index] = 1.0 / len(opts)
        return v


class TabularBlueprintFromCallable:
    """Wraps ``fn(state, player) -> {abstract_index: prob} | sequence``.

    ``fn`` reads ``state.hole_cards(player)``; for batched queries the search
    hands it :class:`CardView` states. Set ``card_independent`` only if ``fn``
    never looks at any card.
    """

    def __init__(
        self,
        fn: Callable[[Any, int], Any],
        spec: ActionSpec = DEFAULT_SPEC,
        card_independent: bool = False,
    ) -> None:
        self.fn = fn
        self.spec = spec
        self.card_independent = card_independent

    def policy(self, state: Any, player: int) -> Any:
        return self.fn(state, player)


def range_reach(
    bp: Any,
    config: Any,
    button: int,
    board: Sequence[int],
    history: Sequence,
    engine: ModuleType | None = None,
    device: torch.device | str = "cpu",
    exclude: Mapping[int, Sequence[int]] | None = None,
) -> torch.Tensor:
    """Both players' reach probabilities over the 1326 combos, ``[2, 1326]``.

    Replays ``history`` (``(street, player, action)`` tuples, as in
    ``GameState.history``) from the deal; at every action the acting player's
    reach of each combo is multiplied by the blueprint probability of that
    action (off-tree sizes via :func:`map_concrete`). Combos that conflict
    with ``board`` get zero reach, and ``exclude[p]`` lists cards removed from
    player ``p``'s range (e.g. the searching agent's own hole cards from the
    opponent's range). A range the blueprint makes empty falls back to uniform.
    """
    engine = engine or get_engine()
    reach = torch.ones(2, NUM_COMBOS, device=device)
    state = make_state(engine, config, button, board, [])
    for _street, player, action in history:
        opts = legal_options(state, bp.spec)
        m = map_concrete(state, bp.spec, int(action.kind), int(action.amount), opts)
        if m:
            P = policy_matrix(bp, state, player, device=device)
            prob = sum(P[:, i] * w for i, w in m.items())
            reach[player] = reach[player] * prob
        state.apply(to_action(engine, int(action.kind), int(action.amount)))
    ok = valid_mask(board, device)
    reach = reach * ok
    if exclude:
        cards = combo_table(device)
        for p, ex in exclude.items():
            for c in ex:
                reach[p] = reach[p] * ((cards[:, 0] != c) & (cards[:, 1] != c))
    for p in range(2):
        if float(reach[p].sum()) <= 0:
            reach[p] = ok.float()
    return reach


# -- registry for ``search:<blueprint spec>`` -------------------------------

BlueprintFactory = Callable[..., Any]


def _tabular(arg: str, **kwargs: Any) -> Any:
    from .adapters import tabular_blueprint

    return tabular_blueprint(arg, **kwargs)


def _neural(arg: str, **kwargs: Any) -> Any:
    from .adapters import neural_blueprint

    return neural_blueprint(arg, **kwargs)


BLUEPRINTS: dict[str, BlueprintFactory] = {
    "uniform": lambda arg, **kw: UniformBlueprint(),
    "blueprint": _tabular,  # search:blueprint:<strategy.bin>
    "neural": _neural,  # search:neural:<checkpoint dir>
}


def register_blueprint(prefix: str, factory: BlueprintFactory) -> None:
    """Make ``search:<prefix>[:<arg>]`` build its blueprint with
    ``factory(arg, **blueprint_kwargs)``."""
    BLUEPRINTS[prefix] = factory


def make_blueprint(spec: str, **kwargs: Any) -> Any:
    """Build a blueprint from a spec string: ``uniform``, ``<prefix>:<arg>`` for
    a registered prefix (``blueprint:<strategy file>``, ``neural:<checkpoint
    dir>``), or any agent name whose agent exposes ``policy`` (and ``spec``)
    or a ``blueprint`` attribute that does. ``kwargs`` go to the factory."""
    prefix, _, arg = spec.partition(":")
    if prefix in BLUEPRINTS:
        return BLUEPRINTS[prefix](arg, **kwargs)
    from ..agents import make_agent

    agent = make_agent(spec, **kwargs)
    for obj in (agent, getattr(agent, "blueprint", None)):
        if obj is not None and callable(getattr(obj, "policy", None)):
            if not hasattr(obj, "spec"):
                obj.spec = DEFAULT_SPEC
            return obj
    raise ValueError(
        f"cannot use {spec!r} as a search blueprint: it exposes no policy(state, player); "
        "wrap it and register it with pokerbot.search.blueprint.register_blueprint"
    )
