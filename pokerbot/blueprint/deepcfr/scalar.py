"""Scalar (one-hand) mirror of the env's action mapping and observation encoder.

The networks are trained on features produced by ``VecNLHE.obs()``. At play
time the agent sees a contract ``GameState`` (Rust ``poker_engine``, the
reference engine, or the match runner's ``MaskedState`` view of either), so
this module re-derives the same features in plain Python integers:

* :func:`raise_targets`, :func:`legal_mask` mirror
  ``pokerbot/env/actions.py`` line by line (exact integer arithmetic, same
  clamping, same dedupe rule).
* :func:`nearest_abstract` maps a concrete action to an abstract index with
  the env's rule (fold/check-call to their index, a raise to exactly the
  all-in to the ``allin`` entry, any other raise to the raise-type entry with
  the closest amount, first on ties), restricted to the entries that are
  legal in the abstraction when any raise entry is legal. This is the index
  ``VecNLHE.step_concrete`` records, and on the tree exactly the index
  ``VecNLHE.step`` records for the abstract action. Off the tree it is the
  nearest legal size.
* :func:`harmonic_abstract` is the alternative for **opponent** off-tree
  raises (``offtree="harmonic"``): the pseudo-harmonic mapping of
  :func:`pokerbot.abstraction.actions.map_offtree` between the neighbouring
  legal abstract sizes, randomized with an ``rng`` (the acting agent) or
  deterministic with ``u = 0.5`` (the stateless range policy). The seat's
  own actions, on-tree sizes, folds, calls, and raises made when no abstract
  raise is legal keep the :func:`nearest_abstract` index.
* :func:`encode_state` rebuilds the history tokens by replaying the public
  action history through the scalar engine (hole cards of the opponent and
  the undealt board are irrelevant to every public quantity, so they are
  filled with arbitrary unused cards), then assembles ``cards``,
  ``card_mask``, ``hist``, ``hist_amt``, ``scalars`` and ``legal`` exactly
  as :func:`pokerbot.env.obs.encode_obs` does (float32 division included).
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from ...engine_select import get_engine, to_engine_action
from ...env.actions import (
    CHECK_CALL,
    FOLD,
    K_ALLIN,
    K_CHECK_CALL,
    K_FOLD,
    K_RAISE_MULT,
    K_RAISE_POT,
    RAISE,
    ActionSpec,
)
from ...env.cards import NO_CARD
from ...env.vec_env import HISTORY_LEN
from .features import FeatureConfig

_KIND_CODES = {
    "fold": K_FOLD,
    "check_call": K_CHECK_CALL,
    "raise": K_RAISE_POT,
    "raise_x": K_RAISE_MULT,
    "allin": K_ALLIN,
}
_RAISE_KINDS = (K_RAISE_POT, K_RAISE_MULT, K_ALLIN)
OFFTREE_MODES = ("harmonic", "nearest")


@dataclass(frozen=True)
class ScalarSpec:
    """Per street: tuple of ``(kind code, size in thousandths)``, padded to A."""

    rows: tuple[tuple[tuple[int, int], ...], ...]
    num_actions: int
    max_raises: int
    dedupe: bool

    @staticmethod
    def build(spec: ActionSpec) -> ScalarSpec:
        A = spec.num_actions
        rows = []
        for st in spec.streets:
            row = []
            for a in st:
                param = int(round(float(a[1]) * 1000)) if len(a) == 2 else 0
                row.append((_KIND_CODES[a[0]], param))
            row += [(-1, 0)] * (A - len(row))
            rows.append(tuple(row))
        return ScalarSpec(tuple(rows), A, spec.max_raises, spec.dedupe)

    def concrete_kind(self, street: int, idx: int) -> int:
        k = self.rows[street][idx][0]
        if k == K_FOLD:
            return FOLD
        if k in _RAISE_KINDS:
            return RAISE
        return CHECK_CALL


def raise_targets(
    sp: ScalarSpec,
    street: int,
    pot: int,
    max_bet: int,
    to_call: int,
    min_raise_to: int,
    max_raise_to: int,
) -> list[int]:
    """Concrete raise-to of every abstract entry (mirror of ``actions.raise_targets``)."""
    out = []
    for kind, param in sp.rows[street]:
        if kind == K_RAISE_POT:
            raw = max_bet + (param * (pot + to_call) + 500) // 1000
        elif kind == K_RAISE_MULT:
            raw = (param * max_bet + 500) // 1000
        else:
            raw = max_raise_to
        out.append(min(max(raw, min_raise_to), max_raise_to))
    return out


def legal_mask(
    sp: ScalarSpec,
    street: int,
    to_call: int,
    raise_ok: bool,
    n_raises: int,
    targets: Sequence[int],
    max_raise_to: int,
) -> list[bool]:
    """Legal abstract actions of a live decision (mirror of ``actions.legal_mask``)."""
    can_raise = raise_ok and n_raises < sp.max_raises
    kinds = [k for k, _ in sp.rows[street]]
    sized = [k in (K_RAISE_POT, K_RAISE_MULT) for k in kinds]
    sized_l = [s and can_raise and targets[i] < max_raise_to for i, s in enumerate(sized)]
    if sp.dedupe:
        dedup = list(sized_l)
        for i in range(len(kinds)):
            if any(sized_l[j] and targets[j] == targets[i] for j in range(i)):
                dedup[i] = False
        sized_l = dedup
    mask = []
    for i, k in enumerate(kinds):
        if k == K_FOLD:
            mask.append(to_call > 0)
        elif k == K_CHECK_CALL:
            mask.append(True)
        elif k == K_ALLIN:
            mask.append(can_raise)
        else:
            mask.append(sized_l[i])
    return mask


def nearest_abstract(
    sp: ScalarSpec,
    street: int,
    kind: int,
    amount: int,
    targets: Sequence[int],
    legal: Sequence[bool] | None = None,
) -> int:
    """Abstract index of a concrete action (see the module docstring)."""
    kinds = [k for k, _ in sp.rows[street]]
    if kind == FOLD:
        return kinds.index(K_FOLD)
    if kind == CHECK_CALL:
        return kinds.index(K_CHECK_CALL)
    cand = [i for i, k in enumerate(kinds) if k in _RAISE_KINDS]
    if legal is not None:
        leg = [i for i in cand if legal[i]]
        if leg:
            cand = leg
    for i, k in enumerate(kinds):
        if k == K_ALLIN and targets[i] == amount:
            return i
    return min(cand, key=lambda i: (abs(targets[i] - amount), i))


def check_offtree(offtree: str) -> str:
    if offtree not in OFFTREE_MODES:
        raise ValueError(f"offtree must be one of {OFFTREE_MODES}, got {offtree!r}")
    return offtree


def harmonic_abstract(
    spec: ActionSpec,
    sp: ScalarSpec,
    state: Any,
    action: Any,
    info: DecisionInfo,
    rng: np.random.Generator | None = None,
) -> int | None:
    """Pseudo-harmonic index of an off-tree raise ``action`` at ``state``.

    Returns None (keep :func:`nearest_abstract`) unless ``action`` is a raise
    whose amount is not the target of a legal abstract raise and some
    abstract raise is legal. Otherwise the index is
    :func:`~pokerbot.abstraction.actions.map_offtree`'s, randomized with
    ``rng`` or deterministic (``u = 0.5``) without one.
    """
    if int(action.kind) != RAISE:
        return None
    amount = int(action.amount)
    street = info.street
    raises = [i for i, (k, _) in enumerate(sp.rows[street]) if k in _RAISE_KINDS and info.legal[i]]
    if not raises or any(info.targets[i] == amount for i in raises):
        return None
    from ...abstraction.actions import map_offtree

    mode = "deterministic" if rng is None else "randomized"
    idx = map_offtree(spec, state, action, rng, mode)
    if idx is None or sp.rows[street][idx][0] not in _RAISE_KINDS:
        return None
    return int(idx)


# ---------------------------------------------------------------------------- engines


def engine_for(config: Any) -> Any:
    """The scalar engine module whose ``GameConfig`` type ``config`` is."""
    mod = type(config).__module__
    if mod.startswith("pokerbot.reference"):
        return importlib.import_module("pokerbot.reference")
    if mod.startswith("poker_engine"):
        return importlib.import_module("poker_engine")
    return get_engine()


def engine_config(engine: Any, config: Any) -> Any:
    if type(config).__module__.split(".")[0] == engine.__name__.split(".")[0]:
        return config
    return engine.GameConfig(
        num_players=2,
        stacks=[int(s) for s in config.stacks],
        small_blind=int(config.small_blind),
        big_blind=int(config.big_blind),
        ante=int(getattr(config, "ante", 0)),
    )


@dataclass
class DecisionInfo:
    """Public quantities of one decision point (what the env's LegalInfo holds)."""

    street: int
    player: int
    pot: int
    street_bets: list[int]
    stacks: list[int]
    max_bet: int
    to_call: int
    raise_ok: bool
    min_raise_to: int
    max_raise_to: int
    n_raises: int
    targets: list[int]
    legal: list[bool]


def decision_info(state: Any, sp: ScalarSpec, n_raises: int) -> DecisionInfo:
    p = int(state.current_player)
    street = int(state.street)
    bets = [int(b) for b in state.street_bets]
    stacks = [int(s) for s in state.stacks]
    la = state.legal_actions()
    max_bet = max(bets)
    to_call = max_bet - bets[p]
    raise_ok = int(la.min_raise_to) > 0
    mn = int(la.min_raise_to) if raise_ok else 0
    mx = int(la.max_raise_to) if raise_ok else 0
    pot = int(state.pot)
    targets = raise_targets(sp, street, pot, max_bet, to_call, mn, mx)
    legal = legal_mask(sp, street, to_call, raise_ok, n_raises, targets, mx)
    return DecisionInfo(
        street, p, pot, bets, stacks, max_bet, to_call, raise_ok, mn, mx, n_raises, targets, legal
    )


def abstract_to_action(engine: Any, info: DecisionInfo, sp: ScalarSpec, idx: int) -> Any:
    """Concrete engine ``Action`` for abstract index ``idx`` (the env's sizing rule)."""
    kind = sp.concrete_kind(info.street, idx)
    if kind == FOLD:
        return engine.Action.fold()
    if kind == CHECK_CALL:
        return engine.Action.check_call()
    return engine.Action.raise_to(int(info.targets[idx]))


@dataclass
class HistoryRecord:
    tokens: list[int]
    amounts: list[int]  # chips added per action
    n_raises: int  # voluntary raises on the current street


def replay_history(
    state: Any,
    config: Any,
    sp: ScalarSpec,
    seat: int,
    spec: ActionSpec | None = None,
    offtree: str = "nearest",
    rng: np.random.Generator | None = None,
    memo: dict | None = None,
) -> HistoryRecord:
    """Rebuild the env's history tokens by replaying ``state.history`` in the engine.

    With ``offtree="harmonic"`` (needs ``spec``) the opponent's off-tree
    raises get :func:`harmonic_abstract` indices (randomized with ``rng``).
    ``memo`` keeps the index drawn for each history position, so a hand's
    earlier opponent actions keep their mapping at later decisions; the
    caller clears it between hands.
    """
    harmonic = check_offtree(offtree) == "harmonic"
    if harmonic and spec is None:
        raise ValueError('offtree="harmonic" needs the ActionSpec')
    engine = engine_for(config)
    cfg = engine_config(engine, config)
    button = int(state.button)
    hole = [int(c) for c in state.hole_cards(seat)]
    board = [int(c) for c in state.board]
    used = set(hole) | set(board)
    rest = iter(c for c in range(52) if c not in used)
    deck: list[int] = [0] * 9
    deck[2 * seat : 2 * seat + 2] = hole
    deck[2 * (1 - seat) : 2 * (1 - seat) + 2] = [next(rest), next(rest)]
    for j in range(5):
        deck[4 + j] = board[j] if j < len(board) else next(rest)
    deck += list(rest)
    sim = engine.GameState.new_hand(cfg, button, deck)
    A = sp.num_actions
    tokens, amounts = [], []
    n_raises, street = 0, 0
    for pos, (st, player, action) in enumerate(state.history):
        st, player = int(st), int(player)
        if st != street:
            street, n_raises = st, 0
        info = decision_info(sim, sp, n_raises)
        kind, amount = int(action.kind), int(getattr(action, "amount", 0) or 0)
        idx = None
        if harmonic and player != seat and kind == RAISE:
            key = (seat, pos, amount)
            idx = None if memo is None else memo.get(key)
            if idx is None:
                idx = harmonic_abstract(spec, sp, sim, action, info, rng)
                if memo is not None and idx is not None:
                    memo[key] = idx
        if idx is None:
            idx = nearest_abstract(sp, st, kind, amount, info.targets, info.legal)
        if kind == CHECK_CALL:
            add = min(info.to_call, info.stacks[player])
        elif kind == RAISE:
            add = amount - info.street_bets[player]
            n_raises += 1
        else:
            add = 0
        tokens.append(1 + (st * 2 + int(player == button)) * A + idx)
        amounts.append(add)
        sim.apply(to_engine_action(engine, action))
    if int(state.street) != street:
        n_raises = 0
    return HistoryRecord(tokens, amounts, n_raises)


def encode_state(
    state: Any,
    seat: int,
    config: Any,
    spec: ActionSpec,
    features: FeatureConfig | None = None,
    history_len: int = HISTORY_LEN,
    generator: torch.Generator | None = None,
    sp: ScalarSpec | None = None,
    offtree: str = "nearest",
    rng: np.random.Generator | None = None,
    memo: dict | None = None,
) -> tuple[dict[str, torch.Tensor], DecisionInfo]:
    """Features ``[1, ...]`` for ``seat`` to act in ``state`` (mirror of ``encode_obs``).

    ``offtree``, ``rng`` and ``memo`` choose how the opponent's off-tree
    raises are tokenised (see :func:`replay_history`); the default
    ``"nearest"`` is exactly the env's ``step_concrete`` rule.

    Returns the canonical feature dict (see :mod:`.features`) on the CPU and
    the decision's :class:`DecisionInfo` (for mapping abstract actions to
    concrete ones).
    """
    sp = sp or ScalarSpec.build(spec)
    features = features or FeatureConfig()
    rec = replay_history(state, config, sp, seat, spec, offtree, rng, memo)
    info = decision_info(state, sp, rec.n_raises)
    if info.player != seat:
        raise ValueError(f"seat {seat} is not to act (current player {info.player})")
    start = np.float32(int(config.stacks[seat]))
    hole = [int(c) for c in state.hole_cards(seat)]
    board = [int(c) for c in state.board]
    cards = hole + board + [NO_CARD] * (5 - len(board))
    card_mask = [True] * (2 + len(board)) + [False] * (5 - len(board))
    T = history_len
    toks = (rec.tokens[:T] + [0] * T)[:T]
    amts = (rec.amounts[:T] + [0] * T)[:T]
    o = 1 - seat
    raw = [
        info.pot,
        info.stacks[seat],
        info.stacks[o],
        info.street_bets[seat],
        info.street_bets[o],
        info.to_call,
        info.min_raise_to,
        info.max_raise_to,
    ]
    scal = (np.asarray(raw, dtype=np.float32) / start).tolist()
    street_oh = [1.0 if info.street == s else 0.0 for s in range(4)]
    is_button = 1.0 if seat == int(state.button) else 0.0
    raises = float(np.float32(rec.n_raises) / np.float32(max(1, sp.max_raises)))
    scalars = scal + street_oh + [is_button, raises]
    hist_amt = (np.asarray(amts, dtype=np.float32) / start).tolist()
    feats = {
        "cards": torch.tensor([cards], dtype=torch.long),
        "card_mask": torch.tensor([card_mask], dtype=torch.bool),
        "hist": torch.tensor([toks], dtype=torch.long),
        "hist_amt": torch.tensor([hist_amt], dtype=torch.float32),
        "scalars": torch.tensor([scalars], dtype=torch.float32),
        "legal": torch.tensor([info.legal], dtype=torch.bool),
    }
    extra = []
    if features.equity_samples > 0 or features.hist_runouts > 0:
        from ...env.equity import equity_histogram, equity_vs_random

        h = torch.tensor([hole], dtype=torch.long)
        b = feats["cards"][:, 2:]
        if features.equity_samples > 0:
            extra.append(equity_vs_random(h, b, features.equity_samples, generator)[:, None])
        if features.hist_runouts > 0:
            extra.append(
                equity_histogram(
                    h,
                    b,
                    features.hist_runouts,
                    features.hist_bins,
                    features.hist_opp_samples,
                    generator,
                )
            )
        feats["scalars"] = torch.cat([feats["scalars"]] + [e.float() for e in extra], 1)
    if features.strength_tables:
        from .strength import add_strength, load_strength

        feats = add_strength(feats, load_strength(features.strength_tables))
    return feats, info
