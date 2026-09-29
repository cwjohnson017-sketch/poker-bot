"""Abstract action spec and its tensor mapping to concrete raise-to amounts.

A spec lists, per street, abstract actions of the forms

* ``("fold",)``
* ``("check_call",)``
* ``("raise", f)``   raise to ``max_bet + f * (pot + to_call)``: a bet or
  raise of ``f`` times the pot after calling (``f = 1`` is a pot raise);
* ``("raise_x", m)`` raise to ``m * max_bet`` (a multiple of the bet being
  faced; preflop opens and 3-bets like "2.5x" and "3x");
* ``("allin",)``.

``pot`` counts every chip committed so far, including the current street.
Fractions and multipliers are stored in thousandths and every amount is
computed in exact integer arithmetic so any engine can reproduce it:
``raise_to = max_bet + (f_milli * (pot + to_call) + 500) // 1000`` and
``raise_to = (m_milli * max_bet + 500) // 1000`` (round half up), then
clamped to ``[min_raise_to, max_raise_to]``; a size at or above the
player's stack becomes all-in, and when only an all-in for less than a
full raise is possible every raise action resolves to that all-in.

Legality of abstract actions (``legal_mask``):

* fold only when facing a bet; check/call always;
* raises need a legal raise and fewer than ``max_raises`` voluntary raises
  on this street; sized raises that would be all-in are left to ``allin``,
  and (with ``dedupe``) a sized raise whose amount equals an earlier sized
  raise's amount is masked, so every legal index is a distinct action.

The module is self-contained so it can move to ``pokerbot/abstraction``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch

# concrete action kinds (contract)
FOLD, CHECK_CALL, RAISE = 0, 1, 2

# abstract kind codes used in the tensor tables
K_INVALID, K_FOLD, K_CHECK_CALL, K_RAISE_POT, K_RAISE_MULT, K_ALLIN = -1, 0, 1, 2, 3, 4
_KIND_CODES = {"fold": K_FOLD, "check_call": K_CHECK_CALL, "raise": K_RAISE_POT, "raise_x": K_RAISE_MULT, "allin": K_ALLIN}

AbstractAction = tuple


@dataclass(frozen=True)
class ActionSpec:
    streets: tuple[tuple[AbstractAction, ...], ...]
    max_raises: int = 4  # voluntary raises per street (blinds do not count)
    dedupe: bool = True

    def __post_init__(self) -> None:
        if len(self.streets) != 4:
            raise ValueError("spec needs 4 streets")
        for st in self.streets:
            for a in st:
                if a[0] not in _KIND_CODES:
                    raise ValueError(f"unknown abstract action {a}")
                if a[0] in ("raise", "raise_x") and not (len(a) == 2 and a[1] > 0):
                    raise ValueError(f"bad size in {a}")
            names = [a[0] for a in st]
            if names.count("fold") != 1 or names.count("check_call") != 1:
                raise ValueError("every street needs exactly one fold and one check_call")

    @property
    def num_actions(self) -> int:
        """Width A of the abstract action axis (max over streets, padded)."""
        return max(len(s) for s in self.streets)

    def describe(self, street: int, index: int) -> str:
        a = self.streets[street][index]
        return a[0] if len(a) == 1 else f"{a[0]} {a[1]:g}"

    def tables(self, device: torch.device | str) -> "SpecTables":
        return SpecTables.build(self, device)


DEFAULT_SPEC = ActionSpec(
    streets=(
        (("fold",), ("check_call",), ("raise_x", 2.5), ("raise_x", 3.0), ("raise", 1.0), ("allin",)),
        (("fold",), ("check_call",), ("raise", 0.33), ("raise", 0.75), ("raise", 1.5), ("allin",)),
        (("fold",), ("check_call",), ("raise", 0.5), ("raise", 1.0), ("allin",)),
        (("fold",), ("check_call",), ("raise", 0.5), ("raise", 1.0), ("raise", 2.0), ("allin",)),
    )
)


@dataclass
class SpecTables:
    kind: torch.Tensor  # [4, A] abstract kind codes (K_*), K_INVALID for padding
    param: torch.Tensor  # [4, A] size in thousandths (0 when unused)
    concrete: torch.Tensor  # [4, A] concrete kind (FOLD/CHECK_CALL/RAISE; CHECK_CALL for padding)
    fold_index: torch.Tensor  # [4]
    call_index: torch.Tensor  # [4]
    tril: torch.Tensor  # [A, A] strictly lower triangular bool
    max_raises: int
    dedupe: bool
    num_actions: int = field(default=0)

    @staticmethod
    def build(spec: ActionSpec, device: torch.device | str) -> "SpecTables":
        A = spec.num_actions
        kind = torch.full((4, A), K_INVALID, dtype=torch.long)
        param = torch.zeros((4, A), dtype=torch.long)
        for s, st in enumerate(spec.streets):
            for i, a in enumerate(st):
                kind[s, i] = _KIND_CODES[a[0]]
                if len(a) == 2:
                    param[s, i] = int(round(float(a[1]) * 1000))
        concrete = torch.full((4, A), CHECK_CALL, dtype=torch.long)
        concrete[kind == K_FOLD] = FOLD
        concrete[(kind == K_RAISE_POT) | (kind == K_RAISE_MULT) | (kind == K_ALLIN)] = RAISE
        fold_index = (kind == K_FOLD).long().argmax(1)
        call_index = (kind == K_CHECK_CALL).long().argmax(1)
        tril = torch.ones(A, A, dtype=torch.bool).tril(-1)
        d = torch.device(device)
        return SpecTables(
            kind.to(d), param.to(d), concrete.to(d), fold_index.to(d), call_index.to(d), tril.to(d),
            spec.max_raises, spec.dedupe, A,
        )


def raise_targets(
    tab: SpecTables,
    street: torch.Tensor,
    pot: torch.Tensor,
    max_bet: torch.Tensor,
    to_call: torch.Tensor,
    min_raise_to: torch.Tensor,
    max_raise_to: torch.Tensor,
) -> torch.Tensor:
    """Concrete raise-to amount of every abstract action, ``[n, A]`` long.

    Entries for non-raise actions are meaningless (equal to the clamp range).
    ``street`` must already be clamped to 0..3.
    """
    kind = tab.kind[street]
    param = tab.param[street]
    pot_raise = max_bet[:, None] + torch.div(param * (pot + to_call)[:, None] + 500, 1000, rounding_mode="floor")
    mult_raise = torch.div(param * max_bet[:, None] + 500, 1000, rounding_mode="floor")
    raw = torch.where(kind == K_RAISE_POT, pot_raise, torch.where(kind == K_RAISE_MULT, mult_raise, max_raise_to[:, None]))
    return torch.minimum(torch.maximum(raw, min_raise_to[:, None]), max_raise_to[:, None])


def legal_mask(
    tab: SpecTables,
    street: torch.Tensor,
    active: torch.Tensor,
    to_call: torch.Tensor,
    raise_ok: torch.Tensor,
    n_raises: torch.Tensor,
    targets: torch.Tensor,
    max_raise_to: torch.Tensor,
) -> torch.Tensor:
    """``[n, A]`` bool. Inactive (finished) rows allow only check_call, a no-op."""
    kind = tab.kind[street]
    can_raise = (raise_ok & (n_raises < tab.max_raises))[:, None]
    fold_l = (kind == K_FOLD) & (to_call > 0)[:, None]
    call_l = kind == K_CHECK_CALL
    allin_l = (kind == K_ALLIN) & can_raise
    sized = (kind == K_RAISE_POT) | (kind == K_RAISE_MULT)
    sized_l = sized & can_raise & (targets < max_raise_to[:, None])
    if tab.dedupe:
        same = (targets[:, :, None] == targets[:, None, :]) & sized_l[:, None, :] & tab.tril
        sized_l = sized_l & ~same.any(2)
    mask = fold_l | call_l | allin_l | sized_l
    return torch.where(active[:, None], mask, call_l)


def nearest_abstract(
    tab: SpecTables, street: torch.Tensor, kind: torch.Tensor, amount: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """Abstract index to record for a concrete action (history tokens only).

    Fold/check-call map to their index; a raise maps to the raise-type entry
    whose concrete amount is closest (first on ties). This is not the
    pseudo-harmonic mapping of the abstraction layer.
    """
    akind = tab.kind[street]
    is_r = (akind == K_RAISE_POT) | (akind == K_RAISE_MULT) | (akind == K_ALLIN)
    dist = torch.where(is_r, (targets - amount[:, None]).abs(), torch.full_like(targets, 2**40))
    r_idx = dist.argmin(1)
    return torch.where(kind == FOLD, tab.fold_index[street], torch.where(kind == CHECK_CALL, tab.call_index[street], r_idx))


def spec_from_lists(streets: Sequence[Sequence[AbstractAction]], max_raises: int = 4, dedupe: bool = True) -> ActionSpec:
    return ActionSpec(tuple(tuple(tuple(a) for a in s) for s in streets), max_raises, dedupe)
