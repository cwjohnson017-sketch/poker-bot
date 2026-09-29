"""Vectorized heads-up NLHE environment on torch tensors.

``VecNLHE(n, config, device, seed)`` keeps ``n`` independent heads-up hands
as tensors with a leading batch dimension. Every step is a fixed sequence of
masked tensor ops over all slots (finished slots are left untouched), with
no Python branching on tensor values unless ``validate`` is on. See
``pokerbot/env/README.md`` for the tensor layouts.

Rules (heads-up, matching the scalar engine contract):

* Deck rows are permutations of 0..51 dealt as P0 hole, P1 hole, flop,
  turn, river (seat order, not button order).
* The button posts the small blind and acts first preflop; the other seat
  acts first on the flop, turn and river. Antes go to the pot, not to the
  street bets.
* A raise is legal when the actor has more chips than the call and the
  opponent still has chips behind. ``min_raise_to = max_bet +
  max(last full raise increment on this street, big blind)``;
  ``max_raise_to`` is the actor's all-in. An all-in for less than a full
  raise does not change the min-raise increment.
* The betting round closes when each player has folded, is all-in, or has
  acted since the last raise and matched the largest bet. When it closes
  with at most one player able to bet, the board is run out and the hand
  goes to showdown in the same step.
* Payoffs are net chip changes: a fold loses the folder's contribution;
  at showdown the better hand wins ``min(contributions)`` (the uncalled
  excess goes back), ties split evenly (always an even pot heads-up).
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from . import actions as act_mod
from .actions import CHECK_CALL, DEFAULT_SPEC, FOLD, RAISE, ActionSpec
from .cards import NUM_CARDS, make_generator, shuffled_decks
from .config import GameConfig
from .evaluator import evaluate_batch

HISTORY_LEN = 24
SHOWDOWN_STREET = 3


@dataclass
class LegalInfo:
    """Per-slot legal action info for the current actor (all ``[n]`` tensors)."""

    active: torch.Tensor
    actor: torch.Tensor
    street: torch.Tensor  # clamped to 0..3
    pot: torch.Tensor
    my_bet: torch.Tensor
    my_stack: torch.Tensor
    max_bet: torch.Tensor
    to_call: torch.Tensor
    can_fold: torch.Tensor
    can_check: torch.Tensor
    call_amount: torch.Tensor
    raise_ok: torch.Tensor
    min_raise_to: torch.Tensor
    max_raise_to: torch.Tensor


class VecNLHE:
    _STATE = (
        "deck",
        "button",
        "stacks",
        "street_bets",
        "contrib",
        "street",
        "actor",
        "folded",
        "all_in",
        "acted",
        "last_raise",
        "n_raises",
        "hist_tok",
        "hist_amt",
        "hist_len",
        "done",
        "payoffs",
        "ranks",
        "last_kind",
        "last_amount",
    )

    def __init__(
        self,
        n: int,
        config: Any = None,
        device: torch.device | str = "cpu",
        seed: int = 0,
        spec: ActionSpec = DEFAULT_SPEC,
        validate: bool = True,
        history_len: int = HISTORY_LEN,
        auto_deal: bool = True,
    ) -> None:
        config = config if config is not None else GameConfig()
        if getattr(config, "num_players", 2) != 2 or len(config.stacks) != 2:
            raise ValueError("VecNLHE is heads-up only")
        self.n = int(n)
        self.config = config
        self.device = torch.device(device)
        self.sb = int(config.small_blind)
        self.bb = int(config.big_blind)
        self.ante = int(getattr(config, "ante", 0))
        self.spec = spec
        self.tab = spec.tables(self.device)
        self.num_actions = spec.num_actions
        self.history_len = int(history_len)
        self.validate = validate
        self.generator = make_generator(seed, self.device)
        dev, n, T = self.device, self.n, self.history_len
        L = dict(dtype=torch.long, device=dev)
        B = dict(dtype=torch.bool, device=dev)
        self.start_stacks = torch.tensor([int(s) for s in config.stacks], **L)  # [2]
        self._seat = torch.arange(2, **L)
        self._hist_pos = torch.arange(T, **L)
        self.deck = torch.zeros(n, NUM_CARDS, dtype=torch.uint8, device=dev)
        self.button = (torch.arange(n, **L) + 1) % 2  # first deal toggles -> slot % 2
        self.stacks = torch.zeros(n, 2, **L)
        self.street_bets = torch.zeros(n, 2, **L)
        self.contrib = torch.zeros(n, 2, **L)
        self.street = torch.zeros(n, **L)
        self.actor = torch.full((n,), -1, **L)
        self.folded = torch.zeros(n, 2, **B)
        self.all_in = torch.zeros(n, 2, **B)
        self.acted = torch.zeros(n, 2, **B)
        self.last_raise = torch.zeros(n, **L)
        self.n_raises = torch.zeros(n, **L)
        self.hist_tok = torch.zeros(n, T, **L)
        self.hist_amt = torch.zeros(n, T, **L)
        self.hist_len = torch.zeros(n, **L)
        self.done = torch.ones(n, **B)
        self.payoffs = torch.zeros(n, 2, **L)
        self.ranks = torch.zeros(n, 2, **L)
        self.last_kind = torch.full((n,), -1, **L)
        self.last_amount = torch.zeros(n, **L)
        if auto_deal:
            self.reset()

    # ------------------------------------------------------------------ views
    @property
    def cards(self) -> torch.Tensor:
        """``[n, 9]``: P0 hole, P1 hole, flop, turn, river (whole deal, incl. undealt)."""
        return self.deck[:, :9].long()

    @property
    def pot(self) -> torch.Tensor:
        return self.contrib.sum(1)

    @property
    def board_len(self) -> torch.Tensor:
        return torch.where(self.street == 0, 0, self.street + 2)

    @property
    def vocab_size(self) -> int:
        return 1 + 8 * self.num_actions

    # ------------------------------------------------------------------ dealing
    @torch.no_grad()
    def reset(
        self, mask: torch.Tensor | None = None, button: torch.Tensor | int | None = None
    ) -> None:
        """Deal new hands into the masked slots (all slots when ``mask`` is None).

        Without ``button`` each reset slot's button alternates from its
        previous hand. Uses one host sync (the number of slots to deal).
        """
        if mask is None:
            idx = torch.arange(self.n, device=self.device)
        else:
            idx = mask.to(self.device).nonzero().squeeze(1)
        k = idx.numel()
        if k == 0:
            return
        decks = shuffled_decks(k, self.generator)
        if button is None:
            btn = 1 - self.button[idx]
        elif isinstance(button, int):
            btn = torch.full((k,), int(button), dtype=torch.long, device=self.device)
        else:  # per-slot tensor [n]
            btn = torch.as_tensor(button, dtype=torch.long, device=self.device).reshape(self.n)[idx]
        self._deal(idx, decks, btn)

    def _deal(self, idx: torch.Tensor, decks: torch.Tensor, btn: torch.Tensor) -> None:
        k = idx.numel()
        dev = self.device
        decks = decks.to(dev).long()
        btn = btn.to(dev).long()
        cards = decks[:, :9]
        board = cards[:, 4:9]
        hands = torch.stack(
            [torch.cat([cards[:, 0:2], board], 1), torch.cat([cards[:, 2:4], board], 1)], 1
        )
        ranks = evaluate_batch(hands)  # [k, 2]

        start = self.start_stacks.expand(k, 2)
        ante = torch.clamp(start, max=self.ante)
        stacks = start - ante
        is_sb = self._seat[None, :] == btn[:, None]
        blind = torch.where(is_sb, torch.full_like(start, self.sb), torch.full_like(start, self.bb))
        blind = torch.minimum(blind, stacks)
        stacks = stacks - blind
        all_in = stacks == 0
        zeros_k = torch.zeros(k, dtype=torch.long, device=dev)
        # button acts first preflop unless already all-in from the blind
        btn_allin = all_in.gather(1, btn[:, None]).squeeze(1)
        actor = torch.where(btn_allin, 1 - btn, btn)

        self.deck[idx] = decks.to(torch.uint8)
        self.button[idx] = btn
        self.stacks[idx] = stacks
        self.street_bets[idx] = blind
        self.contrib[idx] = ante + blind
        self.street[idx] = zeros_k
        self.actor[idx] = actor
        self.folded[idx] = False
        self.all_in[idx] = all_in
        self.acted[idx] = False
        self.last_raise[idx] = torch.full_like(zeros_k, self.bb)
        self.n_raises[idx] = zeros_k
        self.hist_tok[idx] = 0
        self.hist_amt[idx] = 0
        self.hist_len[idx] = zeros_k
        self.done[idx] = False
        self.payoffs[idx] = 0
        self.ranks[idx] = ranks
        self.last_kind[idx] = -1
        self.last_amount[idx] = zeros_k
        # both players all-in from the blinds: nothing to decide, run it out
        both = torch.zeros(self.n, dtype=torch.bool, device=dev)
        both[idx] = all_in.all(1)
        self._showdown(both)

    # ------------------------------------------------------------------ legality
    @torch.no_grad()
    def legal_info(self) -> LegalInfo:
        active = ~self.done
        p = self.actor.clamp(min=0)[:, None]
        o = 1 - p
        my_bet = self.street_bets.gather(1, p).squeeze(1)
        opp_bet = self.street_bets.gather(1, o).squeeze(1)
        my_stack = self.stacks.gather(1, p).squeeze(1)
        opp_stack = self.stacks.gather(1, o).squeeze(1)
        max_bet = torch.maximum(my_bet, opp_bet)
        to_call = max_bet - my_bet
        call_amount = torch.minimum(to_call, my_stack)
        raise_ok = active & (my_stack > to_call) & (opp_stack > 0)
        zero = torch.zeros_like(max_bet)
        min_rt = torch.where(raise_ok, max_bet + torch.clamp(self.last_raise, min=self.bb), zero)
        max_rt = torch.where(raise_ok, my_bet + my_stack, zero)
        return LegalInfo(
            active=active,
            actor=torch.where(active, self.actor, -1),
            street=self.street.clamp(0, 3),
            pot=self.contrib.sum(1),
            my_bet=my_bet,
            my_stack=my_stack,
            max_bet=max_bet,
            to_call=to_call,
            can_fold=active & (to_call > 0),
            can_check=active & (to_call == 0),
            call_amount=torch.where(active, call_amount, zero),
            raise_ok=raise_ok,
            min_raise_to=min_rt,
            max_raise_to=max_rt,
        )

    def _targets(self, info: LegalInfo) -> torch.Tensor:
        return act_mod.raise_targets(
            self.tab,
            info.street,
            info.pot,
            info.max_bet,
            info.to_call,
            info.min_raise_to,
            info.max_raise_to,
        )

    def _mask(self, info: LegalInfo, targets: torch.Tensor) -> torch.Tensor:
        return act_mod.legal_mask(
            self.tab,
            info.street,
            info.active,
            info.to_call,
            info.raise_ok,
            self.n_raises,
            targets,
            info.max_raise_to,
        )

    @torch.no_grad()
    def legal_mask(self) -> torch.Tensor:
        """``[n, A]`` bool over the abstract actions of each slot's street.

        Finished slots allow only check_call, which is a no-op for them.
        """
        info = self.legal_info()
        return self._mask(info, self._targets(info))

    @torch.no_grad()
    def action_amounts(self) -> torch.Tensor:
        """``[n, A]`` concrete raise-to amount for every abstract raise entry."""
        info = self.legal_info()
        return self._targets(info)

    # ------------------------------------------------------------------ stepping
    @torch.no_grad()
    def step(
        self, abstract_action: torch.Tensor, validate: bool | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply one abstract action per slot for the slot's current actor.

        Returns ``(payoffs [n, 2], done [n])``. ``payoffs`` holds the net chip
        result of every finished slot (zero while a hand runs) and stays set
        until the slot is reset; finished slots ignore their action. With
        ``validate`` an illegal action on a live slot raises ``ValueError``
        (one host sync); without it illegal actions become check/call.
        """
        validate = self.validate if validate is None else validate
        a = abstract_action.to(self.device).long().reshape(self.n)
        info = self.legal_info()
        targets = self._targets(info)
        mask = self._mask(info, targets)
        in_range = (a >= 0) & (a < self.num_actions)
        a_safe = a.clamp(0, self.num_actions - 1)
        legal = in_range & mask.gather(1, a_safe[:, None]).squeeze(1)
        if validate and bool((info.active & ~legal).any()):
            bad = (info.active & ~legal).nonzero().squeeze(1)[:8].tolist()
            raise ValueError(f"illegal abstract action in slots {bad}")
        a_safe = torch.where(legal, a_safe, self.tab.call_index[info.street])
        kind = self.tab.concrete[info.street].gather(1, a_safe[:, None]).squeeze(1)
        amount = targets.gather(1, a_safe[:, None]).squeeze(1)
        amount = torch.where(kind == RAISE, amount, torch.zeros_like(amount))
        self._apply(kind, amount, a_safe, info)
        return self.payoffs.clone(), self.done.clone()

    @torch.no_grad()
    def step_concrete(
        self, kind: torch.Tensor, amount: torch.Tensor, validate: bool | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply concrete actions (``FOLD``/``CHECK_CALL``/``RAISE`` + raise-to amount).

        With ``validate`` an illegal action on a live slot raises
        ``ValueError``; without it illegal actions become check/call.
        """
        validate = self.validate if validate is None else validate
        kind = kind.to(self.device).long().reshape(self.n)
        amount = amount.to(self.device).long().reshape(self.n)
        info = self.legal_info()
        in_raise_range = (amount >= info.min_raise_to) & (amount <= info.max_raise_to)
        raise_legal = info.raise_ok & (in_raise_range | (amount == info.max_raise_to))
        legal = torch.where(
            kind == FOLD,
            info.can_fold,
            torch.where(kind == CHECK_CALL, info.active, (kind == RAISE) & raise_legal),
        )
        if validate and bool((info.active & ~legal).any()):
            bad = (info.active & ~legal).nonzero().squeeze(1)[:8].tolist()
            raise ValueError(f"illegal concrete action in slots {bad}")
        kind = torch.where(legal, kind, torch.full_like(kind, CHECK_CALL))
        amount = torch.where(kind == RAISE, amount, torch.zeros_like(amount))
        targets = self._targets(info)
        a_idx = act_mod.nearest_abstract(self.tab, info.street, kind, amount, targets)
        self._apply(kind, amount, a_idx, info)
        return self.payoffs.clone(), self.done.clone()

    def _apply(
        self, kind: torch.Tensor, amount: torch.Tensor, a_idx: torch.Tensor, info: LegalInfo
    ) -> None:
        live = info.active
        p = self.actor.clamp(min=0)
        sel = self._seat[None, :] == p[:, None]  # [n, 2] actor one-hot
        is_fold = live & (kind == FOLD)
        is_call = live & (kind == CHECK_CALL)
        is_raise = live & (kind == RAISE)
        zero = torch.zeros_like(amount)
        add = torch.where(
            is_call, info.call_amount, torch.where(is_raise, amount - info.my_bet, zero)
        )
        add2 = sel.long() * add[:, None]
        self.stacks -= add2
        self.street_bets += add2
        self.contrib += add2

        increment = amount - info.max_bet
        full_raise = is_raise & (increment >= torch.clamp(self.last_raise, min=self.bb))
        self.last_raise = torch.where(full_raise, increment, self.last_raise)
        self.n_raises = self.n_raises + is_raise.long()
        acted_now = sel & live[:, None]
        reopened = ~sel & is_raise[:, None]
        self.acted = (self.acted | acted_now) & ~reopened
        self.all_in = self.all_in | (acted_now & (self.stacks == 0))
        self.folded = self.folded | (sel & is_fold[:, None])

        # history token: 1 + (street * 2 + actor_is_button) * A + abstract index
        pos = (p == self.button).long()
        tok = 1 + (info.street * 2 + pos) * self.num_actions + a_idx
        write = live & (self.hist_len < self.history_len)
        at = (self._hist_pos[None, :] == self.hist_len[:, None]) & write[:, None]
        self.hist_tok = torch.where(at, tok[:, None], self.hist_tok)
        self.hist_amt = torch.where(at, add[:, None], self.hist_amt)
        self.hist_len = self.hist_len + write.long()
        self.last_kind = torch.where(live, kind, self.last_kind)
        self.last_amount = torch.where(live, amount, self.last_amount)

        # betting round closure
        max_bet = self.street_bets.max(1, keepdim=True).values
        settled = self.folded | self.all_in | (self.acted & (self.street_bets == max_bet))
        closed = live & ~is_fold & settled.all(1)
        can_bet = (~self.folded & ~self.all_in).sum(1)
        runout = closed & (can_bet <= 1)
        showdown = runout | (closed & (self.street == SHOWDOWN_STREET))
        next_street = closed & ~showdown

        self.street = torch.where(next_street, self.street + 1, self.street)
        ns = next_street[:, None]
        self.street_bets = torch.where(ns, 0, self.street_bets)
        self.acted = self.acted & ~ns
        self.last_raise = torch.where(next_street, 0, self.last_raise)
        self.n_raises = torch.where(next_street, 0, self.n_raises)
        self.actor = torch.where(live, torch.where(closed, 1 - self.button, 1 - p), self.actor)

        # fold: folder loses its contribution, the other seat wins it
        folder_c = self.contrib.gather(1, p[:, None]).squeeze(1)
        fold_pay = torch.where(sel, -folder_c[:, None], folder_c[:, None])
        self.payoffs = torch.where(is_fold[:, None], fold_pay, self.payoffs)
        self._finish(is_fold)
        self._showdown(showdown)

    def _showdown(self, mask: torch.Tensor) -> None:
        stake = self.contrib.min(1).values
        r0, r1 = self.ranks[:, 0], self.ranks[:, 1]
        pay0 = torch.where(r0 > r1, stake, torch.where(r1 > r0, -stake, torch.zeros_like(stake)))
        pay = torch.stack([pay0, -pay0], 1)
        self.payoffs = torch.where(mask[:, None], pay, self.payoffs)
        self.street = torch.where(mask, SHOWDOWN_STREET, self.street)
        self._finish(mask)

    def _finish(self, mask: torch.Tensor) -> None:
        m = mask[:, None]
        self.stacks = torch.where(m, self.start_stacks[None, :] + self.payoffs, self.stacks)
        self.actor = torch.where(mask, -1, self.actor)
        self.done = self.done | mask

    # ------------------------------------------------------------------ observation
    @torch.no_grad()
    def obs(self, **kwargs: Any) -> dict[str, torch.Tensor]:
        """Acting player's view; see :func:`pokerbot.env.obs.encode_obs`."""
        from .obs import encode_obs

        return encode_obs(self, **kwargs)

    # ------------------------------------------------------------------ batch utilities
    def select(self, idx: torch.Tensor) -> VecNLHE:
        """New env holding copies of the slots ``idx`` (may repeat, e.g. to fan
        out a frontier). Shares the random generator with ``self``."""
        idx = idx.to(self.device).long()
        out = copy.copy(self)
        out.n = int(idx.numel())
        for name in self._STATE:
            setattr(out, name, getattr(self, name)[idx].clone())
        return out

    def clone(self) -> VecNLHE:
        out = copy.copy(self)
        for name in self._STATE:
            setattr(out, name, getattr(self, name).clone())
        return out

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {name: getattr(self, name) for name in self._STATE}

    # ------------------------------------------------------------------ scalar cross-check hook
    @torch.no_grad()
    def replay_check(
        self, deck: Sequence[int], actions: Sequence[Any], button: int = 0
    ) -> dict[str, Any]:
        """Run one hand through the tensor code path (test hook).

        ``deck`` is a full 52-card permutation, ``actions`` a list of concrete
        ``(kind, amount)`` pairs or objects with ``.kind``/``.amount``. Raises
        ``ValueError`` on an illegal action or an action after the hand ended.
        Returns ``payoffs``, ``board`` (visible at the end), ``terminal``,
        ``street``, ``pot``, ``stacks`` and ``legal``: one dict per action with
        the legal info the actor faced before acting.
        """
        deck = [int(c) for c in deck]
        if sorted(deck) != list(range(NUM_CARDS)):
            raise ValueError("deck must be a permutation of 0..51")
        env = VecNLHE(
            1, self.config, self.device, 0, self.spec, True, self.history_len, auto_deal=False
        )
        dev = self.device
        env._deal(
            torch.zeros(1, dtype=torch.long, device=dev),
            torch.tensor([deck], dtype=torch.long, device=dev),
            torch.tensor([int(button)], dtype=torch.long, device=dev),
        )
        steps = []
        for a in actions:
            kind, amount = (
                (a.kind, a.amount) if hasattr(a, "kind") else (a[0], a[1] if len(a) > 1 else 0)
            )
            if bool(env.done[0]):
                raise ValueError("action after the hand ended")
            steps.append(env.slot_info(0))
            env.step_concrete(
                torch.tensor([int(kind)], device=dev),
                torch.tensor([int(amount or 0)], device=dev),
                validate=True,
            )
        blen = int(env.board_len[0])
        return {
            "payoffs": env.payoffs[0].tolist() if bool(env.done[0]) else None,
            "board": env.cards[0, 4 : 4 + blen].tolist(),
            "terminal": bool(env.done[0]),
            "street": int(env.street[0]),
            "pot": int(env.pot[0]),
            "stacks": env.stacks[0].tolist(),
            "street_bets": env.street_bets[0].tolist(),
            "current_player": int(env.actor[0]),
            "legal": steps,
            "final": env.slot_info(0),
        }

    def slot_info(self, i: int) -> dict[str, Any]:
        """Python-side legal info of one slot (for tests and debugging)."""
        info = self.legal_info()
        return {
            "current_player": int(info.actor[i]),
            "street": int(self.street[i]),
            "pot": int(info.pot[i]),
            "stacks": self.stacks[i].tolist(),
            "street_bets": self.street_bets[i].tolist(),
            "can_fold": bool(info.can_fold[i]),
            "can_check": bool(info.can_check[i]),
            "call_amount": int(info.call_amount[i]),
            "min_raise_to": int(info.min_raise_to[i]),
            "max_raise_to": int(info.max_raise_to[i]),
        }
