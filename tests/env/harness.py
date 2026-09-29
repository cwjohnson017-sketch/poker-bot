"""Random-play harness: drives VecNLHE, checks invariants, records hands."""

from __future__ import annotations

import torch

from pokerbot.env.actions import CHECK_CALL, FOLD, RAISE
from pokerbot.env.vec_env import LegalInfo, VecNLHE

from .scalar_rules import ScalarHand

LEGAL_KEYS = ("current_player", "street", "pot", "can_fold", "can_check", "call_amount", "min_raise_to", "max_raise_to")


def random_abstract(env: VecNLHE, g: torch.Generator) -> torch.Tensor:
    mask = env.legal_mask()
    return torch.multinomial(mask.float(), 1, generator=g).squeeze(1)


def random_concrete(info: LegalInfo, g: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    n = info.active.shape[0]
    u = torch.rand(n, generator=g)
    v = torch.rand(n, generator=g)
    fold = info.can_fold & (u < 0.15)
    rais = info.raise_ok & (u > 0.55)
    kind = torch.where(fold, FOLD, torch.where(rais, RAISE, CHECK_CALL))
    lo, hi = info.min_raise_to, info.max_raise_to
    amt = lo + ((hi - lo).clamp(min=0).float() * v**3).long()
    amt = torch.where((v > 0.9) | (hi < lo), hi, amt)
    return kind, torch.where(kind == RAISE, amt, 0)


def check_invariants(env: VecNLHE) -> None:
    start = env.start_stacks[None, :]
    live = ~env.done
    assert (env.stacks >= 0).all()
    assert (env.street_bets <= env.contrib).all()
    assert (env.street_bets >= 0).all()
    assert ((env.stacks + env.contrib) == start)[live].all(), "chips created or lost in a live hand"
    assert (env.payoffs.sum(1) == 0).all(), "payoffs do not sum to zero"
    assert (env.payoffs[live] == 0).all()
    assert (env.payoffs >= -start).all()
    assert (env.stacks == start + env.payoffs)[env.done].all()
    assert ((env.actor >= 0) == live).all()
    assert ((env.street >= 0) & (env.street <= 3)).all()
    assert (env.hist_len <= env.history_len).all()
    # all-in flags agree with empty stacks in live hands
    assert (env.all_in == (env.stacks == 0))[live].all()


def check_fresh_deal(env: VecNLHE, idx: torch.Tensor) -> None:
    if idx.numel() == 0:
        return
    deck = env.deck[idx].long()
    assert (deck.sort(1).values == torch.arange(52, device=deck.device)).all()
    assert (env.hist_len[idx] == 0).all() and (env.hist_tok[idx] == 0).all()
    assert (env.n_raises[idx] == 0).all()
    live = ~env.done[idx]
    assert (env.street[idx][live] == 0).all()
    btn = env.button[idx]
    sb_bet = env.street_bets[idx].gather(1, btn[:, None]).squeeze(1)
    bb_bet = env.street_bets[idx].gather(1, 1 - btn[:, None]).squeeze(1)
    full = (env.start_stacks.min() - env.ante) > env.bb
    if bool(full):
        assert (sb_bet == env.sb).all() and (bb_bet == env.bb).all()
        assert (env.actor[idx] == btn).all()
        assert live.all()


def _slot_record(env: VecNLHE, i: int, deck_rows, buttons) -> dict:
    return {"deck": deck_rows[i], "button": buttons[i], "actions": [], "legal": []}


def play_and_record(env: VecNLHE, steps: int, g: torch.Generator, concrete: bool = False) -> list[dict]:
    """Play random actions, check invariants every step, return finished hands."""
    n = env.n
    decks, buttons = env.deck.long().tolist(), env.button.tolist()
    cur = [_slot_record(env, i, decks, buttons) for i in range(n)]
    finished: list[dict] = []

    def finish(i: int) -> None:
        rec = cur[i]
        blen = int(env.board_len[i])
        rec["payoffs"] = env.payoffs[i].tolist()
        rec["board"] = env.cards[i, 4 : 4 + blen].tolist()
        finished.append(rec)

    for i in env.done.nonzero().squeeze(1).tolist():  # finished straight from the deal
        finish(i)
    for _ in range(steps):
        info = env.legal_info()
        snap = {
            "current_player": info.actor.tolist(),
            "street": info.street.tolist(),
            "pot": info.pot.tolist(),
            "can_fold": info.can_fold.tolist(),
            "can_check": info.can_check.tolist(),
            "call_amount": info.call_amount.tolist(),
            "min_raise_to": info.min_raise_to.tolist(),
            "max_raise_to": info.max_raise_to.tolist(),
        }
        live = info.active.tolist()
        if concrete:
            kind, amount = random_concrete(info, g)
            env.step_concrete(kind, amount, validate=True)
        else:
            env.step(random_abstract(env, g), validate=True)
        check_invariants(env)
        kinds, amts = env.last_kind.tolist(), env.last_amount.tolist()
        done = env.done.tolist()
        for i in range(n):
            if live[i]:
                cur[i]["actions"].append((kinds[i], amts[i]))
                cur[i]["legal"].append({k: snap[k][i] for k in LEGAL_KEYS})
                if done[i]:
                    finish(i)
        done_t = env.done.clone()
        env.reset(done_t)
        idx = done_t.nonzero().squeeze(1)
        check_fresh_deal(env, idx)
        if idx.numel():
            decks, buttons = env.deck.long().tolist(), env.button.tolist()
            for i in idx.tolist():
                cur[i] = _slot_record(env, i, decks, buttons)
                if bool(env.done[i]):
                    finish(i)
    return finished


def replay_scalar(hand: dict, config) -> None:
    """Replay a recorded hand in the independent scalar rules and compare."""
    h = ScalarHand(config.stacks, config.small_blind, config.big_blind, config.ante, hand["button"], hand["deck"])
    for (kind, amount), legal in zip(hand["actions"], hand["legal"]):
        assert not h.terminal
        assert h.legal() == legal, (h.legal(), legal, hand)
        h.apply(kind, amount)
    assert h.terminal
    assert h.pay == hand["payoffs"], (h.pay, hand)
    assert h.board() == hand["board"]
