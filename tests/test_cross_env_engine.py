"""Torch vectorized env vs Rust engine agreement.

The Rust engine (``poker_engine``) is the source of truth for the rules. Random
heads-up hands are played on the Rust ``GameState`` with random concrete
actions; the same deck, button and actions are then replayed through
``VecNLHE.replay_check`` (one hand at a time) and through ``step_concrete`` on
a batch env, and the legal info before every action, the final payoffs, the
board and the terminal street must agree exactly. Skipped when
``poker_engine`` is not built.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pokerbot.env import GameConfig, VecNLHE

pe = pytest.importorskip("poker_engine")

FOLD, CHECK_CALL, RAISE = 0, 1, 2
LEGAL_KEYS = (
    "current_player",
    "street",
    "pot",
    "can_fold",
    "can_check",
    "call_amount",
    "min_raise_to",
    "max_raise_to",
)


# ---------------------------------------------------------------------- setups
def _setups(rng: np.random.Generator) -> list[tuple[str, tuple[list[int], int, int, int], int]]:
    """``(name, (stacks, sb, bb, ante), hands)``; hands share one config so
    they can also be played as one batch."""
    out = [
        ("equal_deep", ([20000, 20000], 50, 100, 0), 400),
        ("unequal", ([20000, 7350], 50, 100, 0), 250),
        ("unequal_rev", ([1234, 20000], 50, 100, 0), 150),
        ("short_below_bb", ([60, 20000], 50, 100, 0), 80),
        ("short_below_sb", ([30, 20000], 50, 100, 0), 40),
        ("both_short", ([70, 40], 50, 100, 0), 80),
        ("both_blinds_all_in", ([50, 100], 50, 100, 0), 40),
        ("exact_blinds", ([100, 100], 50, 100, 0), 50),
        ("ante", ([10000, 10000], 50, 100, 10), 200),
        ("ante_short", ([10, 250], 50, 100, 10), 60),
        ("ante_bb_short", ([5000, 105], 50, 100, 10), 60),
        ("sb_equals_bb", ([3000, 3000], 100, 100, 0), 80),
        ("zero_sb", ([3000, 900], 0, 100, 5), 80),
        ("tiny_blinds", ([37, 23], 1, 2, 0), 100),
    ]
    for k in range(24):
        bb = int(rng.choice([2, 10, 100]))
        sb = bb // 2
        ante = int(rng.integers(0, bb // 2 + 1)) if k % 3 == 0 else 0
        stacks = [int(x) for x in rng.integers(1, 300 * bb + 1, size=2)]
        out.append((f"random_{k}", (stacks, sb, bb, ante), 30))
    return out


# ---------------------------------------------------------------------- Rust play
def _legal(s) -> dict:
    la = s.legal_actions()
    return {
        "current_player": s.current_player,
        "street": s.street,
        "pot": s.pot,
        "can_fold": la.can_fold,
        "can_check": la.can_check,
        "call_amount": la.call_amount,
        "min_raise_to": la.min_raise_to,
        "max_raise_to": la.max_raise_to,
    }


def _pick(lg: dict, rng: np.random.Generator) -> tuple[int, int]:
    """A random legal concrete action: fold, check/call, or a raise-to that is
    the minimum, all-in, all-in for less, a random size or an odd size."""
    opts = ["c", "c", "c"] + (["f"] if lg["can_fold"] else [])
    lo, hi = lg["min_raise_to"], lg["max_raise_to"]
    if lo > 0:
        opts += ["min", "allin", "rand", "rand", "odd"]
    o = opts[rng.integers(len(opts))]
    if o == "c":
        return CHECK_CALL, 0
    if o == "f":
        return FOLD, 0
    if o == "allin" or hi <= lo:  # all-in, possibly for less than a full raise
        return RAISE, hi
    if o == "min":
        return RAISE, lo
    if o == "odd":  # off-size amounts next to the bounds
        return RAISE, int(rng.choice([lo + 1, hi - 1, (lo + hi) // 2 + 1]))
    top = min(hi, lo + int(rng.integers(0, 4)) * lo)
    return RAISE, int(rng.integers(lo, top + 1))


def _illegal(lg: dict, rng: np.random.Generator) -> tuple[int, int] | None:
    """An illegal concrete action for this spot, or None when none is handy."""
    cands = []
    if not lg["can_fold"]:
        cands.append((FOLD, 0))
    lo, hi = lg["min_raise_to"], lg["max_raise_to"]
    if lo == 0:
        cands.append((RAISE, lg["call_amount"] + 10**6))
    else:
        cands.append((RAISE, hi + 1))
        if lo - 1 > 0 and lo - 1 != hi and lo <= hi:
            cands.append((RAISE, lo - 1))
    return cands[rng.integers(len(cands))] if cands else None


def _mk(kind: int, amount: int):
    if kind == FOLD:
        return pe.Action.fold()
    if kind == CHECK_CALL:
        return pe.Action.check_call()
    return pe.Action.raise_to(amount)


def _play_rust(cfg, button: int, deck: list[int], rng: np.random.Generator) -> dict:
    s = pe.GameState.new_hand(pe.GameConfig(**cfg), button, deck)
    assert list(s.hole_cards(0)) == deck[0:2] and list(s.hole_cards(1)) == deck[2:4]
    rec = {"deck": deck, "button": button, "actions": [], "legal": [], "illegal": None}
    while not s.is_terminal:
        lg = _legal(s)
        if rec["illegal"] is None and rng.random() < 0.05:
            bad = _illegal(lg, rng)
            if bad is not None:
                with pytest.raises(ValueError):
                    s.apply(_mk(*bad))
                rec["illegal"] = (len(rec["actions"]), bad)
        a = _pick(lg, rng)
        rec["legal"].append(lg)
        rec["actions"].append(a)
        s.apply(_mk(*a))
    rec["payoffs"] = list(s.payoffs())
    rec["board"] = list(s.board)
    rec["street"] = s.street
    rec["pot"] = s.pot
    rec["contributed"] = list(s.contributed)
    rec["stacks"] = list(s.stacks)
    return rec


def _hands(seed: int = 0):
    rng = np.random.default_rng(seed)
    for name, (stacks, sb, bb, ante), k in _setups(rng):
        cfg = dict(num_players=2, stacks=stacks, small_blind=sb, big_blind=bb, ante=ante)
        recs = []
        for _ in range(k):
            deck = [int(c) for c in rng.permutation(52)]
            recs.append(_play_rust(cfg, int(rng.integers(2)), deck, rng))
        yield name, cfg, recs


@pytest.fixture(scope="module")
def played():
    # batch-of-one tensors: extra intra-op threads only add overhead
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield list(_hands())
    finally:
        torch.set_num_threads(threads)


# ---------------------------------------------------------------------- tests
def test_replay_check_agrees(played):
    total = 0
    for name, cfg, recs in played:
        env = VecNLHE(1, GameConfig(**cfg), "cpu", 0, auto_deal=False)
        for rec in recs:
            r = env.replay_check(rec["deck"], rec["actions"], rec["button"])
            ctx = (name, cfg, rec["button"], rec["actions"])
            assert len(r["legal"]) == len(rec["legal"]), ctx
            for i, (got, want) in enumerate(zip(r["legal"], rec["legal"], strict=True)):
                assert {k: got[k] for k in LEGAL_KEYS} == want, (i, ctx)
            assert r["terminal"], ctx
            assert r["payoffs"] == rec["payoffs"], ctx
            assert r["board"] == rec["board"], ctx
            assert r["street"] == rec["street"], ctx
            assert r["pot"] == rec["pot"], ctx
            assert r["current_player"] == -1, ctx
            # env convention: terminal stacks are the final stacks
            final = [
                s + c + p
                for s, c, p in zip(rec["stacks"], rec["contributed"], rec["payoffs"], strict=True)
            ]
            assert r["stacks"] == final, ctx
            if rec["illegal"] is not None:
                at, bad = rec["illegal"]
                with pytest.raises(ValueError):
                    env.replay_check(rec["deck"], rec["actions"][:at] + [bad], rec["button"])
            total += 1
    assert total >= 2000


def test_batched_step_concrete_agrees(played):
    for name, cfg, recs in played:
        n = len(recs)
        env = VecNLHE(n, GameConfig(**cfg), "cpu", 0, auto_deal=False)
        env._deal(
            torch.arange(n),
            torch.tensor([r["deck"] for r in recs]),
            torch.tensor([r["button"] for r in recs]),
        )
        lengths = torch.tensor([len(r["actions"]) for r in recs])
        steps = int(lengths.max())
        acts = torch.zeros(n, max(steps, 1), 2, dtype=torch.long)
        for i, r in enumerate(recs):
            if r["actions"]:
                acts[i, : len(r["actions"])] = torch.tensor(r["actions"])
        assert (env.done == (lengths == 0)).all(), name
        for t in range(steps):
            live = ~env.done
            assert (live == (lengths > t)).all(), (name, t)
            info = env.legal_info()
            for i in live.nonzero().squeeze(1).tolist():
                want = recs[i]["legal"][t]
                got = {
                    "current_player": int(info.actor[i]),
                    "street": int(info.street[i]),
                    "pot": int(info.pot[i]),
                    "can_fold": bool(info.can_fold[i]),
                    "can_check": bool(info.can_check[i]),
                    "call_amount": int(info.call_amount[i]),
                    "min_raise_to": int(info.min_raise_to[i]),
                    "max_raise_to": int(info.max_raise_to[i]),
                }
                assert got == want, (name, i, t, recs[i]["actions"])
            env.step_concrete(acts[:, t, 0], acts[:, t, 1], validate=True)
        assert env.done.all(), name
        assert env.payoffs.tolist() == [r["payoffs"] for r in recs], name
        assert env.street.tolist() == [r["street"] for r in recs], name
