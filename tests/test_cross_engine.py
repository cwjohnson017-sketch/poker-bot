"""Reference vs Rust engine agreement. Skipped when ``poker_engine`` is not built."""

import numpy as np
import pytest

import pokerbot.reference as ref
from pokerbot.agents import AlwaysCallAgent, EquityThresholdAgent, RandomAgent
from pokerbot.eval import run_duplicate_match

pe = pytest.importorskip("poker_engine")


def _snap(s):
    out = dict(
        cur=s.current_player,
        street=s.street,
        board=list(s.board),
        stacks=list(s.stacks),
        bets=list(s.street_bets),
        pot=s.pot,
        term=s.is_terminal,
        folded=list(s.folded),
        all_in=list(s.all_in),
        hist=[(st, p, a.kind, a.amount) for st, p, a in s.history],
        pk=bytes(s.public_key()),
        ik=[bytes(s.infoset_key(i)) for i in range(s.num_players)],
    )
    if not s.is_terminal:
        la = s.legal_actions()
        out["legal"] = (
            la.can_fold,
            la.can_check,
            la.call_amount,
            la.min_raise_to,
            la.max_raise_to,
        )
    return out


def _mk(engine, kind, amount):
    if kind == 0:
        return engine.Action.fold()
    if kind == 1:
        return engine.Action.check_call()
    return engine.Action.raise_to(amount)


def test_random_play_agrees():
    rng = np.random.default_rng(0)
    for trial in range(1500):
        n = int(rng.integers(2, 10)) if trial % 2 else 2
        bb = int(rng.choice([2, 10, 100]))
        ante = int(rng.integers(0, 3)) if trial % 3 == 0 else 0
        hi = (300 if rng.random() < 0.5 else 8) * bb
        stacks = [int(x) for x in rng.integers(1, hi, size=n)]
        kw = dict(num_players=n, stacks=stacks, small_blind=bb // 2, big_blind=bb, ante=ante)
        button = int(rng.integers(n))
        deck = [int(c) for c in rng.permutation(52)]
        a = ref.GameState.new_hand(ref.GameConfig(**kw), button, deck)
        b = pe.GameState.new_hand(pe.GameConfig(**kw), button, deck)
        while True:
            assert _snap(a) == _snap(b), (kw, button, a.history)
            if a.is_terminal:
                assert a.payoffs() == list(b.payoffs())
                break
            la = a.legal_actions()
            opts = ["c", "c", "c"] + (["f"] if la.can_fold else [])
            if la.min_raise_to > 0:
                opts += ["r", "r"] + (["a"] if rng.random() < 0.3 else [])
            o = opts[rng.integers(len(opts))]
            if o == "c":
                act = (1, 0)
            elif o == "f":
                act = (0, 0)
            elif o == "a" or la.max_raise_to <= la.min_raise_to:
                act = (2, la.max_raise_to)
            else:
                top = min(la.max_raise_to, 3 * la.min_raise_to)
                act = (2, int(rng.integers(la.min_raise_to, top + 1)))
            a.apply(_mk(ref, *act))
            b.apply(_mk(pe, *act))


def test_evaluators_agree():
    rng = np.random.default_rng(1)
    arr = np.array([rng.permutation(52)[:7] for _ in range(50000)], dtype=np.uint8)
    ra = ref.evaluate_batch(arr).astype(np.int64)
    rb = np.asarray(pe.evaluate_batch(arr)).astype(np.int64)
    assert (np.sign(ra[1:] - ra[:-1]) == np.sign(rb[1:] - rb[:-1])).all()
    cats = np.array([pe.hand_category(int(x)) for x in rb[:5000]])
    assert (cats == (ra[:5000] >> 20)).all()


def test_match_runner_same_results_on_both_engines():
    res = {}
    for name, eng in (("ref", ref), ("rust", pe)):
        cfg = eng.GameConfig(num_players=2, stacks=[20000, 20000])
        r = run_duplicate_match(
            EquityThresholdAgent(samples=40), RandomAgent(), cfg, 20, seed=7, engine=eng
        )
        res[name] = r.seat_payoffs
    assert (res["ref"] == res["rust"]).all()


def test_masked_clone_works_with_rust_states():
    from pokerbot.eval import MaskedState

    cfg = pe.GameConfig(num_players=2, stacks=[20000, 20000])
    s = pe.GameState.new_hand(cfg, 0, list(range(52)))
    s.apply(pe.Action.check_call())
    s.apply(pe.Action.check_call())
    v = MaskedState(s, 0, cfg, np.random.default_rng(0))
    assert v.hole_cards(1) == []
    c = v.clone()
    assert list(c.hole_cards(0)) == list(s.hole_cards(0))
    assert list(c.board) == list(s.board)
    assert AlwaysCallAgent().act(v, 0, np.random.default_rng(0)).kind == 1
