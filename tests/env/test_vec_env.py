import pytest
import torch

from pokerbot.env import DEFAULT_SPEC, GameConfig, VecNLHE
from pokerbot.env.actions import K_ALLIN, K_RAISE_MULT, K_RAISE_POT, ActionSpec
from pokerbot.env.cards import NO_CARD
from pokerbot.env.obs import NUM_SCALARS

from .harness import check_invariants, play_and_record, random_abstract, replay_scalar

N = 512


def test_random_abstract_play_invariants_and_scalar_agreement():
    cfg = GameConfig()
    env = VecNLHE(N, cfg, "cpu", seed=11)
    g = torch.Generator().manual_seed(1)
    hands = play_and_record(env, 150, g, concrete=False)
    assert len(hands) > 1000
    for h in hands:
        replay_scalar(h, cfg)
    # the abstraction never needs more than 24 history tokens
    assert max(len(h["actions"]) for h in hands) <= 24
    streets_seen = {len(h["board"]) for h in hands}
    assert streets_seen == {0, 3, 4, 5}


@pytest.mark.parametrize(
    "stacks,ante,steps",
    [
        ([3000, 20000], 0, 60),
        ([20000, 5000], 10, 60),
        ([260, 1000], 5, 30),
        ([60, 20000], 0, 12),  # SB nearly all-in from the blind
        ([40, 100], 0, 3),  # both all-in from the blinds: hands end at the deal
    ],
)
def test_random_concrete_play_matches_scalar_rules(stacks, ante, steps):
    cfg = GameConfig(stacks=stacks, ante=ante)
    env = VecNLHE(N, cfg, "cpu", seed=5)
    g = torch.Generator().manual_seed(2)
    hands = play_and_record(env, steps, g, concrete=True)
    assert len(hands) > 200
    for h in hands:
        replay_scalar(h, cfg)


def test_payoffs_sum_to_zero_many_steps():
    env = VecNLHE(N, GameConfig(), "cpu", seed=3, validate=False)
    g = torch.Generator().manual_seed(3)
    total = torch.zeros(2, dtype=torch.long)
    hands = 0
    for _ in range(400):
        pay, done = env.step(random_abstract(env, g))
        check_invariants(env)
        assert torch.equal(pay[~done], torch.zeros_like(pay[~done]))
        total += pay[done].sum(0)
        hands += int(done.sum())
        env.reset(done)
    assert hands > 5000
    assert int(total.sum()) == 0


def test_every_legal_action_applies():
    env = VecNLHE(N, GameConfig(stacks=[20000, 7000]), "cpu", seed=8)
    g = torch.Generator().manual_seed(8)
    call_idx = env.tab.call_index
    for _ in range(12):
        info = env.legal_info()
        mask = env.legal_mask()
        targets = env.action_amounts()
        kinds = env.tab.kind[info.street]
        for a in range(env.num_actions):
            e = env.clone()
            use = mask[:, a]
            acts = torch.where(use, a, call_idx[info.street])
            e.step(acts, validate=True)
            check_invariants(e)
            is_raise = use & info.active & ((kinds[:, a] == K_RAISE_POT) | (kinds[:, a] == K_RAISE_MULT) | (kinds[:, a] == K_ALLIN))
            if is_raise.any():
                p = env.actor[is_raise]
                bet_after = e.street_bets[is_raise].gather(1, p[:, None]).squeeze(1)
                assert torch.equal(bet_after, targets[is_raise, a])
                lo, hi = info.min_raise_to[is_raise], info.max_raise_to[is_raise]
                t = targets[is_raise, a]
                assert (((t >= lo) & (t <= hi)) | (t == hi)).all()
                assert (e.actor[is_raise] == 1 - p).all()  # a raise never closes the round
        env.step(random_abstract(env, g))
        env.reset(env.done)


def test_legal_sized_raises_are_distinct_and_below_allin():
    env = VecNLHE(N, GameConfig(), "cpu", seed=4)
    g = torch.Generator().manual_seed(4)
    for _ in range(40):
        info = env.legal_info()
        mask = env.legal_mask()
        t = env.action_amounts()
        kinds = env.tab.kind[info.street]
        sized = mask & ((kinds == K_RAISE_POT) | (kinds == K_RAISE_MULT))
        assert (t[sized] < info.max_raise_to[:, None].expand_as(t)[sized]).all()
        for i in sized.any(1).nonzero().squeeze(1).tolist()[:64]:
            vals = t[i][sized[i]].tolist()
            assert len(set(vals)) == len(vals)
        # fold is legal exactly when facing a bet, check/call always
        fold_col = env.tab.fold_index[info.street]
        assert torch.equal(mask.gather(1, fold_col[:, None]).squeeze(1), info.can_fold)
        assert mask.gather(1, env.tab.call_index[info.street][:, None]).all()
        env.step(random_abstract(env, g))
        env.reset(env.done)


def test_illegal_actions_raise_or_fall_back_to_call():
    env = VecNLHE(4, GameConfig(), "cpu", seed=0)
    env.step(torch.full((4,), 1))  # SB completes
    assert bool(env.legal_info().can_check.all())  # BB may check
    with pytest.raises(ValueError):
        env.clone().step(torch.zeros(4, dtype=torch.long), validate=True)  # fold while check is possible
    with pytest.raises(ValueError):
        env.clone().step(torch.full((4,), 99), validate=True)
    a, b = env.clone(), env.clone()
    a.step(torch.zeros(4, dtype=torch.long), validate=False)
    b.step(torch.ones(4, dtype=torch.long))
    for k in VecNLHE._STATE:
        if k not in ("hist_tok",):
            assert torch.equal(getattr(a, k), getattr(b, k)), k
    # concrete: raise below the minimum
    with pytest.raises(ValueError):
        env.clone().step_concrete(torch.full((4,), 2), torch.full((4,), 150), validate=True)
    # finished slots ignore actions
    e = VecNLHE(2, GameConfig(), "cpu", seed=0)
    e.step(torch.tensor([0, 1]))
    assert e.done.tolist() == [True, False]
    before = e.state_dict()
    before = {k: v.clone() for k, v in before.items()}
    e.step(torch.tensor([5, 1]))
    assert torch.equal(e.payoffs[0], before["payoffs"][0]) and e.done[0]
    assert e.legal_mask()[0].tolist() == [False, True, False, False, False, False]


def test_raise_cap_per_street():
    env = VecNLHE(1, GameConfig(), "cpu", seed=0)
    env.reset(button=0)
    min_raise = [2, 3, 4, 5]  # any sized raise; take the smallest legal each time
    for _ in range(4):
        m = env.legal_mask()[0]
        raise_idx = [i for i in min_raise if m[i]]
        assert raise_idx
        env.step(torch.tensor([raise_idx[0]]))
    m = env.legal_mask()[0].tolist()
    assert m == [True, True, False, False, False, False]
    assert int(env.n_raises[0]) == 4
    env.step(torch.tensor([1]))  # call -> flop, counter resets
    assert int(env.street[0]) == 1 and int(env.n_raises[0]) == 0
    assert env.legal_mask()[0, 5]


def test_determinism_from_seed():
    def run(seed):
        env = VecNLHE(256, GameConfig(), "cpu", seed=seed)
        g = torch.Generator().manual_seed(99)
        for _ in range(50):
            env.step(random_abstract(env, g))
            env.reset(env.done)
        return env.state_dict()

    a, b, c = run(1), run(1), run(2)
    for k in a:
        assert torch.equal(a[k], b[k]), k
    assert not torch.equal(a["deck"], c["deck"])


def test_select_and_clone_are_independent():
    env = VecNLHE(8, GameConfig(), "cpu", seed=2)
    idx = torch.tensor([3, 3, 5])
    sub = env.select(idx)
    assert sub.n == 3
    assert torch.equal(sub.deck, env.deck[idx])
    sub.step(torch.tensor([0, 1, 1]))
    assert sub.done.tolist() == [True, False, False]
    assert not env.done.any()
    c = env.clone()
    c.step(torch.zeros(8, dtype=torch.long))
    assert c.done.all() and not env.done.any()


def test_obs_layout():
    env = VecNLHE(N, GameConfig(), "cpu", seed=6)
    g = torch.Generator().manual_seed(6)
    for _ in range(6):
        env.step(random_abstract(env, g))
        env.reset(env.done)
    o = env.obs()
    assert o["cards"].shape == (N, 7) and o["card_mask"].shape == (N, 7)
    assert o["hist"].shape == (N, 24) and o["hist_amt"].shape == (N, 24)
    assert o["scalars"].shape == (N, NUM_SCALARS)
    assert o["legal"].shape == (N, DEFAULT_SPEC.num_actions)
    p = env.actor
    hole = torch.where(p[:, None] == 0, env.cards[:, 0:2], env.cards[:, 2:4])
    assert torch.equal(o["cards"][:, :2], hole)
    assert torch.equal(o["card_mask"].sum(1), 2 + env.board_len)
    assert (o["cards"][~o["card_mask"]] == NO_CARD).all()
    assert (o["cards"][o["card_mask"]] < 52).all()
    assert torch.allclose(o["scalars"][:, 0], env.pot.float() / 20000)
    assert torch.equal(o["scalars"][:, 8:12].argmax(1), env.street)
    assert torch.equal(o["hist_mask"].sum(1), env.hist_len)
    assert (o["hist"] < env.vocab_size).all()
    o2 = env.obs(equity_samples=16, hist_runouts=4, hist_opp_samples=8, generator=torch.Generator().manual_seed(0))
    assert o2["equity"].shape == (N,) and o2["equity_hist"].shape == (N, 10)
    assert torch.allclose(o2["equity_hist"].sum(1), torch.ones(N))


def test_custom_spec():
    spec = ActionSpec(
        streets=(
            (("fold",), ("check_call",), ("allin",)),
            (("fold",), ("check_call",), ("raise", 1.0)),
            (("fold",), ("check_call",)),
            (("fold",), ("check_call",), ("raise", 0.5), ("raise", 0.5), ("allin",)),
        ),
        max_raises=2,
    )
    env = VecNLHE(64, GameConfig(), "cpu", seed=1, spec=spec)
    assert env.num_actions == 5
    m = env.legal_mask()
    assert m[:, 3:].sum() == 0  # padding entries never legal
    g = torch.Generator().manual_seed(0)
    for _ in range(60):
        env.step(random_abstract(env, g))
        check_invariants(env)
        env.reset(env.done)
