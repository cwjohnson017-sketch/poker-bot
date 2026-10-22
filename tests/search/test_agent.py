"""SearchAgent end to end: legal play through the match runner (masked states)."""

from __future__ import annotations

import numpy as np

from pokerbot.agents import RandomAgent, make_agent
from pokerbot.engine_select import get_engine
from pokerbot.eval.match import run_match
from pokerbot.search import SearchAgent, UniformBlueprint

TINY = {
    "device": "cpu",
    "time_budget": 0.02,
    "min_iterations": 2,
    "fallback_on_error": False,
    "tree": {"max_nodes": 1500, "chance_cards": 3, "max_raises": 2},
    "solver": {"iterations": 4, "max_runouts": 6},
    "leaf": {"rollouts": 1, "max_total_rollouts": 64},
    "gadget": {"rollouts": 8},
}


def test_search_agent_plays_legal_hands_vs_random():
    engine = get_engine()
    config = engine.GameConfig(
        num_players=2, stacks=[1000, 1000], small_blind=50, big_blind=100, ante=0
    )
    agent = SearchAgent(UniformBlueprint(), TINY)
    res = run_match([agent, RandomAgent()], config, num_hands=100, seed=3)
    assert res.hands == 100
    assert int(res.seat_payoffs.sum()) == 0
    postflop = [s for s in agent.stats if "nodes" in s]
    assert postflop, "the agent never searched"
    assert not any(s.get("fallback") for s in agent.stats)
    # a later decision in the same hand should have reused a cached street root
    assert all(s["nodes"] <= 1500 for s in postflop)


def test_search_agent_unsafe_and_cached_roots():
    engine = get_engine()
    config = engine.GameConfig(
        num_players=2, stacks=[1500, 1500], small_blind=50, big_blind=100, ante=0
    )
    cfg = dict(TINY, gadget={"safe": False, "rollouts": 8})
    agent = SearchAgent(UniformBlueprint(), cfg)
    from pokerbot.agents import AlwaysCallAgent

    run_match([agent, AlwaysCallAgent()], config, num_hands=20, seed=5)
    assert any(s.get("cached_root") for s in agent.stats if "nodes" in s)


def test_make_agent_registers_search_prefix():
    a = make_agent("search:uniform", config=TINY)
    assert isinstance(a, SearchAgent)
    assert isinstance(a.blueprint, UniformBlueprint)
    assert a.name == "search:uniform"
    rng = np.random.default_rng(0)
    engine = get_engine()
    cfg = engine.GameConfig(num_players=2, stacks=[1000, 1000], small_blind=50, big_blind=100)
    s = engine.GameState.new_hand(cfg, 0, list(range(52)))
    a.new_hand(0, cfg)
    act = a.act(s, 0, rng)  # preflop: blueprint
    assert s.legal_actions().is_legal(act)


def test_played_lock_keeps_the_taken_action_exact():
    """A lock on a tree that lost one of our played sizes: the action we took
    keeps its probability, the lost mass goes to the other children."""
    import torch

    from pokerbot.search.agent import played_lock

    g = torch.Generator().manual_seed(0)
    old_acts = [(1, 0), (2, 375), (2, 500), (2, 10000)]
    old = torch.rand(6, 4, generator=g)
    old[5, 2:] = 0  # a combo that only checks or bets 375
    old = old / old.sum(1, keepdim=True)
    acts = [(1, 0), (2, 500), (2, 10000)]  # the re-search tree dropped 375
    for slot in range(3):
        s = played_lock(acts, slot, old_acts, old)
        torch.testing.assert_close(s.sum(1), torch.ones(6))
        taken = old[:, old_acts.index(acts[slot])]
        torch.testing.assert_close(s[:, slot], taken)
        others = [j for j in range(3) if j != slot]
        rest = old[:, [old_acts.index(acts[j]) for j in others]]
        lost = old[:, 1:2]
        tot = rest.sum(1, keepdim=True)
        want = torch.where(tot > 0, rest * (1 + lost / tot.clamp(min=1e-30)), lost / 2)
        torch.testing.assert_close(s[:, others], want)
    # nothing dropped: the played strategy as it was
    torch.testing.assert_close(played_lock(old_acts, 1, old_acts, old), old)


def test_recording_keeps_the_play_and_logs_every_decision():
    """``record=True`` changes no action (same deals, same RNG, fixed iteration
    counts) and keeps one record per decision of the hand."""
    from pokerbot.agents import AlwaysCallAgent
    from pokerbot.eval.match import play_hand
    from pokerbot.search.combos import combo_index, combo_table

    cards = combo_table().numpy()
    engine = get_engine()
    config = engine.GameConfig(
        num_players=2, stacks=[1000, 1000], small_blind=50, big_blind=100, ante=0
    )
    cfg = dict(TINY, time_budget=60.0)  # the iteration cap, not the clock, ends each solve
    searched = 0
    for seed in range(4):
        deck = np.random.default_rng(seed).permutation(52).tolist()
        runs = []
        for record in (False, True):
            agent = SearchAgent(UniformBlueprint(), cfg, record=record)
            rng = np.random.default_rng(seed + 100)
            state = play_hand(config, [agent, AlwaysCallAgent()], 0, deck, rng, engine)
            runs.append([(s, p, int(a.kind), int(a.amount)) for s, p, a in state.history])
        assert runs[0] == runs[1]
        ours = [h for h in runs[1] if h[1] == 0]
        assert len(agent.decisions) == len(ours)
        c = combo_index(*state.hole_cards(0))
        for d, (street, _, kind, amount) in zip(agent.decisions, ours, strict=True):
            assert d["street"] == street
            assert d["source"] == ("blueprint" if street == 0 else "search")
            k, a = d["actions"][d["taken"]]
            assert k == kind and (kind != 2 or a == amount)
            assert d["strategy"].shape == (1326, len(d["actions"]))
            assert np.allclose(d["strategy"][c].sum(), 1.0, atol=1e-5)
            assert d["reach"].shape == (2, 1326)
            if street:
                searched += 1
                assert d["values"].shape == d["strategy"].shape
                assert np.isfinite(d["values"][c]).all()
                # values are chips for the whole hand: within the stacks
                assert np.abs(d["values"][c]).max() <= 1000 + 1e-3
                # values_vs: our combo against each opponent combo; NaN on blocked
                # combos, and averaged by the opponent's reach it is values[c]
                vs = d["values_vs"]
                assert vs.shape == d["strategy"].shape
                blocked = np.isin(cards, cards[c]).any(1)
                assert np.isnan(vs[blocked]).all() and np.isfinite(vs[~blocked]).all()
                w = d["reach"][1].astype(np.float64) * ~blocked
                avg = (w[:, None] * np.nan_to_num(vs)).sum(0) / w.sum()
                np.testing.assert_allclose(avg, d["values"][c], atol=0.05)
                # (no stack bound per pair: sampled chance cards and rollouts weight a
                # pair's runouts exactly only on average over pairs)
            else:
                assert d["values_vs"] is None
    assert searched, "no postflop decision in these deals"
