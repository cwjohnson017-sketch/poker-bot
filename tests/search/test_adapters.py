"""Search adapters for the trained blueprints (tabular MCCFR and Deep CFR).

The batched ``policy_combos`` must equal the single-hand ``policy`` (which
goes through a different code path: the Rust key and bucket code for the
tabular blueprint, the agent's own encoder for the neural one), give zero
mass to combos that share a board card, and both blueprints must drive a
``SearchAgent`` through legal hands from the registry.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
import torch

from pokerbot.agents import RandomAgent, make_agent
from pokerbot.engine_select import get_engine
from pokerbot.eval.masking import MaskedState
from pokerbot.eval.match import run_match
from pokerbot.search import SearchAgent, search_config
from pokerbot.search.abstract import CardView, legal_options, to_action
from pokerbot.search.adapters import NeuralBlueprint, TabularBlueprint
from pokerbot.search.blueprint import make_blueprint, policy_matrix
from pokerbot.search.combos import NUM_COMBOS, combo_cards, valid_mask

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


@pytest.fixture
def one_thread():
    """Small kernels: extra torch threads only add contention on a shared CPU."""
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


def random_states(spec, config, hands, seed, engine=None):
    """Decision states of random play over ``spec``'s abstract actions."""
    engine = engine or get_engine()
    rng = np.random.default_rng(seed)
    out = []
    for h in range(hands):
        s = engine.GameState.new_hand(config, h % 2, rng.permutation(52).tolist())
        while not s.is_terminal:
            out.append(s.clone())
            opts = legal_options(s, spec)
            o = opts[int(rng.integers(len(opts)))]
            s.apply(to_action(engine, o.kind, o.amount))
    return out


def check_combos(bp, states, per_state, seed, atol):
    rng = np.random.default_rng(seed)
    checked = 0
    for s in states:
        p = s.current_player
        view = CardView(s)
        P = bp.policy_combos(view, p)
        assert P.shape == (NUM_COMBOS, bp.spec.num_actions)
        ok = valid_mask(s.board)
        assert bool((P[~ok] == 0).all()), "card-conflict combos must get zero mass"
        assert torch.allclose(P[ok].sum(1), torch.ones(int(ok.sum())), atol=1e-5)
        legal = torch.zeros(bp.spec.num_actions, dtype=torch.bool)
        legal[[o.index for o in legal_options(s, bp.spec)]] = True
        assert bool((P[:, ~legal] == 0).all())
        for c in rng.choice(np.nonzero(ok.numpy())[0], per_state, replace=False):
            one = np.asarray(bp.policy(view.with_hole(p, combo_cards(int(c))), p))
            np.testing.assert_allclose(one, P[c].double().numpy(), atol=atol)
            checked += 1
    return checked


def test_tabular_policy_combos_match_policy(small_strategy, one_thread):
    import poker_engine as pe

    bp = make_blueprint(f"blueprint:{small_strategy}")
    assert isinstance(bp, TabularBlueprint)
    table = bp.table
    train = table.strategy.game_config
    # the MCCFR action abstraction and the ActionSpec agree on the tree
    for s in random_states(bp.spec, train, 30, seed=1, engine=pe):
        node = table.tracked(s, s.current_player)
        assert node is not None
        legal, _ = node.info(table.abstraction)
        assert sorted(legal) == [o.index for o in legal_options(s, bp.spec)]
    # the batched buckets are the strategy's own buckets
    rng = np.random.default_rng(0)
    for street, n in ((0, 0), (1, 3), (2, 4), (3, 5)):
        board = rng.permutation(52)[:n].tolist()
        bk = table.board_buckets(street, board)
        for c in rng.choice(np.nonzero(valid_mask(board).numpy())[0], 25, replace=False):
            assert bk[c] == table.strategy.bucket(street, list(combo_cards(int(c))), board)
    # on the training game and off it (20bb: every bet is translated)
    engine = get_engine()
    short = engine.GameConfig(num_players=2, stacks=[2000, 2000], small_blind=50, big_blind=100)
    states = random_states(bp.spec, train, 12, seed=2) + random_states(bp.spec, short, 12, seed=3)
    assert check_combos(bp, states, 30, seed=4, atol=1e-6) > 2000
    # a rollout-style view: full runout board on a flop state
    s = next(x for x in states if x.street == 1)
    unseen = [c for c in range(52) if c not in s.board]
    full = list(s.board) + unseen[:2]
    P = policy_matrix(bp, CardView(s, full), s.current_player)
    np.testing.assert_allclose(
        P.numpy(), policy_matrix(bp, CardView(s), s.current_player).numpy(), atol=1e-6
    )


def test_neural_policy_combos_match_policy(tiny_neural_run, one_thread):
    bp = make_blueprint(f"neural:{tiny_neural_run}")
    assert isinstance(bp, NeuralBlueprint)
    assert len(bp.agent.policies[0]) == 2 and bp.agent.policies[0].reach_weighted
    engine = get_engine()
    g = bp.game
    config = engine.GameConfig(
        num_players=2, stacks=g["stacks"], small_blind=g["small_blind"], big_blind=g["big_blind"]
    )
    states = random_states(bp.spec, config, 5, seed=5)
    assert check_combos(bp, states, 5, seed=6, atol=1e-5) > 50


@pytest.mark.parametrize("kind", ["blueprint", "neural"])
def test_search_agent_with_trained_blueprint_plays_legal_hands(
    kind, small_strategy, tiny_neural_run, one_thread
):
    src = small_strategy if kind == "blueprint" else tiny_neural_run
    engine = get_engine()
    config = engine.GameConfig(
        num_players=2, stacks=[1000, 1000], small_blind=50, big_blind=100, ante=0
    )
    agent = make_agent(f"search:{kind}:{src}", config=TINY)
    assert isinstance(agent, SearchAgent) and agent.name == f"search:{kind}:{src}"
    res = run_match([agent, RandomAgent()], config, num_hands=20, seed=7, engine=engine)
    assert res.hands == 20
    assert int(res.seat_payoffs.sum()) == 0
    postflop = [s for s in agent.stats if "nodes" in s]
    assert postflop, "the agent never searched"
    assert not any(s.get("fallback") for s in agent.stats)


@pytest.mark.slow
def test_flop_decision_timing_tabular_blueprint(small_strategy, monkeypatch):
    """Informational (run with ``-s``; about a minute on a shared 4-core CPU):
    one flop decision of ``search:blueprint:<mccfr_small>`` at the default
    search config, the depth-limit leaf rollout cost per leaf, and the cost of
    the batched blueprint queries inside it."""
    import pokerbot.search.agent as agent_mod

    engine = get_engine()
    cfg = engine.GameConfig(num_players=2, stacks=[10000, 10000], small_blind=50, big_blind=100)
    deck = np.random.default_rng(11).permutation(52).tolist()
    s = engine.GameState.new_hand(cfg, 0, deck)
    s.apply(engine.Action.raise_to(250))
    s.apply(engine.Action.check_call())
    seat = s.current_player
    sc = search_config(device="cpu", fallback_on_error=False, min_iterations=1)
    agent = make_agent(f"search:blueprint:{small_strategy}", config=sc)
    bp = agent.blueprint
    agent.new_hand(seat, cfg)
    rng = np.random.default_rng(0)
    calls = {"n": 0, "t": 0.0}
    leaf = {}
    inner = bp.policy_combos
    build = agent_mod.build_leaf_rollouts

    def timed(state, player):
        t0 = time.perf_counter()
        out = inner(state, player)
        calls["t"] += time.perf_counter() - t0
        calls["n"] += 1
        return out

    def timed_build(tree, *args, **kwargs):
        from pokerbot.search.tree import LEAF

        n0, t0 = calls["n"], time.perf_counter()
        rs = build(tree, *args, **kwargs)
        leaf.update(
            seconds=time.perf_counter() - t0,
            leaves=int((tree.kind == LEAF).sum()),
            rows=rs.num_rows,
            queries=calls["n"] - n0,
        )
        return rs

    bp.policy_combos = timed
    monkeypatch.setattr(agent_mod, "build_leaf_rollouts", timed_build)
    action = agent.act(MaskedState(s, seat, cfg, rng), seat, rng)
    assert s.legal_actions().is_legal(action)
    st = agent.last_stats
    per_iter = st["solve_seconds"] / max(1, st["iterations"])
    nl = max(1, leaf.get("leaves", 0))
    print(
        f"\n[search:blueprint timing, CPU] flop after a 2.5x open and call, default config: "
        f"{st['nodes']} nodes; tree {st['tree_seconds']:.2f}s, setup {st['setup_seconds']:.2f}s, "
        f"{st['iterations']} iterations in {st['solve_seconds']:.2f}s "
        f"({per_iter:.2f}s/iteration), decision total {st['total_seconds']:.2f}s\n"
        f"  leaf rollouts: {leaf.get('leaves', 0)} leaves, {leaf.get('rows', 0)} rollout rows, "
        f"{leaf.get('seconds', 0.0):.2f}s = {1e3 * leaf.get('seconds', 0.0) / nl:.1f} ms/leaf, "
        f"{leaf.get('queries', 0)} blueprint queries\n"
        f"  blueprint policy_combos: {calls['n']} calls, {calls['t']:.2f}s total, "
        f"{1e3 * calls['t'] / max(1, calls['n']):.2f} ms/call"
    )


def test_incremental_reach_rollouts_match_history_replay(tiny_neural_run, one_thread):
    """Rollouts that carry the neural blueprint's own reach forward give the
    same weights, actions and payoffs as replaying the history at every step."""
    from pokerbot.search.leaf import LeafConfig, _rollout

    bp = make_blueprint(f"neural:{tiny_neural_run}")
    assert hasattr(bp, "policy_combos_nets")

    class Replay:  # the same blueprint without the incremental hooks
        spec = bp.spec

        def policy_combos(self, state, player):
            return bp.policy_combos(state, player)

    engine = get_engine()
    g = bp.game
    config = engine.GameConfig(
        num_players=2, stacks=g["stacks"], small_blind=g["small_blind"], big_blind=g["big_blind"]
    )
    flops = [s for s in random_states(bp.spec, config, 30, seed=13) if s.street == 1][:6]
    assert flops
    cfg = LeafConfig()
    checked = 0
    for i, s in enumerate(flops):
        rng = np.random.default_rng(i)
        full = list(s.board) + [c for c in range(52) if c not in s.board][:2]
        for chooser in (0, 1):
            a = _rollout(
                s,
                full,
                bp,
                chooser,
                cfg.strategies,
                cfg,
                config,
                engine,
                np.random.default_rng(100 + i),
            )
            b = _rollout(
                s,
                full,
                Replay(),
                chooser,
                cfg.strategies,
                cfg,
                config,
                engine,
                np.random.default_rng(100 + i),
            )
            assert a[2:] == b[2:]  # kind, folder, amount: the same sampled line
            torch.testing.assert_close(a[0], b[0], rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(a[1], b[1], rtol=1e-5, atol=1e-6)
            checked += 1
        del rng
    assert checked >= 6
