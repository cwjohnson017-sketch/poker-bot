"""Traversal value backup vs a brute-force recursion on the scalar engine.

With a fixed deal and a deterministic opponent, external sampling is
deterministic, so every node value, child value and regret the batched
frontier traversal produces must equal an exact recursive computation over
the same abstract game tree built with the scalar engine and the scalar
feature encoder.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pokerbot.blueprint.deepcfr.networks import AdvantageNet, NetConfig
from pokerbot.blueprint.deepcfr.scalar import (
    ScalarSpec,
    abstract_to_action,
    encode_state,
    engine_config,
)
from pokerbot.blueprint.deepcfr.traversal import (
    FrontierTraverser,
    NetPolicy,
    TraversalConfig,
    deal_env,
    rollout,
    uniform_policy,
)
from pokerbot.engine_select import get_engine
from pokerbot.env import GameConfig, VecNLHE
from pokerbot.env.actions import spec_from_lists

SMALL_SPEC = spec_from_lists(
    [
        [("fold",), ("check_call",), ("raise_x", 3.0), ("allin",)],
        [("fold",), ("check_call",), ("raise", 1.0), ("allin",)],
        [("fold",), ("check_call",), ("raise", 0.5), ("allin",)],
        [("fold",), ("check_call",), ("raise", 1.0), ("allin",)],
    ],
    max_raises=2,
)


def rule_policy(feats: dict[str, torch.Tensor], no_fold: bool = False) -> torch.Tensor:
    """Deterministic, hand- and history-dependent: the k-th legal action with
    k = (sum of history tokens + first hole card) mod #legal."""
    legal = feats["legal"].clone()
    if no_fold:
        legal[:, 0] = False
    nl = legal.sum(1)
    k = (feats["hist"].sum(1) + feats["cards"][:, 0]) % nl
    rank = legal.long().cumsum(1) - 1
    pick = legal & (rank == k[:, None])
    return pick.float()


def small_net(seed: int) -> AdvantageNet:
    torch.manual_seed(seed)
    net = AdvantageNet(
        NetConfig(
            num_actions=4,
            vocab_size=1 + 8 * 4,
            card_dim=8,
            card_hidden=16,
            hist_dim=8,
            hist_hidden=16,
            width=16,
        )
    )
    # larger output scale -> sharply non-uniform regret-matching policies
    with torch.no_grad():
        net.head.weight.mul_(20)
    return net.eval()


def brute_force(state, p, config, spec, policies, scale, out):
    """Exact external-sampling value of ``state`` for traverser ``p``."""
    engine = get_engine()
    if state.is_terminal:
        return state.payoffs()[p] / scale
    cur = state.current_player
    feats, info = encode_state(state, cur, config, spec)
    probs = policies[cur](feats)[0].double()
    sp = ScalarSpec.build(spec)
    if cur != p:
        (nz,) = torch.nonzero(probs > 0, as_tuple=True)
        assert len(nz) == 1, "opponent must be deterministic for the exact check"
        a = int(nz[0])
        return brute_force(
            state.child(abstract_to_action(engine, info, sp, a)),
            p,
            config,
            spec,
            policies,
            scale,
            out,
        )
    child = np.zeros(len(info.legal))
    for a, ok in enumerate(info.legal):
        if ok:
            nxt = state.child(abstract_to_action(engine, info, sp, a))
            child[a] = brute_force(nxt, p, config, spec, policies, scale, out)
    v = float((probs.numpy() * child).sum())
    key = tuple(int(t) for t in feats["hist"][0] if t != 0)
    out[key] = (v, child, np.asarray(info.legal))
    return v


def run_case(stacks, seeds, traverser, pol_tr, pol_opp, tcfg=None):
    cfg = GameConfig(stacks=stacks)
    engine = get_engine()
    ecfg = engine_config(engine, cfg)
    rng = np.random.default_rng(seeds)
    decks = [rng.permutation(52).tolist() for _ in range(4)]
    buttons = [0, 1, 0, 1]
    policies = [None, None]
    policies[traverser] = pol_tr
    policies[1 - traverser] = pol_opp
    tr = FrontierTraverser(cfg, SMALL_SPEC, tcfg or TraversalConfig(), "cpu", seed=0)
    # all four roots in one batch: slots of different hands must not mix
    env = deal_env(decks, buttons, cfg, SMALL_SPEC)
    res = tr.traverse(traverser, policies, iteration=3, roots=env)
    assert res.stats["nodes"] > 0
    scale = float(cfg.big_blind)
    for r in range(4):
        state = engine.GameState.new_hand(ecfg, buttons[r], decks[r])
        nodes: dict = {}
        v_root = brute_force(state, traverser, ecfg, SMALL_SPEC, policies, scale, nodes)
        assert float(res.root_value[r]) == pytest.approx(v_root, abs=1e-4)
        sel = (res.node_root == r).nonzero().squeeze(1)
        got = {}
        for i in sel.tolist():
            key = tuple(int(t) for t in res.samples["hist"][i] if t != 0)
            assert key not in got
            got[key] = i
        return_nodes = nodes if tcfg is None else {k: nodes[k] for k in got}
        assert set(got) == set(return_nodes), (r, len(got), len(nodes))
        for key, (v, child, legal) in return_nodes.items():
            i = got[key]
            assert float(res.node_value[i]) == pytest.approx(v, abs=1e-4)
            np.testing.assert_allclose(res.edge_value[i].numpy()[legal], child[legal], atol=1e-4)
            assert np.array_equal(res.samples["legal"][i].numpy(), legal)
            regret = np.where(legal, child - v, 0.0)
            np.testing.assert_allclose(
                res.samples["target"][i].float().numpy(), regret, atol=2e-2, rtol=2e-3
            )
            assert int(res.samples["iteration"][i]) == 3
    return res


@pytest.mark.parametrize("traverser", [0, 1])
def test_backup_matches_brute_force_net_traverser(traverser):
    net = small_net(1)
    run_case([1000, 1000], 7 + traverser, traverser, NetPolicy(net), rule_policy)


def test_backup_matches_brute_force_deeper_stacks():
    net = small_net(2)
    res = run_case([5000, 5000], 11, 0, NetPolicy(net), lambda f: rule_policy(f, no_fold=True))
    assert res.stats["nodes"] > 40


def test_capped_traversal_with_deterministic_rollouts_is_exact():
    # frontier cap and depth cap force rollouts; with a deterministic traverser
    # the rollout value is the exact continuation value, so every recorded node
    # must still match the brute force exactly
    tcfg = TraversalConfig(max_frontier_nodes=12, max_depth=3)
    opp = lambda f: rule_policy(f, no_fold=True)  # noqa: E731
    res = run_case([5000, 5000], 5, 1, rule_policy, opp, tcfg)
    assert res.stats["cut_slots"] > 0 and res.stats["max_frontier"] <= 12
    assert res.stats["nodes"] >= 8


def brute_force_rb(state, p, config, spec, policies, scale, out, E, beta, start):
    """``brute_force`` with the variance-reduced leaf values: all-ins called
    before the river score ``stake * (2 E - 1)``, and every new street adds
    ``-beta * 2c * (E_new - E_old)`` to the value of the edge that dealt it.
    ``E`` is the root's per-street equity (from the traversal under test)."""
    engine = get_engine()
    cur = state.current_player
    feats, info = encode_state(state, cur, config, spec)
    probs = policies[cur](feats)[0].double()
    sp = ScalarSpec.build(spec)

    def edge(nxt):
        if nxt.is_terminal:
            if not any(nxt.folded) and state.street < 3:
                stake = min(start[q] - nxt.stacks[q] for q in (0, 1))
                return stake * (2 * E[state.street] - 1) / scale
            return nxt.payoffs()[p] / scale
        v = brute_force_rb(nxt, p, config, spec, policies, scale, out, E, beta, start)
        if nxt.street != state.street:
            c = start[p] - nxt.stacks[p]
            v -= beta * 2 * c * (E[nxt.street] - E[state.street]) / scale
        return v

    if cur != p:
        (nz,) = torch.nonzero(probs > 0, as_tuple=True)
        assert len(nz) == 1, "opponent must be deterministic for the exact check"
        return edge(state.child(abstract_to_action(engine, info, sp, int(nz[0]))))
    child = np.zeros(len(info.legal))
    for a, ok in enumerate(info.legal):
        if ok:
            child[a] = edge(state.child(abstract_to_action(engine, info, sp, a)))
    v = float((probs.numpy() * child).sum())
    key = tuple(int(t) for t in feats["hist"][0] if t != 0)
    out[key] = (v, child, np.asarray(info.legal))
    return v


def run_case_rb(stacks, seeds, traverser, pol_tr, pol_opp, beta, **caps):
    cfg = GameConfig(stacks=stacks)
    engine = get_engine()
    ecfg = engine_config(engine, cfg)
    rng = np.random.default_rng(seeds)
    decks = [rng.permutation(52).tolist() for _ in range(4)]
    buttons = [0, 1, 0, 1]
    policies = [None, None]
    policies[traverser] = pol_tr
    policies[1 - traverser] = pol_opp
    tcfg = TraversalConfig(allin_equity=True, chance_cv=beta, preflop_equity_samples=64, **caps)
    tr = FrontierTraverser(cfg, SMALL_SPEC, tcfg, "cpu", seed=0)
    env = deal_env(decks, buttons, cfg, SMALL_SPEC)
    res = tr.traverse(traverser, policies, iteration=3, roots=env)
    assert res.root_equity is not None and res.root_equity.shape == (4, 4)
    scale = float(cfg.big_blind)
    for r in range(4):
        state = engine.GameState.new_hand(ecfg, buttons[r], decks[r])
        E = res.root_equity[r].double().numpy()
        nodes: dict = {}
        v_root = brute_force_rb(
            state, traverser, ecfg, SMALL_SPEC, policies, scale, nodes, E, beta, list(stacks)
        )
        assert float(res.root_value[r]) == pytest.approx(v_root, abs=1e-4)
        sel = (res.node_root == r).nonzero().squeeze(1)
        got = {tuple(int(t) for t in res.samples["hist"][i] if t != 0): i for i in sel.tolist()}
        compare = {k: nodes[k] for k in got} if caps else nodes
        assert set(got) == set(compare)
        for key, (v, child, legal) in compare.items():
            i = got[key]
            assert float(res.node_value[i]) == pytest.approx(v, abs=1e-4)
            np.testing.assert_allclose(res.edge_value[i].numpy()[legal], child[legal], atol=1e-4)
            regret = np.where(legal, child - v, 0.0)
            np.testing.assert_allclose(
                res.samples["target"][i].float().numpy(), regret, atol=2e-2, rtol=2e-3
            )
    return res, env


@pytest.mark.parametrize(("traverser", "beta"), [(0, 1.0), (1, 1.0), (0, 0.5)])
def test_variance_reduced_backup_matches_brute_force(traverser, beta):
    net = small_net(3)
    res, env = run_case_rb(
        [1000, 1000], 13 + traverser, traverser, NetPolicy(net), rule_policy, beta
    )
    assert res.stats["allin_leaves"] > 0
    # the corrections change the values, never the sampled play
    policies = [rule_policy, rule_policy]
    policies[traverser] = NetPolicy(net)
    plain = FrontierTraverser(GameConfig(stacks=[1000, 1000]), SMALL_SPEC, seed=0)
    base = plain.traverse(traverser, policies, 3, env)
    assert base.stats["nodes"] == res.stats["nodes"]
    assert not torch.allclose(base.root_value, res.root_value)


def test_variance_reduced_backup_deeper_stacks():
    net = small_net(4)
    opp = lambda f: rule_policy(f, no_fold=True)  # noqa: E731
    res, _ = run_case_rb([5000, 5000], 17, 0, NetPolicy(net), opp, 1.0)
    assert res.stats["nodes"] > 40 and res.stats["allin_leaves"] > 0


def test_variance_reduced_backup_with_caps_and_step_limit():
    # frontier, depth and step caps force rollouts; with a deterministic
    # traverser the rolled-out values (and their corrections) are exact
    opp = lambda f: rule_policy(f, no_fold=True)  # noqa: E731
    res, _ = run_case_rb(
        [5000, 5000], 5, 1, rule_policy, opp, 1.0, max_frontier_nodes=12, max_depth=3, max_steps=5
    )
    assert res.stats["cut_slots"] > 0 and res.stats["frontier_steps"] > 5


def test_strategy_samples_and_stats():
    cfg = GameConfig(stacks=[1500, 1500])
    tcfg = TraversalConfig(record_strategy=True)
    tr = FrontierTraverser(cfg, SMALL_SPEC, tcfg, "cpu", seed=3)
    res = tr.traverse(0, [uniform_policy, rule_policy], iteration=2, roots=32)
    s = res.strategy
    assert s is not None and s["target"].shape[0] > 0
    probs = s["target"].float()
    assert torch.allclose(probs.sum(1), torch.ones(len(probs)), atol=1e-3)
    assert ((probs > 0) <= s["legal"]).all()
    # regrets of a node sum (policy-weighted) to zero
    samples = res.samples
    assert samples["target"].shape == samples["legal"].shape
    assert res.stats["slot_steps"] >= res.stats["nodes"]


def test_rollout_finishes_all_slots():
    env = VecNLHE(64, GameConfig(), "cpu", seed=4, validate=False)
    g = torch.Generator().manual_seed(0)
    pay = rollout(env, [uniform_policy, uniform_policy], g)
    assert bool(env.done.all())
    assert torch.equal(pay.sum(1), torch.zeros(64, dtype=torch.long))
