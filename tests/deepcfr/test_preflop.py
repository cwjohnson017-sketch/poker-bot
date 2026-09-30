"""Tabular preflop regrets (:mod:`pokerbot.blueprint.deepcfr.preflop`)."""

import csv
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest
import torch

from pokerbot.blueprint.deepcfr.checkpoint import list_checkpoints, load_checkpoint
from pokerbot.blueprint.deepcfr.config import DeepCFRConfig
from pokerbot.blueprint.deepcfr.features import features_from_obs
from pokerbot.blueprint.deepcfr.policy import SDCFRPolicy
from pokerbot.blueprint.deepcfr.preflop import (
    NUM_CLASSES,
    PreflopRegrets,
    PreflopTree,
    TablePolicy,
    hand_class,
)
from pokerbot.blueprint.deepcfr.trainer import DeepCFRTrainer
from pokerbot.blueprint.deepcfr.traversal import NetPolicy, TraversalConfig, uniform_policy
from pokerbot.env import DEFAULT_SPEC, GameConfig, VecNLHE

from .test_traversal import SMALL_SPEC, run_case, small_net

TINY = Path(__file__).resolve().parents[2] / "configs" / "deepcfr_tiny.yaml"


def test_tree_matches_the_tabular_blueprints_node_count():
    tree = PreflopTree.build(GameConfig(stacks=[10000, 10000]), DEFAULT_SPEC)
    assert len(tree) == 92  # configs/mccfr_hunl.yaml: 92 preflop decision nodes
    hist = torch.zeros(len(tree), 24, dtype=torch.long)
    for i, t in enumerate(tree.tokens):
        hist[i, : len(t)] = torch.tensor(t)
    assert torch.equal(tree.node_index(hist), torch.arange(len(tree)))
    # a flop history is not a preflop node
    env = VecNLHE(1, GameConfig(stacks=[10000, 10000]), "cpu", 0, DEFAULT_SPEC, validate=False)
    env.step(torch.tensor([1]))  # button limps
    env.step(torch.tensor([1]))  # big blind checks: flop
    assert int(env.street[0]) == 1
    assert int(tree.node_index(env.hist_tok)[0]) == -1
    rt = PreflopTree.from_meta(tree.to_meta())
    assert rt.tokens == tree.tokens and torch.equal(rt.legal, tree.legal)


def test_hand_class_is_the_169_lossless_classes():
    hole = torch.tensor(list(combinations(range(52), 2)))
    cls = hand_class(hole)
    counts = torch.bincount(cls, minlength=NUM_CLASSES)
    assert int(cls.max()) == NUM_CLASSES - 1 and (counts > 0).all()
    assert sorted(set(counts.tolist())) == [4, 6, 12]  # suited, pairs, offsuit
    assert int((counts == 6).sum()) == 13 and int((counts == 4).sum()) == 78


def test_regrets_accumulate_into_a_regret_matching_table():
    tree = PreflopTree.build(GameConfig(stacks=[1000, 1000]), SMALL_SPEC)
    reg = PreflopRegrets(tree)
    A = tree.num_actions
    root = tree.tokens.index([])
    legal = tree.legal[root]
    # two samples of the root with a pocket pair of aces (class 12 * 14)
    samples = {
        "cards": torch.tensor([[51, 50] + [52] * 5] * 2, dtype=torch.uint8),
        "hist": torch.zeros(2, 24, dtype=torch.uint8),
        "scalars": torch.zeros(2, 14, dtype=torch.float16),
        "legal": legal.expand(2, A),
        "target": torch.tensor([[-1.0, 2.0, 1.0, 0.0], [0.0, -1.0, 3.0, 0.0]])[:, :A].half(),
    }
    samples["scalars"][:, 8] = 1  # preflop
    acc = reg.pending()
    assert reg.add_samples(acc, samples, 2.0) == 2
    reg.regret += acc
    cell = root * NUM_CLASSES + 12 * 14
    r = reg.regret[cell]
    assert torch.allclose(r[:3], torch.tensor([-2.0, 2.0, 8.0], dtype=torch.float64))
    strat = reg.strategy()
    pos = r.clamp(min=0) * legal
    assert torch.allclose(strat[cell], (pos / pos.sum()).float())
    other = root * NUM_CLASSES + 0  # untouched class: uniform over the legal actions
    assert torch.allclose(strat[other], legal.float() / legal.sum())


def test_table_policy_routes_preflop_rows_to_the_table():
    cfg = GameConfig(stacks=[1000, 1000])
    tree = PreflopTree.build(cfg, SMALL_SPEC)
    torch.manual_seed(0)
    strat = torch.rand(len(tree) * NUM_CLASSES, tree.num_actions)
    pol = TablePolicy(uniform_policy, tree, strat)
    env = VecNLHE(64, cfg, "cpu", 3, SMALL_SPEC, validate=False)
    g = torch.Generator().manual_seed(1)
    for _ in range(4):
        f = features_from_obs(env.obs())
        p = pol(f)
        pre = f["scalars"][:, 8] > 0.5
        uni = uniform_policy(f)
        assert torch.allclose(p[~pre], uni[~pre])
        rows, cells = tree.locate(f)
        assert torch.equal(rows, pre.nonzero().squeeze(1))  # every preflop row is on the tree
        want = strat[cells] * f["legal"][rows]
        assert torch.allclose(p[rows], want / want.sum(1, keepdim=True))
        a = torch.multinomial(env.legal_mask().float(), 1, generator=g).squeeze(1)
        env.step(torch.where(a == 0, 1, a))
        env.reset(env.done)


def test_traversal_with_a_table_policy_matches_brute_force():
    # the same table policy must act identically on env and scalar features
    tree = PreflopTree.build(GameConfig(stacks=[1000, 1000]), SMALL_SPEC)
    torch.manual_seed(2)
    strat = torch.rand(len(tree) * NUM_CLASSES, tree.num_actions)
    from .test_traversal import rule_policy

    pol = TablePolicy(NetPolicy(small_net(5)), tree, strat)
    run_case([1000, 1000], 21, 0, pol, rule_policy)


def _tiny(tmp_path, **train):
    cfg = DeepCFRConfig.load(TINY)
    cfg.logging.print = False
    cfg.logging.tensorboard = False
    cfg.eval.every = 0
    cfg.training.tabular_preflop = True
    for k, v in train.items():
        setattr(cfg.training, k, v)
    return cfg


@pytest.fixture
def _threads():
    n = torch.get_num_threads()
    yield
    torch.set_num_threads(n)


def test_training_with_tabular_preflop(tmp_path, _threads):
    tr = DeepCFRTrainer(_tiny(tmp_path), tmp_path)
    tr.run_iteration(1)
    tr.run_iteration(2)
    tr.close()
    for p in (0, 1):
        assert tr.preflop[p].regret.abs().sum() > 0
    with open(tmp_path / "log.csv") as fh:
        assert all(int(r["preflop_samples"]) > 0 for r in csv.DictReader(fh))
    cks = list_checkpoints(tmp_path, 0)
    assert [t for t, _ in cks] == [1, 2]
    _, _, table = load_checkpoint(cks[-1][1])
    assert table is not None and table.shape == (len(tr.preflop[0].tree) * NUM_CLASSES, 4)
    assert torch.allclose(table, tr.preflop[0].strategy())

    # the SD-CFR average plays each iteration's table at preflop nodes
    pol = SDCFRPolicy.from_dir(tmp_path, 0)
    assert pol.preflop_tables is not None and pol.preflop_tables.shape[0] == 2
    env = VecNLHE(16, tr.game_config, "cpu", 5, tr.spec, validate=False)
    f = features_from_obs(env.obs())
    rows, cells = pol.preflop_tree.locate(f)
    assert rows.numel() == 16
    P = pol.net_policies(f)
    for i in range(2):
        want = pol.preflop_tables[i][cells] * f["legal"].float()
        want = torch.where(want.sum(1, keepdim=True) > 0, want, f["legal"].float())
        assert torch.allclose(P[i], want / want.sum(1, keepdim=True))
        assert torch.allclose(pol.net_probs(f, i), P[i])

    # resume restores the regret tables
    tr2 = DeepCFRTrainer(_tiny(tmp_path), tmp_path, resume=True)
    for p in (0, 1):
        assert torch.equal(tr2.preflop[p].regret, tr.preflop[p].regret)
    tr2.close()


def test_agent_plays_with_tabular_preflop(tmp_path, _threads):
    from pokerbot.agents import RandomAgent
    from pokerbot.blueprint.deepcfr.agent import NeuralBlueprintAgent
    from pokerbot.config import game_config
    from pokerbot.engine_select import get_engine
    from pokerbot.eval.match import run_match

    tr = DeepCFRTrainer(_tiny(tmp_path), tmp_path)
    tr.run_iteration(1)
    tr.close()
    engine = get_engine()
    config = game_config(tr.meta["game"], engine)
    for sample_net in (False, True):
        agent = NeuralBlueprintAgent.from_dir(tmp_path, sample_net=sample_net)
        res = run_match([agent, RandomAgent()], config, 30, seed=1, engine=engine)
        assert res.hands == 30
    # the stateless range query reads the table at the root
    state = engine.GameState.new_hand(config, 0, list(range(52)))
    probs = agent.policy_all(state, 0).double()
    cls = hand_class(torch.tensor(list(combinations(range(52), 2))))
    root = tr.preflop[0].tree.tokens.index([])
    table = tr.preflop[0].strategy()[root * NUM_CLASSES + cls].double()
    legal = tr.preflop[0].tree.legal[root].double()
    want = table * legal
    want = want / want.sum(1, keepdim=True)
    np.testing.assert_allclose(probs.numpy(), want.numpy(), atol=1e-5)


def test_policy_without_tables_is_unchanged():
    net = small_net(1)
    pol = SDCFRPolicy([(1, net)])
    assert pol.preflop_tree is None
    f = features_from_obs(VecNLHE(4, GameConfig(stacks=[1000, 1000]), "cpu", 0, SMALL_SPEC).obs())
    assert torch.allclose(pol.net_policies(f)[0], NetPolicy(net)(f))
    assert TraversalConfig().allin_equity is False
