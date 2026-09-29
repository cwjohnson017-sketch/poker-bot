"""MCCFR solver: convergence on a reduced game, checkpoints, strategy export."""

import json

import numpy as np
import poker_engine as pe
import pytest
from mccfr_helpers import TINY_BUCKETS, TINY_HS_SAMPLES, tiny_config

from pokerbot.agents import AlwaysCallAgent, RandomAgent
from pokerbot.blueprint.mccfr import BlueprintAgent, decode_key, make_key, read_strategy
from pokerbot.blueprint.mccfr.br import best_response, strategy_from_file, uniform_strategy
from pokerbot.blueprint.mccfr.train import solver_config, train
from pokerbot.config import REPO_ROOT, load_yaml
from pokerbot.eval import run_duplicate_match

DECK = list(range(52))


def exploitability(trainer, path, deals=12000):
    trainer.export_strategy(str(path))
    cards = pe.CardAbstraction(TINY_BUCKETS, TINY_HS_SAMPLES)
    strat = strategy_from_file(read_strategy(path))
    game, actions = trainer.game_config, trainer.action_abstraction()
    return best_response(strat, game, actions, cards, num_deals=deals, seed=0)


def test_exploitability_proxy_improves(tmp_path):
    t = pe.Trainer(tiny_config())
    game, actions = t.game_config, t.action_abstraction()
    cards = pe.CardAbstraction(TINY_BUCKETS, TINY_HS_SAMPLES)
    uniform = best_response(uniform_strategy, game, actions, cards, num_deals=12000, seed=0)
    t.run(10_000, 1)
    early = exploitability(t, tmp_path / "a.bin")
    t.run(110_000, 2)
    late = exploitability(t, tmp_path / "b.bin")
    # Measured: uniform ~1.75 bb/hand, 10k iterations ~0.3, 120k ~0.12.
    assert uniform.exploitability > 1.2
    assert early.exploitability < 0.5 * uniform.exploitability
    assert late.exploitability < 0.75 * early.exploitability
    assert late.exploitability < 0.2
    for r in (uniform, early, late):
        assert min(r.br_value) > -0.5  # a best response never loses much


def test_blueprint_beats_simple_agents(tiny_blueprint):
    game = tiny_blueprint["trainer"].game_config
    for opp in (AlwaysCallAgent(), RandomAgent()):
        agent = BlueprintAgent(tiny_blueprint["path"])
        res = run_duplicate_match(agent, opp, game, 400, seed=11, engine=pe)
        lo, _hi = res.ci
        assert res.mbb_per_hand > 500, res.summary()
        assert lo > 0, res.summary()
        if isinstance(opp, AlwaysCallAgent):  # never leaves the abstract tree
            assert agent.counters["fallback"] == 0


def test_checkpoint_round_trip(tmp_path):
    t = pe.Trainer(tiny_config(seed=5))
    t.meta = json.dumps({"note": "round trip"})
    t.run(3000, 2)
    path = tmp_path / "ckpt.bin"
    t.save(str(path))
    u = pe.Trainer.load(str(path))
    assert u.iterations == t.iterations == 3000
    assert u.config_json == t.config_json
    assert json.loads(u.meta) == {"note": "round trip"}
    a = {k: (r, s) for k, r, s in t.entries()}
    b = {k: (r, s) for k, r, s in u.entries()}
    assert a == b and len(a) > 100
    s = pe.GameState.new_hand(t.game_config, 0, DECK)
    assert t.strategy(s, 0) == u.strategy(s, 0)
    # Training continues from the checkpoint.
    u.run(1000, 1)
    assert u.iterations == 4000
    st = u.stats(detailed=True)
    assert st["infosets"] == sum(st["infosets_per_street"]) >= len(a)
    assert st["table_bytes"] > 0 and st["rss_bytes"] > 0
    with pytest.raises(ValueError):
        pe.Trainer.load(str(tmp_path / "missing.bin"))


def test_single_thread_is_deterministic():
    a, b = pe.Trainer(tiny_config(seed=9)), pe.Trainer(tiny_config(seed=9))
    a.run(2000, 1)
    b.run(2000, 1)
    assert sorted(a.entries()) == sorted(b.entries())


def test_keys_and_export_file(tiny_blueprint):
    t = tiny_blueprint["trainer"]
    sf = read_strategy(tiny_blueprint["path"])
    bp = pe.BlueprintStrategy(str(tiny_blueprint["path"]))
    assert len(sf) == len(bp) > 1000
    assert sf.game["stacks"] == [1000, 1000]
    # Keys decode to (street, bucket, sequence); numpy and Rust readers agree.
    rng = np.random.default_rng(0)
    streets = set()
    for i in rng.choice(len(sf), 200, replace=False):
        key = sf.key(int(i))
        street, bucket, seq = decode_key(key)
        streets.add(street)
        assert (street, bucket, seq) == tuple(pe.decode_infoset_key(key))
        assert make_key(street, bucket, seq) == key == pe.make_infoset_key(street, bucket, seq)
        assert 0 <= bucket < TINY_BUCKETS[street]
        np.testing.assert_allclose(sf.probs(key), bp.lookup(key), atol=1e-4)
        assert sf.probs(key).sum() == pytest.approx(1.0)
    assert streets == {0, 1, 2, 3}
    assert sf.probs(make_key(0, 0, [15, 15])) is None
    # Keys include the acting player's own bucket and the betting sequence:
    # the root infosets of different preflop buckets differ, and the Trainer
    # agrees with the exported averaged strategy.
    root = pe.GameState.new_hand(t.game_config, 0, DECK)
    k0 = t.infoset_key(root, 0)
    assert decode_key(k0) == (0, t.bucket(0, root.hole_cards(0), []), [])
    np.testing.assert_allclose(t.strategy(root, 0), sf.probs(k0), atol=1e-4)
    child = root.child(pe.Action.check_call())
    k1 = t.infoset_key(child, 1)
    assert decode_key(k1)[2] == [1]
    with pytest.raises(ValueError):
        t.infoset_key(root, 1)  # not to act


def test_small_config_trains_via_driver(tmp_path):
    cfg = load_yaml(REPO_ROOT / "configs" / "mccfr_small.yaml")
    sc = solver_config(cfg)
    assert sc["cards"]["buckets"] == [169, 20, 20, 20]
    assert sc["actions"]["max_raises"] == 3
    cfg["mccfr"]["checkpoint"] = {"path": str(tmp_path / "c.bin"), "interval_seconds": 3600}
    cfg["mccfr"]["output"] = {"strategy": str(tmp_path / "s.bin")}
    lines = []
    trainer, summary = train(cfg, "mccfr_small.yaml", iterations=3000, threads=2, log=lines.append)
    assert summary["iterations"] == 3000 and summary["strategy_infosets"] > 0
    assert (tmp_path / "c.bin").exists() and (tmp_path / "s.bin").exists()
    meta = json.loads(read_strategy(tmp_path / "s.bin").meta and trainer.meta)
    assert meta["config_path"] == "mccfr_small.yaml"
    # Resume up to a larger total.
    trainer2, summary2 = train(
        cfg, None, iterations=4000, threads=1, resume=tmp_path / "c.bin", log=lines.append
    )
    assert summary2["iterations"] == 4000


def test_configs_build_trainers():
    for name in ("mccfr_small.yaml", "mccfr_hunl.yaml"):
        cfg = load_yaml(REPO_ROOT / "configs" / name)
        t = pe.Trainer(solver_config(cfg))
        info = json.loads(t.config_json)
        assert info["game"]["stacks"] == [10000, 10000]
        assert t.iterations == 0
    assert info["cards"]["buckets"] == [169, 1000, 1000, 1000]
    assert len(info["actions"]["streets"][1]) == 6
