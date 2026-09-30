"""Table-lookup hand-strength features (:mod:`pokerbot.blueprint.deepcfr.strength`)."""

import numpy as np
import pytest
import torch

pytest.importorskip("poker_engine")

from pokerbot.blueprint.deepcfr import strength  # noqa: E402
from pokerbot.blueprint.deepcfr.config import DeepCFRConfig  # noqa: E402
from pokerbot.blueprint.deepcfr.features import FeatureConfig  # noqa: E402
from pokerbot.blueprint.deepcfr.strength import (  # noqa: E402
    NUM_STRENGTH,
    StrengthTables,
    add_strength,
)
from pokerbot.blueprint.deepcfr.traversal import (  # noqa: E402
    FrontierTraverser,
    TraversalConfig,
    uniform_policy,
)
from pokerbot.config import REPO_ROOT  # noqa: E402
from pokerbot.env import GameConfig  # noqa: E402
from pokerbot.env.cards import NO_CARD, cards_from_str, make_generator, shuffled_decks  # noqa: E402


class HashTables(StrengthTables):
    """Deterministic stand-in derived from the canonical index (no files)."""

    def __init__(self) -> None:
        super().__init__(None)

    def preflop(self) -> np.ndarray:
        return np.linspace(0.3, 0.85, 169, dtype=np.float32)

    def street_rows(self, street: int, idx: np.ndarray) -> np.ndarray:
        out = np.zeros((len(idx), NUM_STRENGTH), np.float32)
        out[:, 0] = (idx % 1009) / 1009 + street
        if street < 3:
            out[:, 1:] = ((idx[:, None] * np.arange(1, 11)) % 17) / 17
        return out


@pytest.fixture
def fake():
    strength._CACHE["hash"] = HashTables()
    yield "hash"
    strength._CACHE.pop("hash", None)


def test_lookup_is_suit_invariant(fake):
    t = strength.load_strength(fake)
    hole = np.array([cards_from_str("AsKs"), cards_from_str("AhKh"), cards_from_str("AsKh")])
    flop = [cards_from_str(b) + [NO_CARD] * 2 for b in ("QsJs2d", "QhJh2d", "QsJs2d")]
    board = np.array(flop)  # AhKh on QhJh2d mirrors AsKs on QsJs2d
    f = t.lookup(hole, board, np.array([1, 1, 1]))
    assert np.array_equal(f[0], f[1])
    assert not np.array_equal(f[0], f[2])
    pre = t.lookup(hole, board, np.array([0, 0, 0]))
    assert pre[0, 0] == pre[1, 0] and pre[0, 0] != pre[2, 0]  # suited vs offsuit
    assert (pre[:, 1:] == 0).all()
    with pytest.raises(ValueError):  # the turn is not dealt
        t.lookup(hole, board, np.array([2, 2, 2]))


def test_root_features_layout(fake):
    t = strength.load_strength(fake)
    d = shuffled_decks(5, make_generator(3))[:, :9].numpy()
    R = t.root_features(d)
    assert R.shape == (5, 2, 4, NUM_STRENGTH)
    for seat in (0, 1):
        hole = d[:, 2 * seat : 2 * seat + 2]
        for s, k in enumerate((0, 3, 4, 5)):
            board = np.full((5, 5), NO_CARD)
            board[:, :k] = d[:, 4 : 4 + k]
            np.testing.assert_array_equal(R[:, seat, s], t.lookup(hole, board, np.full(5, s)))


def test_traversal_appends_the_same_columns_as_add_strength(fake):
    fc = FeatureConfig(strength_tables=fake)
    tcfg = TraversalConfig(features=fc)
    tr = FrontierTraverser(GameConfig(stacks=[2000, 2000]), cfg=tcfg, seed=1)
    res = tr.traverse(0, [uniform_policy, uniform_policy], 1, 64)
    s = res.samples
    assert s["scalars"].shape[1] == fc.num_scalars == 14 + NUM_STRENGTH
    feats = {"cards": s["cards"].long(), "scalars": s["scalars"][:, :14].float()}
    want = add_strength(feats, strength.load_strength(fake))["scalars"][:, 14:]
    torch.testing.assert_close(s["scalars"][:, 14:].float(), want.half().float())


def test_training_and_play_with_strength_features(fake, tmp_path):
    from pokerbot.agents import RandomAgent
    from pokerbot.blueprint.deepcfr.agent import NeuralBlueprintAgent
    from pokerbot.blueprint.deepcfr.trainer import DeepCFRTrainer
    from pokerbot.config import game_config
    from pokerbot.engine_select import get_engine
    from pokerbot.eval.match import run_match

    cfg = DeepCFRConfig.load(REPO_ROOT / "configs" / "deepcfr_tiny.yaml")
    cfg.logging.print = False
    cfg.logging.tensorboard = False
    cfg.eval.every = 0
    cfg.features = FeatureConfig(strength_tables=fake)
    threads = torch.get_num_threads()
    try:
        tr = DeepCFRTrainer(cfg, tmp_path)
        tr.run_iteration(1)
        tr.close()
    finally:
        torch.set_num_threads(threads)
    assert tr.nets[0].cfg.num_scalars == 14 + NUM_STRENGTH
    engine = get_engine()
    config = game_config(tr.meta["game"], engine)
    agent = NeuralBlueprintAgent.from_dir(tmp_path)
    res = run_match([agent, RandomAgent()], config, 20, seed=2, engine=engine)
    assert res.hands == 20
    # the stateless range query recomputes the columns for every hand
    state = engine.GameState.new_hand(config, 0, list(range(52)))
    probs = agent.policy_all(state, 0)
    assert probs.shape[0] == 1326 and torch.isfinite(probs).all()


def test_real_bucket_tables_if_built():
    path = REPO_ROOT / "data" / "abstraction" / "buckets_hunl"
    if not (path / "features" / "river_equity.npy").exists():
        pytest.skip("bucket build not present")
    t = StrengthTables(path)
    hole = np.array([cards_from_str("AsKs"), cards_from_str("7c2d")])
    board = np.array([cards_from_str("QsJsTs3h2h")] * 2)
    f = t.lookup(hole, board, np.array([3, 3]))
    assert f[0, 0] > 0.99 and f[1, 0] < 0.3  # royal flush vs seven high
    b3 = np.array([cards_from_str("QsJs3h") + [NO_CARD] * 2] * 2)
    flop = t.lookup(hole, b3, np.array([1, 1]))
    assert abs(flop[0, 1:].sum() - 1) < 1e-4 and flop[0, 0] > flop[1, 0]
