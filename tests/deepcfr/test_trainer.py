import csv
import math
from pathlib import Path

import pytest
import torch

from pokerbot.blueprint.deepcfr.checkpoint import list_checkpoints
from pokerbot.blueprint.deepcfr.config import DeepCFRConfig
from pokerbot.blueprint.deepcfr.trainer import DeepCFRTrainer

TINY = Path(__file__).resolve().parents[2] / "configs" / "deepcfr_tiny.yaml"


@pytest.fixture(autouse=True)
def _restore_threads():
    # the tiny config pins torch to one thread; do not leak that into other tests
    n = torch.get_num_threads()
    yield
    torch.set_num_threads(n)


def tiny_cfg(**eval_overrides):
    cfg = DeepCFRConfig.load(TINY)
    cfg.logging.print = False
    cfg.eval.every = 1
    cfg.eval.deals = 3
    for k, v in eval_overrides.items():
        setattr(cfg.eval, k, v)
    return cfg


def test_one_iteration_smoke_and_resume(tmp_path):
    cfg = tiny_cfg()
    tr = DeepCFRTrainer(cfg, tmp_path)
    rows = tr.run_iteration(1)
    tr.close()
    assert [r["player"] for r in rows] == [0, 1]
    for r in rows:
        assert r["nodes"] > 0 and math.isfinite(r["loss"]) and r["adv_mem_size"] > 0
        assert r["loss"] < r["loss_first"] * 1.5
    for p in (0, 1):
        assert [t for t, _ in list_checkpoints(tmp_path, p)] == [1]
    assert (tmp_path / "trainer_state.pt").exists()
    assert (tmp_path / "memory" / "adv_p0" / "target.npy").exists()
    with open(tmp_path / "log.csv") as fh:
        assert len(list(csv.DictReader(fh))) == 2
    with open(tmp_path / "eval.csv") as fh:
        ev = list(csv.DictReader(fh))
    assert [e["opponent"] for e in ev] == ["equity"]
    assert any(p.name.startswith("events") for p in (tmp_path / "tb").iterdir())

    # resume: memories, nets and iteration counter come back; iteration 2 runs
    # and evaluates against the previous average strategy too
    tr2 = DeepCFRTrainer(tiny_cfg(), tmp_path, resume=True)
    assert tr2.iteration == 1
    assert len(tr2.adv_mem[0]) == rows[0]["adv_mem_size"]
    assert tr2.nets[0] is not None
    tr2.run(2)
    tr2.close()
    with open(tmp_path / "eval.csv") as fh:
        ev = list(csv.DictReader(fh))
    assert [e["opponent"] for e in ev] == ["equity", "equity", "previous"]
    assert [t for t, _ in list_checkpoints(tmp_path, 1)] == [1, 2]


def test_config_rejects_unknown_keys():
    with pytest.raises(ValueError):
        DeepCFRConfig.from_dict({"training": {"sgd_step": 3}})
    with pytest.raises(ValueError):
        DeepCFRConfig.from_dict({"trainig": {}})
    cfg = DeepCFRConfig.from_dict({})
    assert cfg.net_config().num_actions == 6
