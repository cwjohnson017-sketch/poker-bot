"""The tabular MCCFR solver trains on generated bucket tables (``cards.tables``)."""

from __future__ import annotations

import time

import numpy as np
import poker_engine as pe

from pokerbot.abstraction.buckets import table_paths
from pokerbot.blueprint.mccfr.train import solver_config, train
from pokerbot.config import REPO_ROOT, load_yaml


def test_mccfr_trains_on_generated_tables(tiny_tables, tmp_path):
    paths = table_paths(tiny_tables["out_dir"])
    cfg = load_yaml(REPO_ROOT / "configs" / "mccfr_small.yaml")
    cards = cfg["mccfr"]["cards"]
    cards["buckets"] = [169, 8, 8, 8]
    cards["tables"] = {"preflop": None, "flop": paths[1], "turn": paths[2], "river": paths[3]}
    cfg["mccfr"]["checkpoint"] = {}
    cfg["mccfr"]["output"] = {"strategy": str(tmp_path / "strategy.bin")}
    assert solver_config(cfg)["cards"]["tables"] == paths

    # A couple of seconds of training through the driver.
    trainer, summary = train(cfg, None, iterations=1000, threads=2, log=lambda _m: None)
    t0 = time.time()
    while time.time() - t0 < 2.0:
        trainer.run(2000, 2)
    per_street = trainer.stats(detailed=True)["infosets_per_street"]
    assert all(n > 0 for n in per_street)

    # Every infoset's bucket is in range for its street.
    limits = [169, 8, 8, 8]
    seen = [set(), set(), set(), set()]
    for key, _regrets, _sums in trainer.entries():
        street, bucket, _seq = pe.decode_infoset_key(key)
        assert bucket < limits[street]
        seen[street].add(bucket)
    assert all(len(seen[s]) == 8 for s in (1, 2, 3))

    # The solver's buckets are the table entries of the canonical index.
    tables = [None] + [np.load(p, mmap_mode="r") for p in paths[1:]]
    rng = np.random.default_rng(0)
    for _ in range(300):
        c = rng.permutation(52).tolist()
        for street, nb in ((1, 3), (2, 4), (3, 5)):
            hole, board = c[:2], c[2 : 2 + nb]
            want = int(tables[street][pe.canonical_index(street, hole, board)])
            assert trainer.bucket(street, hole, board) == want

    # The exported strategy carries the same abstraction.
    bp = pe.BlueprintStrategy(summary["strategy_path"])
    assert bp.bucket(3, c[:2], c[2:7]) == trainer.bucket(3, c[:2], c[2:7])
