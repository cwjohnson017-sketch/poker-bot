"""Bucket generation: features, EMD k-means, tables in the engine's format."""

from __future__ import annotations

import json

import numpy as np
import poker_engine as pe
import pytest
import torch

from pokerbot.abstraction.buckets import (
    BuildConfig,
    StreetConfig,
    assign,
    build_street,
    centre_equity,
    kmeans,
    load_config,
    main,
    num_buckets,
    street_features,
    table_paths,
    to_cdf,
)
from pokerbot.abstraction.isomorphism import unindex_batch
from pokerbot.config import REPO_ROOT


def test_tiny_config_produces_valid_tables(tiny_tables):
    bc, results = tiny_tables["config"], tiny_tables["results"]
    assert set(results) == {"flop", "turn", "river"}
    for street, name in ((1, "flop"), (2, "turn"), (3, "river")):
        r = results[name]
        k = bc.streets[name].buckets
        table = np.load(r.table_path, mmap_mode="r")
        assert table.shape == (pe.canonical_size(street),) and table.dtype == np.uint16
        assert int(table.max()) < k
        # The computed entries hold the fitted buckets; all buckets are used.
        assert np.array_equal(table[r.indices], r.labels)
        assert len(np.unique(r.labels)) == k
        info = json.loads((r.table_path.with_suffix(".json")).read_text())
        assert info["computed"] == bc.limit and info["padded"] == table.shape[0] - bc.limit
        assert info["git"] and info["config"]["street"]["buckets"] == k
        assert 0.3 < info["features"]["equity_mean_weighted"] < 0.7
        # Buckets are ordered by strength.
        eq = centre_equity(torch.tensor(info["centres"]), "l2" if street == 3 else "emd")
        assert (eq.diff() >= 0).all()
    paths = table_paths(tiny_tables["out_dir"])
    assert num_buckets(tiny_tables["out_dir"]) == [None, 8, 8, 8]
    ca = pe.CardAbstraction(buckets=[169, 8, 8, 8], tables=paths)
    rng = np.random.default_rng(0)
    for _ in range(100):
        c = rng.permutation(52).tolist()
        for street, nb in ((1, 3), (2, 4), (3, 5)):
            b = ca.bucket(street, c[:2], c[2 : 2 + nb])
            table = np.load(paths[street], mmap_mode="r")
            assert b == table[pe.canonical_index(street, c[:2], c[2 : 2 + nb])] < 8


def test_river_buckets_are_monotone_in_equity(tiny_tables):
    r = tiny_tables["results"]["river"]
    order = np.argsort(r.equity, kind="stable")
    assert (np.diff(r.labels[order].astype(int)) >= 0).all()
    means = [
        np.average(r.equity[r.labels == b], weights=r.weights[r.labels == b]) for b in range(8)
    ]
    assert (np.diff(means) > 0).all()


def test_street_features_are_exact():
    rng = np.random.default_rng(1)
    # River: exact equity vs a random hand, as the Rust hand_strength computes it.
    reps = unindex_batch(3, rng.integers(0, pe.canonical_size(3), 20))
    f, eq = street_features(3, torch.from_numpy(reps))
    assert f.shape == (20, 1)
    for row, e in zip(reps.tolist(), eq.tolist(), strict=True):
        assert e == pytest.approx(pe.hand_strength(row[:2], row[2:]), abs=1e-6)
    # Turn: histogram over all 46 rivers, each with exact equity.
    reps = unindex_batch(2, rng.integers(0, pe.canonical_size(2), 3))
    hist, eq = street_features(2, torch.from_numpy(reps), bins=10)
    for row, h, e in zip(reps.tolist(), hist, eq.tolist(), strict=True):
        rivers = [c for c in range(52) if c not in row]
        per = [pe.hand_strength(row[:2], row[2:] + [c]) for c in rivers]
        assert e == pytest.approx(float(np.mean(per)), abs=1e-5)
        want = np.bincount(np.minimum((np.array(per) * 10).astype(int), 9), minlength=10) / 46
        assert np.allclose(h.numpy(), want, atol=1e-6)
    # Flop, sampled runouts: rows are distributions; the mean tracks Monte Carlo.
    reps = unindex_batch(1, rng.integers(0, pe.canonical_size(1), 4))
    g = torch.Generator().manual_seed(0)
    hist, eq = street_features(1, torch.from_numpy(reps), bins=10, runouts=64, generator=g)
    assert torch.allclose(hist.sum(1), torch.ones(4))
    for row, e in zip(reps.tolist(), eq.tolist(), strict=True):
        assert e == pytest.approx(pe.hand_strength(row[:2], row[2:], 20000, 1), abs=0.06)


def test_emd_is_l1_between_cdfs():
    bins = 10
    eye = torch.eye(bins)
    d = torch.cdist(to_cdf(eye), to_cdf(eye), p=1)
    ij = torch.arange(bins)
    assert torch.equal(d, (ij[:, None] - ij[None, :]).abs().float())
    # Half the mass moved by 4 bins costs 2.
    a = torch.zeros(1, bins)
    a[0, 2] = 1
    b = torch.zeros(1, bins)
    b[0, 2] = b[0, 6] = 0.5
    assert float(torch.cdist(to_cdf(a), to_cdf(b), p=1)) == pytest.approx(2.0)


def test_emd_kmeans_recovers_separated_clusters():
    g = torch.Generator().manual_seed(0)
    bins, per = 10, 150
    modes = [(1,), (4, 5), (8,), (0, 9)]  # the last one is bimodal
    xs, truth = [], []
    for c, m in enumerate(modes):
        base = torch.zeros(bins)
        base[list(m)] = 1.0 / len(m)
        noise = torch.rand(per, bins, generator=g) * 0.04
        h = base + noise
        xs.append(h / h.sum(1, keepdim=True))
        truth += [c] * per
    x = torch.cat(xs)
    truth = torch.tensor(truth)
    w = torch.randint(1, 25, (x.shape[0],), generator=g).float()
    centres, hist = kmeans(x, 4, w, "emd", iters=20, generator=g)
    assert centres.shape == (4, bins - 1)
    assert hist[-1] <= hist[0]
    labels, _ = assign(to_cdf(x), centres, "emd")
    for c in range(4):
        got = labels[truth == c]
        assert (got == got[0]).all()  # each true cluster lands in one bucket
    assert len(set(labels.tolist())) == 4


def test_one_dimensional_kmeans_matches_sorted_quantiles():
    g = torch.Generator().manual_seed(0)
    x = torch.cat([torch.rand(300, generator=g) * 0.1 + c for c in (0.0, 0.45, 0.9)])[:, None]
    centres, _ = kmeans(x, 3, None, "l2", iters=30)
    assert torch.allclose(centres[:, 0].sort().values, torch.tensor([0.05, 0.5, 0.95]), atol=0.01)


def test_sample_mode_fits_on_a_subset_and_assigns_all(tmp_path):
    bc = BuildConfig(
        streets={"flop": StreetConfig(buckets=5, runouts=8, sample=40, iters=10, chunk=64)},
        out_dir=str(tmp_path),
        limit=120,
        feature_cache=True,
    )
    r = build_street("flop", bc, log=lambda _m: None)
    assert r.info["fit"]["rows"] == 40 and r.labels.shape == (120,)
    assert int(r.labels.max()) < 5
    table = np.load(r.table_path, mmap_mode="r")
    assert np.array_equal(table[r.indices], r.labels)
    del table  # Windows cannot replace a file that is still memory-mapped
    # A second run reuses the cached features and gives the same table.
    logs: list[str] = []
    r2 = build_street("flop", bc, log=logs.append)
    assert any("reusing cached features" in m for m in logs)
    assert np.array_equal(r2.labels, r.labels)


def test_configs_parse_and_cli_smoke(tmp_path, capsys):
    hunl = load_config(REPO_ROOT / "configs" / "buckets_hunl.yaml")
    assert [hunl.streets[s].buckets for s in ("flop", "turn", "river")] == [1000] * 3
    assert all(hunl.streets[s].sample for s in ("flop", "turn", "river"))
    assert hunl.limit is None and hunl.device == "cuda"
    tiny = load_config(REPO_ROOT / "configs" / "buckets_tiny.yaml")
    assert tiny.limit and tiny.device == "cpu"
    with pytest.raises(ValueError):
        BuildConfig.from_dict({"streets": {"flop": {"buckets": 3, "bogus": 1}}})
    code = main(
        [
            "--config",
            str(REPO_ROOT / "configs" / "buckets_tiny.yaml"),
            "--streets",
            "flop",
            "--limit",
            "40",
            "--out-dir",
            str(tmp_path),
        ]
    )
    assert code == 0 and (tmp_path / "flop.npy").exists() and (tmp_path / "flop.json").exists()
    assert "tables:" in capsys.readouterr().out
