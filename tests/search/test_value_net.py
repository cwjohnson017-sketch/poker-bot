"""River leaf value net: strength buckets, zero-sum layer, predictor, checkpoints,
learning exact check-down values and the shard / CLI path."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch

from pokerbot.search.combos import blocked_sum, combo_index
from pokerbot.search.showdown import combo_strengths, naive_showdown
from pokerbot.search.value_net import (
    BoardFeatureCache,
    RiverValueNet,
    ValueNetConfig,
    ValueNetPredictor,
    board_strengths,
    load_value_net,
    normalise_ranges,
    opponent_mass,
    river_board_features,
    save_value_net,
    zero_sum,
)
from pokerbot.search.value_train import (
    ValueTrainConfig,
    checkdown_samples,
    load_shards,
    save_shard,
    train_value_net,
)

C = 1326
SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "train_value_net.py"


def card(rank: str, suit: int) -> int:
    return "23456789TJQKA".index(rank) * 4 + suit


def _random_boards(n: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, 52, generator=g).argsort(1)[:, :5]


def _noisy_net(cfg: ValueNetConfig, seed: int = 0) -> RiverValueNet:
    """An untrained net with non-zero outputs (the output layer starts at zero)."""
    torch.manual_seed(seed)
    net = RiverValueNet(cfg)
    with torch.no_grad():
        for p in net.parameters():
            p.add_(0.05 * torch.randn_like(p))
    return net.eval()


# --------------------------------------------------------------------------- features


def test_percentile_buckets_match_definition():
    K = 64
    boards = _random_boards(6, 0)
    f = river_board_features(boards, K)
    assert torch.equal(board_strengths(boards), combo_strengths(boards))
    s = combo_strengths(boards)
    for i in range(boards.shape[0]):
        valid = f["valid"][i]
        assert int(valid.sum()) == 1081 and torch.equal(valid, s[i] >= 0)
        assert bool((f["bucket"][i][~valid] == K).all())
        sv = s[i][valid]
        weaker = (sv[None, :] < sv[:, None]).sum(1).double()
        ties = (sv[None, :] == sv[:, None]).sum(1).double()
        pct = (weaker + (ties - 1) / 2) / (1081 - 1)
        assert torch.allclose(f["pct"][i][valid].double(), pct, atol=1e-6)
        assert torch.equal(f["bucket"][i][valid], torch.clamp((K * pct).floor().long(), max=K - 1))
        # ties share a bucket; bucket counts sum to the valid combos
        same = sv[None, :] == sv[:, None]
        b = f["bucket"][i][valid]
        assert bool((b[None, :] == b[:, None])[same].all())
        counts = torch.bincount(b, minlength=K)
        assert counts.shape[0] == K and int(counts.sum()) == 1081


def test_nuts_in_top_bucket_and_board_ties():
    K = 256
    # A K Q of one suit: the J-T of that suit is the only royal flush
    board = torch.tensor([[card("A", 0), card("K", 0), card("Q", 0), card("2", 1), card("3", 2)]])
    f = river_board_features(board, K)
    nuts = combo_index(card("J", 0), card("T", 0))
    assert float(f["pct"][0, nuts]) == 1.0 and int(f["bucket"][0, nuts]) == K - 1
    assert int((f["pct"][0] == 1.0).sum()) == 1
    # a royal flush on the board: every combo plays the board and ties
    royal = torch.tensor([[card(r, 0) for r in "AKQJT"]])
    f = river_board_features(royal, K)
    v = f["valid"][0]
    assert bool((f["pct"][0][v] == 0.5).all()) and bool((f["bucket"][0][v] == K // 2).all())


def test_board_feature_cache():
    boards = _random_boards(5, 1)
    rows = boards[torch.tensor([0, 1, 2, 3, 4, 2, 0])]
    rows = rows[:, torch.randperm(5)]  # card order does not matter
    ref = river_board_features(rows, 32)
    for tables in (True, False):
        cache = BoardFeatureCache("cpu", tables=tables)
        ids = cache.ids(rows)
        assert len(cache) == 5 and ids[0] == ids[6] and ids[2] == ids[5]
        for K in (32, 7):  # two bucket tables side by side
            f = cache.features(ids, K)
            ref = river_board_features(rows, K)
            assert f["bucket"].dtype == torch.int32
            for k in ("valid", "pct", "bucket", "onehot"):
                assert torch.equal(f[k].to(ref[k].dtype), ref[k]), k
        more = _random_boards(1500, 9)  # grows the tables past their capacity
        ids2 = cache.ids(more)
        assert len(cache) == 1505 and torch.equal(cache.ids(rows), ids)
        f = cache.features(ids2, 32)
        assert torch.equal(f["bucket"].long(), river_board_features(more, 32)["bucket"])
    small = BoardFeatureCache("cpu", max_boards=3)
    small.ids(boards[:3])
    gen = small.generation
    ids = small.ids(boards[3:])  # would exceed max_boards: reset first
    assert small.generation == gen + 1 and len(small) == 2 and ids.tolist() == [0, 1]


# --------------------------------------------------------------------------- model


def _random_ranges(boards: torch.Tensor, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    n = boards.shape[0]
    r = torch.rand(n, 2, C, generator=g) ** 3
    r = r * (torch.rand(n, 2, C, generator=g) < 0.3)  # sparse
    return r


def test_opponent_mass_is_blocked_sum():
    boards = _random_boards(4, 10)
    valid = river_board_features(boards, 8)["valid"]
    r = normalise_ranges(_random_ranges(boards, 11), valid)
    m = opponent_mass(r, valid)
    for p in (0, 1):
        ref = blocked_sum(r[:, 1 - p].double()) * valid
        assert torch.allclose(m[:, p].double(), ref, atol=1e-6)
    assert torch.allclose(r.sum(-1), torch.ones(4, 2), atol=1e-6)


def test_zero_sum_layer():
    boards = _random_boards(16, 2)
    f = river_board_features(boards, 64)
    r = normalise_ranges(_random_ranges(boards, 3), f["valid"])
    m = opponent_mass(r, f["valid"])
    w = r * m
    # the two pair masses agree
    assert torch.allclose(w[:, 0].sum(-1), w[:, 1].sum(-1), atol=1e-6)
    ev = zero_sum(torch.randn(16, 2, C, dtype=torch.float64), r.double(), m.double())
    assert (r.double() * m.double() * ev).sum((1, 2)).abs().max() < 1e-12
    c, stack = torch.full((16,), 300), torch.full((16,), 9700)
    invalid = ~f["valid"][:, None, :].expand(16, 2, C)
    for head in (False, True):
        net = _noisy_net(ValueNetConfig(buckets=64, width=64, layers=2, residual_head=head))
        with torch.no_grad():
            ev = net(r, f["bucket"], f["onehot"], c, stack, pct=f["pct"])
        assert ev.abs().mean() > 0.01
        gv = (w * ev).sum((1, 2))
        assert gv.abs().max() < 1e-5 * ev.abs().max(), gv
        assert bool((ev[invalid] == 0).all())


def test_predictor_handles_zero_and_scaled_ranges():
    cfg = ValueNetConfig(buckets=32, width=64, layers=2, residual_head=True)
    pred = ValueNetPredictor(_noisy_net(cfg), "cpu")
    boards = _random_boards(4, 4)
    valid = river_board_features(boards, 32)["valid"]
    r = _random_ranges(boards, 5)
    r[0, 0] = 0.0  # one empty range
    r[1] = 0.0  # both empty
    c = torch.tensor([100, 400, 2500, 5000])
    stack = 10000 - c
    ev = pred.predict(boards, r, c, stack, chunk=3)
    assert ev.shape == (4, 2, C) and ev.dtype == torch.float32
    assert bool(torch.isfinite(ev).all())
    assert bool((ev[~valid[:, None, :].expand(4, 2, C)] == 0).all())
    uni = r.clone()
    uni[0, 0] = valid[0].float()
    uni[1] = valid[1].float()
    assert torch.allclose(ev, pred.predict(boards, uni * 3.0, c, stack), atol=1e-5)
    ids = pred.board_ids(boards)
    assert torch.equal(pred.predict_ids(ids, r, c, stack, chunk=3), ev)
    # zero-stack (all-in) rows are fine too
    assert bool(torch.isfinite(pred.predict(boards, r, c, 0 * stack)).all())


def test_checkpoint_round_trip(tmp_path):
    for i, cfg in enumerate(
        [
            ValueNetConfig(buckets=32, width=48, layers=2),
            ValueNetConfig(
                buckets=16, width=32, layers=3, block="resnet", residual_head=True, count_input=True
            ),
        ]
    ):
        net = _noisy_net(cfg, seed=i)
        path = tmp_path / f"vn{i}.pt"
        save_value_net(path, net, {"note": "x"})
        loaded = load_value_net(path)
        assert loaded.cfg == cfg and loaded.meta == {"note": "x"}
        boards = _random_boards(3, 6)
        r = _random_ranges(boards, 7)
        c, stack = torch.tensor([150, 800, 3000]), torch.tensor([9850, 9200, 7000])
        a = ValueNetPredictor(net, "cpu").predict(boards, r, c, stack)
        b = ValueNetPredictor.from_path(path, "cpu").predict(boards, r, c, stack)
        assert torch.equal(a, b)


# --------------------------------------------------------------------------- learning


def test_checkdown_targets_are_exact():
    d = checkdown_samples(3, seed=8)
    for i in range(3):
        board = d["boards"][i].tolist()
        valid = river_board_features(d["boards"][i : i + 1], 2)["valid"][0]
        for p in (0, 1):
            opp = d["ranges"][i, 1 - p].double()
            m = blocked_sum(opp) * valid
            sd = naive_showdown(opp[None], board)[0]
            ref = torch.where(m > 1e-9, 0.5 * sd / m.clamp(min=1e-12), torch.zeros_like(sd))
            assert torch.allclose(d["targets"][i, p].double(), ref, atol=1e-5)


def test_learns_checkdown_values():
    torch.manual_seed(0)
    data = checkdown_samples(600, seed=11)
    net_cfg = ValueNetConfig(buckets=64, width=256, layers=3)
    cfg = ValueTrainConfig(steps=300, batch=64, warmup=20, holdout=0.15, seed=1)
    _, rep = train_value_net(None, None, net_cfg, cfg, device="cpu", log=None, raw=data)
    o = rep["overall"]
    assert o["samples"] > 50
    assert o["mae"] < 0.2 * o["zero"]["mae"], o
    assert o["wmae"] < 0.2 * o["zero"]["wmae"], o
    assert o["oracle"]["mae"] < o["mae"]
    assert abs(rep["gv_target_sum_mean"]) < 1e-3  # check-down targets are zero-sum
    assert set(rep["by_pot"]) and set(rep["by_source"]) == {"blueprint", "perturbed", "random"}


def test_shards_and_cli(tmp_path):
    data = checkdown_samples(160, seed=12)
    root = tmp_path / "data"
    save_shard(root / "shard_000.pt", {k: v[:100] for k, v in data.items()})
    save_shard(root / "shard_001.pt", {k: v[100:] for k, v in data.items()})
    # a slice of a larger tensor is stored on its own: ~10.7 kB per sample
    assert (root / "shard_001.pt").stat().st_size < 60 * 12_000
    (root / "meta.json").write_text(json.dumps({"kind": "checkdown"}))
    held = checkdown_samples(40, seed=13)
    save_shard(tmp_path / "held" / "shard_000.pt", held)
    loaded = load_shards(root)
    assert loaded["ranges"].dtype == torch.float16 and loaded["boards"].dtype == torch.uint8
    assert loaded["c"].shape == (160,) and torch.equal(loaded["boards"].long(), data["boards"])
    with pytest.raises(KeyError):
        save_shard(tmp_path / "bad.pt", {"boards": data["boards"]})

    spec = importlib.util.spec_from_file_location("train_value_net", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = tmp_path / "vn" / "river.pt"
    argv = ["--data", str(root), "--heldout-data", str(tmp_path / "held"), "--out", str(out)]
    argv += ["--device", "cpu", "--steps", "20", "--batch", "16", "--buckets", "32"]
    argv += ["--width", "64", "--layers", "2", "--residual-head", "--quiet"]
    assert mod.main(argv) == 0
    rep = json.loads(out.with_suffix(".json").read_text())
    assert rep["train_samples"] == 160 and rep["heldout_samples"] == 40
    assert rep["overall"]["mae"] < rep["overall"]["zero"]["mae"]
    pred = ValueNetPredictor.from_path(out, "cpu")
    assert pred.net.cfg.residual_head and pred.buckets == 32
    ev = pred.predict(held["boards"][:4], held["ranges"][:4], held["c"][:4], held["stack"][:4])
    assert bool(torch.isfinite(ev).all())
