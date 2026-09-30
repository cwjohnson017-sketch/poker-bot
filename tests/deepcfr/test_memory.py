import numpy as np
import torch

from pokerbot.blueprint.deepcfr.memory import ReservoirMemory


def _batch(ids, A=4, S=5, T=6):
    m = len(ids)
    ids = np.asarray(ids)
    return {
        "cards": np.tile((ids % 52)[:, None], (1, 7)).astype(np.uint8),
        "hist": np.tile((ids % 200)[:, None], (1, T)),
        "hist_amt": np.tile(ids[:, None] / 1000.0, (1, T)),
        "scalars": np.tile(ids[:, None].astype(np.float32), (1, S)),
        "legal": np.ones((m, A), bool),
        "target": np.tile((ids % 1000)[:, None].astype(np.float32), (1, A)),
        "iteration": (ids % 7) + 1,
    }


def _mem(cap, seed):
    return ReservoirMemory(cap, 4, 5, 6, vocab_size=64, seed=seed)


def test_uniform_inclusion_probability():
    cap, total, trials = 10, 60, 3000
    counts = np.zeros(total)
    for trial in range(trials):
        mem = _mem(cap, trial)
        # uneven batch sizes, including batches larger than the capacity
        start = 0
        for size in (3, 14, 1, 25, 17):
            mem.add_batch(_batch(np.arange(start, start + size)))
            start += size
        assert start == total and len(mem) == cap and mem.seen == total
        held = mem.arrays["scalars"][:, 0].astype(int)
        assert len(set(held.tolist())) == cap  # no duplicates
        counts[held] += 1
    freq = counts / trials
    expected = cap / total
    # binomial sd = sqrt(p(1-p)/trials) ~ 0.0068; allow ~5 sd
    assert np.abs(freq - expected).max() < 0.035, freq
    # first and last halves of the stream are held equally often
    assert abs(freq[:30].mean() - freq[30:].mean()) < 0.01


def test_fill_then_sample_and_stats():
    mem = _mem(100, 0)
    mem.add_batch(_batch(np.arange(40)))
    assert len(mem) == 40
    b = mem.sample(64, "cpu")
    assert b["cards"].dtype == torch.long and b["cards"].shape == (64, 7)
    assert b["target"].dtype == torch.float32 and b["legal"].dtype == torch.bool
    assert b["card_mask"].all()
    assert (b["scalars"][:, 0] < 40).all()
    st = mem.stats()
    assert st["size"] == 40 and st["seen"] == 40 and st["bytes_per_sample"] > 0
    assert 1 <= st["mean_iteration"] <= 7


def test_save_load_roundtrip(tmp_path):
    mem = _mem(50, 1)
    mem.add_batch(_batch(np.arange(130)))
    mem.save(tmp_path / "m")
    back = ReservoirMemory.load(tmp_path / "m")
    assert len(back) == 50 and back.seen == 130
    for k in mem.arrays:
        np.testing.assert_array_equal(back.arrays[k][:50], mem.arrays[k][:50])
    # memory-mapped load of a full buffer, then keep adding (copy-on-write)
    mm = ReservoirMemory.load(tmp_path / "m", mmap=True)
    assert isinstance(mm.arrays["cards"], np.memmap)
    mm.add_batch(_batch(np.arange(1000, 1100)))
    assert mm.seen == 230
    again = ReservoirMemory.load(tmp_path / "m")
    np.testing.assert_array_equal(again.arrays["scalars"], mem.arrays["scalars"])
    # partial buffer into a larger capacity
    small = _mem(50, 2)
    small.add_batch(_batch(np.arange(20)))
    small.save(tmp_path / "s")
    big = ReservoirMemory.load(tmp_path / "s", capacity=80)
    assert len(big) == 20 and big.capacity == 80
    big.add_batch(_batch(np.arange(100, 170)))
    assert len(big) == 80


def test_chunk_sampler_draws_uniform_minibatches():
    from pokerbot.blueprint.deepcfr.trainer import _ChunkSampler

    mem = _mem(50, 0)
    mem.add_batch(_batch(np.arange(50)))
    sampler = _ChunkSampler(mem, batch=10, device=torch.device("cpu"), rows=40, steps=400)
    counts = np.zeros(50)
    for _ in range(400):
        b = sampler.get()
        assert b["cards"].dtype == torch.long and b["target"].dtype == torch.float32
        assert b["scalars"].shape == (10, 5) and b["legal"].dtype == torch.bool
        ids = b["scalars"][:, 0].long().numpy()
        counts[ids] += 1
    freq = counts / counts.sum()
    assert np.abs(freq - 1 / 50).max() < 0.012  # 4000 draws over 50 rows
