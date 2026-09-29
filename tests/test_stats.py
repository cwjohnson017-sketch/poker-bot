import numpy as np

from pokerbot.eval.stats import bootstrap_ci, mbb_per_hand, win_rate


def test_mbb_per_hand():
    assert mbb_per_hand([100, -100, 300], 100) == 1000.0
    assert mbb_per_hand([200, 0], 100, hands_per_sample=2) == 500.0


def test_bootstrap_ci_matches_normal_theory():
    rng = np.random.default_rng(0)
    x = rng.normal(5.0, 10.0, size=4000)
    lo, hi = bootstrap_ci(x, n_boot=2000, rng=1)
    se = x.std() / np.sqrt(x.size)
    assert lo < x.mean() < hi
    assert abs((hi - lo) - 2 * 1.96 * se) < 0.15 * (2 * 1.96 * se)


def test_bootstrap_ci_coverage():
    rng = np.random.default_rng(1)
    hits = 0
    trials = 150
    for t in range(trials):
        x = rng.exponential(2.0, size=300) - 2.0  # skewed, true mean 0
        lo, hi = bootstrap_ci(x, n_boot=400, rng=t)
        hits += lo <= 0.0 <= hi
    assert 0.88 <= hits / trials <= 0.99


def test_win_rate_scaling_and_significance():
    rng = np.random.default_rng(2)
    x = rng.normal(50.0, 200.0, size=5000)  # chips per duplicate pair
    wr = win_rate(x, big_blind=100, hands_per_sample=2)
    assert wr.hands == 10000
    assert abs(wr.mbb_per_hand - 1000 * x.mean() / 200) < 1e-9
    assert wr.ci_low < wr.mbb_per_hand < wr.ci_high
    assert wr.significant()
    assert "mbb/h" in str(wr)
    zero = win_rate(np.zeros(10), big_blind=100)
    assert zero.mbb_per_hand == 0 and zero.ci_low == 0 and zero.ci_high == 0
    assert not zero.significant()


def test_bootstrap_chunking_large_input():
    x = np.ones(300_000)
    lo, hi = bootstrap_ci(x, n_boot=50)
    assert lo == hi == 1.0
