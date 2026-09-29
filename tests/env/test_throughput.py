"""Throughput report (printed, not asserted). Run with ``-s`` to see it."""

import time

import torch

from pokerbot.env import GameConfig, VecNLHE
from pokerbot.env.cards import make_generator, shuffled_decks
from pokerbot.env.equity import equity_river
from pokerbot.env.evaluator import evaluate7_batch


def test_throughput_report():
    n, steps = 4096, 100
    env = VecNLHE(n, GameConfig(), "cpu", seed=0, validate=False)
    g = torch.Generator().manual_seed(0)
    # warm-up
    for _ in range(5):
        env.step(torch.multinomial(env.legal_mask().float(), 1, generator=g).squeeze(1))
        env.reset(env.done)
    hands = 0
    t_mask = t_sample = t_step = t_reset = 0.0
    for _ in range(steps):
        t0 = time.perf_counter()
        mask = env.legal_mask()
        t1 = time.perf_counter()
        a = torch.multinomial(mask.float(), 1, generator=g).squeeze(1)
        t2 = time.perf_counter()
        _, done = env.step(a)
        t3 = time.perf_counter()
        env.reset(done)
        t4 = time.perf_counter()
        hands += int(done.sum())
        t_mask, t_sample, t_step, t_reset = t_mask + t1 - t0, t_sample + t2 - t1, t_step + t3 - t2, t_reset + t4 - t3
    total = t_mask + t_sample + t_step + t_reset
    print(
        f"\nVecNLHE n={n} CPU ({torch.get_num_threads()} threads): {steps * n / total:,.0f} steps/s "
        f"(legal_mask + sample + step + reset), {hands / total:,.0f} hands/s; "
        f"step() alone {steps * n / t_step:,.0f} steps/s; time split mask/sample/step/reset = "
        f"{t_mask / total:.0%}/{t_sample / total:.0%}/{t_step / total:.0%}/{t_reset / total:.0%}"
    )

    cards = shuffled_decks(1 << 20, make_generator(1))[:, :7]
    t0 = time.perf_counter()
    evaluate7_batch(cards)
    dt = time.perf_counter() - t0
    print(f"evaluate7_batch: {cards.shape[0] / dt:,.0f} hands/s")

    d = shuffled_decks(1024, make_generator(2))
    t0 = time.perf_counter()
    equity_river(d[:, :2], d[:, 2:7])
    dt = time.perf_counter() - t0
    print(f"equity_river: {1024 / dt:,.0f} hands/s ({1024 * 990 / dt:,.0f} evals/s)")
