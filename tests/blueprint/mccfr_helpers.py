"""Shared helpers for the MCCFR blueprint tests."""

from __future__ import annotations

# A heavily reduced game: 10 big blinds, fold / call / pot / all-in on every
# street, two raises per street, 8 buckets per street. Its abstract betting
# tree has 248 decision nodes, so exact tree walks in Python are cheap.
TINY_BUCKETS = [8, 8, 8, 8]
TINY_HS_SAMPLES = 32


def tiny_config(seed: int = 3, **overrides) -> dict:
    cfg = {
        "game": {"num_players": 2, "stacks": [1000, 1000], "small_blind": 50, "big_blind": 100},
        "actions": {
            "streets": [["fold", "check_call", ("raise", 1.0), "allin"]] * 4,
            "max_raises": 2,
        },
        "cards": {"buckets": TINY_BUCKETS, "hs_samples": TINY_HS_SAMPLES},
        "lcfr_discount_every": 2000,
        "lcfr_stop": 100_000,
        "prune_start": 50_000,
        "prune_threshold": -20.0,
        "regret_floor": -25.0,
        "shards": 64,
        "seed": seed,
    }
    cfg.update(overrides)
    return cfg
