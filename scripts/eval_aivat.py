#!/usr/bin/env python
"""Validate AIVAT's action terms on a small CPU search bot, without duplicate dealing.

    python scripts/eval_aivat.py                  # equity, random, always_call; 3000 hands each
    python scripts/eval_aivat.py --opponents equity --hands 500 --workers 4
    python scripts/eval_aivat.py --iterations 0   # the test config's clock: 2-4 iterations

A recording ``SearchAgent`` (uniform blueprint, the tiny CPU config of
``tests/search/test_agent.py`` with ``--iterations`` DCFR iterations per solve,
unless ``--search-config`` names a YAML) plays
seat 0 against each opponent, button alternating, 100bb stacks. Every hand is
scored four ways from the bot's seat: ``raw`` chips, ``chance`` terms only
(:mod:`pokerbot.eval.luck`), chance plus action terms valued against the
opponent's range (``range``) and AIVAT as scored by the play app, chance plus
action terms valued against the opponent's actual hand (``aivat``,
:mod:`pokerbot.eval.aivat`). The AIVAT mean must agree with the raw mean
within its CI; the per-hand standard deviations show the variance reduction.

To score a real play session instead: ``python scripts/score_play_session.py
runs/play/<session>``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

KINDS = ("raw", "chance", "range", "aivat")
TINY = {
    "device": "cpu",
    "time_budget": 0.02,
    "min_iterations": 2,
    "fallback_on_error": False,
    "tree": {"max_nodes": 1500, "chance_cards": 3, "max_raises": 2},
    "solver": {"iterations": 4, "max_runouts": 6},
    "leaf": {"rollouts": 1, "max_total_rollouts": 64},
    "gadget": {"rollouts": 8},
}


def play_chunk(job: tuple) -> dict:
    """Play hands ``[start, stop)`` against ``opponent`` and score them."""
    opponent, start, stop, seed, search_cfg, stacks, samples = job
    import torch

    torch.set_num_threads(1)
    from pokerbot.agents import make_agent
    from pokerbot.config import game_config
    from pokerbot.engine_select import get_engine
    from pokerbot.eval.aivat import action_correction, aivat_adjusted
    from pokerbot.eval.match import play_hand
    from pokerbot.search import SearchAgent, UniformBlueprint

    engine = get_engine()
    config = game_config({"stacks": [stacks, stacks]}, engine)
    bot = SearchAgent(UniformBlueprint(), search_cfg, record=True)
    opp = make_agent(opponent)
    hands, decisions = [], []
    for h in range(start, stop):
        deck = np.random.default_rng((seed, h, 1)).permutation(52).tolist()
        rng = np.random.default_rng((seed, h))
        state = play_hand(config, [bot, opp], h % 2, deck, rng, engine)
        hands.append((h % 2, deck, state))
        decisions.append(bot.decisions)
    chance, aivat = aivat_adjusted(
        hands, decisions, 0, config, engine, preflop_samples=samples, seed=seed * 7919 + start
    )
    rng_terms = [
        action_correction(d, s.hole_cards(0), s.hole_cards(1), values="range")
        for (_, _, s), d in zip(hands, decisions, strict=True)
    ]
    return {
        "raw": [float(s.payoffs()[0]) for _, _, s in hands],
        "chance": chance[:, 0].tolist(),
        "range": (chance[:, 0] + np.array(rng_terms)).tolist(),
        "aivat": aivat[:, 0].tolist(),
        "decisions": sum(len(d) for d in decisions),
        "valued": sum(d["values"] is not None for ds in decisions for d in ds),
    }


def report(opponent: str, res: dict, big_blind: int) -> dict:
    from pokerbot.eval.stats import win_rate

    raw, chance, rng_, aivat = (np.asarray(res[k]) for k in KINDS)
    n = len(raw)
    out = {"opponent": opponent, "hands": n, "decisions": res["decisions"], "valued": res["valued"]}
    for k, x in zip(KINDS, (raw, chance, rng_, aivat), strict=True):
        wr = win_rate(x, big_blind, 1, 0.95, 2000, rng=0)
        out[k] = {
            "mbb": wr.mbb_per_hand,
            "lo": wr.ci_low,
            "hi": wr.ci_high,
            "sd_bb": float(x.std(ddof=1) / big_blind),
        }
    act = aivat - chance
    out["action_sd_bb"] = float(act.std(ddof=1) / big_blind)
    # unbiasedness: the paired difference raw - aivat must be ~0 (z score)
    d = raw - aivat
    out["raw_minus_aivat_z"] = float(d.mean() / (d.std(ddof=1) / np.sqrt(n)))
    out["action_mean_z"] = float(act.mean() / max(act.std(ddof=1) / np.sqrt(n), 1e-12))
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--opponents", nargs="+", default=["equity", "random", "always_call"])
    p.add_argument("--hands", type=int, default=3000)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--chunk", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stacks", type=int, default=10000)
    p.add_argument("--preflop-samples", type=int, default=2048)
    p.add_argument(
        "--iterations",
        type=int,
        default=32,
        help="DCFR iterations per solve of the tiny config (0: its 0.02 s clock, 2-4 iterations)",
    )
    p.add_argument("--search-config", default=None, help="YAML search config (overrides the above)")
    p.add_argument("--out", default=None, help="write the results as JSON here")
    args = p.parse_args(argv)
    search_cfg = args.search_config or dict(TINY)
    if not args.search_config and args.iterations > 0:
        # a fixed iteration count, not the clock, ends each solve: reproducible
        n = args.iterations
        search_cfg.update(
            time_budget=600.0, min_iterations=n, solver={**TINY["solver"], "iterations": n}
        )

    results = []
    with ProcessPoolExecutor(args.workers) as pool:
        for opponent in args.opponents:
            t0 = time.perf_counter()
            jobs = [
                (
                    opponent,
                    s,
                    min(s + args.chunk, args.hands),
                    args.seed,
                    search_cfg,
                    args.stacks,
                    args.preflop_samples,
                )
                for s in range(0, args.hands, args.chunk)
            ]
            merged: dict = {k: [] for k in KINDS} | {"decisions": 0, "valued": 0}
            for r in pool.map(play_chunk, jobs):
                for k in KINDS:
                    merged[k] += r[k]
                merged["decisions"] += r["decisions"]
                merged["valued"] += r["valued"]
            rep = report(opponent, merged, 100)
            rep["seconds"] = time.perf_counter() - t0
            results.append(rep)
            print(
                f"{opponent}: {rep['hands']} hands, {rep['valued']}/{rep['decisions']} "
                f"decisions with values, {rep['seconds']:.0f} s"
            )
            for k in KINDS:
                r = rep[k]
                print(
                    f"  {k:>6}: bot {r['mbb']:+8.0f} mbb/hand [{r['lo']:+.0f}, {r['hi']:+.0f}]"
                    f"  SD {r['sd_bb']:.2f} bb/hand"
                )
            print(
                f"  action terms SD {rep['action_sd_bb']:.3f} bb/hand, mean z "
                f"{rep['action_mean_z']:+.2f}; raw - aivat z {rep['raw_minus_aivat_z']:+.2f}",
                flush=True,
            )
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
