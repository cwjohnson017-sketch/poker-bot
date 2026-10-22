#!/usr/bin/env python
r"""Search speed on this machine, on the same deals every time so machines compare.

Plays ``--hands`` hands (deals fixed by ``--seed``, the button alternating) of the
bot (agent ``a`` of ``--config``) against ``--opponent`` (default ``always_call``:
the bot then searches every street) and reports per street: decisions, DCFR
iterations and seconds per decision, iterations per second of solving, and the
seconds per hand.

Two runs per machine answer the cloud questions (``cloud/aws/README.md``,
"Calibration"):

* production budgets (``configs/search_value_net.yaml``): how many iterations
  this GPU fits in the 4 / 2 / 1 s budgets;
* fixed iterations (``configs/search_value_net_fixed.yaml``): how long this GPU
  takes for production's iterations, so its cost per hand.

python scripts/calibrate_search.py --search-config configs/search_value_net_fixed.yaml \
    --hands 24 --out runs/calibration/local_fixed.json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pokerbot.agents import AlwaysCallAgent, make_agent  # noqa: E402
from pokerbot.config import game_config, git_hash, load_yaml  # noqa: E402
from pokerbot.engine_select import get_engine  # noqa: E402
from pokerbot.eval.match import play_hand  # noqa: E402

STREETS = {1: "flop", 2: "turn", 3: "river"}


def street_summary(stats: list[dict]) -> dict:
    """Per-street numbers of the searched decisions in ``stats`` (``SearchAgent.stats``)."""
    out = {}
    for street, name in STREETS.items():
        rows = [s for s in stats if s.get("street") == street and "nodes" in s]
        if not rows:
            continue
        it = np.array([s["iterations"] for s in rows], dtype=float)
        sec = np.array([s["total_seconds"] for s in rows], dtype=float)
        solve = np.array([s["solve_seconds"] for s in rows], dtype=float)
        out[name] = {
            "decisions": len(rows),
            "iterations_mean": float(it.mean()),
            "seconds_mean": float(sec.mean()),
            "seconds_p50": float(np.percentile(sec, 50)),
            "seconds_p90": float(np.percentile(sec, 90)),
            "iterations_per_solve_second": float(it.sum() / max(solve.sum(), 1e-9)),
            "nodes_mean": float(np.mean([s["nodes"] for s in rows])),
        }
    return out


def machine() -> dict:
    info = {"host": platform.node(), "gpu": "cpu"}
    try:
        import torch

        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["torch"] = torch.__version__
    except ImportError:  # pragma: no cover
        pass
    return info


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default="configs/match_search_vn.yaml", help="match config")
    ap.add_argument("--bot", help="bot spec (default: the config's match.a)")
    ap.add_argument("--search-config", help="search config replacing the agents section's")
    ap.add_argument("--opponent", default="always_call", help="opponent spec")
    ap.add_argument("--hands", type=int, default=24)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-warmup", action="store_true", help="skip the unlogged warm-up hand")
    ap.add_argument("--label", default="", help="a name for this run in the report")
    ap.add_argument("--out", help="JSON file for the results")
    args = ap.parse_args(argv)

    cfg = load_yaml(args.config)
    engine = get_engine(cfg.get("engine", "auto"))
    config = game_config(cfg.get("game") or {}, engine)
    spec = args.bot or (cfg.get("match") or {}).get("a")
    if not spec:
        ap.error("no bot: give --bot or a config with match.a")
    params = dict(cfg.get("agents") or {})
    kw = dict(params.get(spec.partition(":")[0]) or {})
    if args.search_config:
        kw["config"] = args.search_config
    bot = make_agent(spec, **kw)
    opponent = make_agent(args.opponent, **(params.get(args.opponent.partition(":")[0]) or {}))

    if not args.no_warmup and hasattr(bot, "stats"):
        # load the nets and compile kernels outside the timed hands
        aces = [51, 50]
        rest = [c for c in np.random.default_rng(1).permutation(52).tolist() if c not in aces]
        t0 = time.perf_counter()
        play_hand(
            config, [bot, AlwaysCallAgent()], 0, aces + rest, np.random.default_rng(1), engine
        )
        print(f"warm-up hand {time.perf_counter() - t0:.1f} s", flush=True)
        bot.stats.clear()

    t0 = time.perf_counter()
    for h in range(args.hands):
        deck = np.random.default_rng((args.seed, h)).permutation(52).tolist()
        rng = np.random.default_rng((args.seed, h, 1))
        play_hand(config, [bot, opponent], h % 2, deck, rng, engine)
    wall = time.perf_counter() - t0
    stats = list(getattr(bot, "stats", []))
    result = {
        "label": args.label,
        "machine": machine(),
        "git": git_hash(short=True),
        "bot": spec,
        "search_config": args.search_config or kw.get("config"),
        "opponent": args.opponent,
        "hands": args.hands,
        "seed": args.seed,
        "wall_seconds": wall,
        "seconds_per_hand": wall / max(args.hands, 1),
        "fallbacks": sum(1 for s in stats if s.get("fallback")),
        "streets": street_summary(stats),
    }
    m = result["machine"]
    print(
        f"{m['gpu']} | {args.hands} hands in {wall:.0f} s, {result['seconds_per_hand']:.2f} s/hand"
    )
    for name, s in result["streets"].items():
        print(
            f"  {name:5s} {s['decisions']:4d} decisions, {s['iterations_mean']:7.1f} iterations, "
            f"{s['seconds_mean']:5.2f} s mean ({s['seconds_p90']:5.2f} p90), "
            f"{s['iterations_per_solve_second']:7.1f} it/s"
        )
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
