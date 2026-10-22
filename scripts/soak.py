#!/usr/bin/env python
r"""Soak test (``pokerbot.eval.soak``): ``--hands`` hands of the bot against a rotating
mix of opponents, in resumable chunks. Counts crashes, illegal actions and search
fallbacks, and records decision times per street and memory growth; checks Gate 1's
soak bar (0 crashes, 0 illegal actions, fallbacks < 0.1%, p99 decision < 8 s).

The bot is agent ``a`` of a match config (production: configs/match_search_vn.yaml),
built with that config's ``agents`` section; ``--bot`` replaces the spec. Opponents are
any agent specs (``--opponents random equity always_call``, archetypes once merged).
``--search-config`` replaces the search config (e.g. a tiny one for a CPU smoke run)
and ``--device`` the blueprint's and the search's device. Rerunning with the same
``--out-dir`` resumes; ``--summary-only`` rewrites ``soak.json`` / ``soak.md``.
Run it in the main checkout (the blueprint and nets are found from there).

Production, 100,000 hands on the GPU (about 4.5 days at ~4 s a hand):

python scripts/soak.py --config configs/match_search_vn.yaml --hands 100000 \
    --opponents random equity always_call --out-dir runs/soak/prod_100k
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pokerbot.agents import make_agent  # noqa: E402
from pokerbot.config import game_config, load_yaml, run_info  # noqa: E402
from pokerbot.engine_select import get_engine  # noqa: E402
from pokerbot.eval.soak import run_soak, summarize  # noqa: E402


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def build(spec: str, params: dict, device: str | None, search_config: str | None):
    kw = dict(params.get(spec.partition(":")[0]) or {})
    if spec.startswith("search"):
        if search_config:
            kw["config"] = search_config
        if device:
            kw["blueprint"] = {**dict(kw.get("blueprint") or {}), "device": device}
            kw["device"] = device
    elif device and "device" in kw:
        kw["device"] = device
    return make_agent(spec, **kw)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default="configs/match_search_vn.yaml", help="match config")
    ap.add_argument("--bot", help="bot spec (default: the config's match.a)")
    ap.add_argument("--opponents", nargs="+", default=["random", "equity", "always_call"])
    ap.add_argument("--hands", type=int, default=100_000)
    ap.add_argument("--chunk", type=int, default=100, help="hands per chunk file")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--search-config", help="search config replacing the agents section's")
    ap.add_argument("--device", help="device of the bot's blueprint and search, e.g. cpu")
    ap.add_argument("--engine", choices=["auto", "reference", "rust"])
    ap.add_argument("--out-dir", default="runs/soak/run")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_yaml(args.config)
    engine = get_engine(args.engine or cfg.get("engine", "auto"))
    config = game_config(cfg.get("game") or {}, engine)
    if args.summary_only:
        s = summarize(args.out_dir, args.hands, int(config.big_blind))
        print((Path(args.out_dir) / "soak.md").read_text(encoding="utf-8"))
        return 0 if s["gate"]["pass"] else 1
    spec = args.bot or (cfg.get("match") or {}).get("a")
    if not spec:
        ap.error("no bot: give --bot or a config with match.a")
    params = dict(cfg.get("agents") or {})
    bot = build(spec, params, args.device, args.search_config)
    opponents = [build(o, params, None, None) for o in args.opponents]
    info = run_info(args.config, engine)
    log(f"git {info['git']} | bot {spec} | opponents {args.opponents} | {args.hands} hands")
    meta = {"bot": spec, "search_config": args.search_config, "config": args.config}
    s = run_soak(
        bot,
        opponents,
        config,
        args.hands,
        args.out_dir,
        args.chunk,
        args.seed,
        engine,
        labels=args.opponents,
        meta=meta,
        log=log,
    )
    print((Path(args.out_dir) / "soak.md").read_text(encoding="utf-8"))
    return 0 if s["gate"]["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
