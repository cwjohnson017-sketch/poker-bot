"""``play_match`` command line: run a (duplicate) match between two agents.

Example::

    python scripts/play_match.py --a always_call --b equity --hands 2000 --duplicate --seed 0
    python scripts/play_match.py --config configs/match.yaml --hands 500
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import time
from typing import Any

from ..agents import AGENTS, make_agent
from ..config import game_config, load_yaml, run_info
from ..engine_select import get_engine
from .match import run_duplicate_match, run_match


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", help="YAML match config (see configs/match.yaml)")
    ap.add_argument("--a", help=f"agent A ({', '.join(sorted(AGENTS))})")
    ap.add_argument("--b", help="agent B")
    ap.add_argument("--hands", type=int, help="number of hands (duplicate: total hands)")
    ap.add_argument(
        "--duplicate",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="duplicate match (each deal played from both seats)",
    )
    ap.add_argument("--seed", type=int)
    ap.add_argument("--engine", choices=["auto", "reference", "rust"])
    ap.add_argument("--history", help="write hand histories to this file")
    ap.add_argument("--boot", type=int, default=2000, help="bootstrap resamples")
    return ap


def resolve(args: argparse.Namespace) -> dict[str, Any]:
    cfg: dict[str, Any] = load_yaml(args.config) if args.config else {}
    match = dict(cfg.get("match") or {})
    agents_cfg = dict(cfg.get("agents") or {})
    out = {
        "a": args.a or match.get("a", "always_call"),
        "b": args.b or match.get("b", "equity"),
        "hands": args.hands if args.hands is not None else int(match.get("hands", 2000)),
        "duplicate": (
            args.duplicate if args.duplicate is not None else bool(match.get("duplicate", False))
        ),
        "seed": args.seed if args.seed is not None else int(match.get("seed", 0)),
        "engine": args.engine or cfg.get("engine", "auto"),
        "history": args.history or match.get("history"),
        "game": cfg.get("game") or {},
        "agent_params": agents_cfg,
    }
    return out


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    opts = resolve(args)
    engine = get_engine(opts["engine"])
    config = game_config(opts["game"], engine)
    params = opts["agent_params"]
    a = make_agent(opts["a"], **(params.get(opts["a"]) or {}))
    b = make_agent(opts["b"], **(params.get(opts["b"]) or {}))
    if a.name == b.name:
        a.name, b.name = a.name + "_A", b.name + "_B"
    info = run_info(args.config, engine)
    print(f"# git {info['git']} | engine {info['engine']} | config {info['config']}")
    t0 = time.time()
    with contextlib.ExitStack() as stack:
        hist = stack.enter_context(open(opts["history"], "w")) if opts["history"] else None
        if opts["duplicate"]:
            deals = max(1, opts["hands"] // 2)
            res = run_duplicate_match(
                a, b, config, deals, opts["seed"], engine, hist, n_boot=args.boot
            )
        else:
            res = run_match(
                [a, b], config, opts["hands"], opts["seed"], engine, hist, n_boot=args.boot
            )
    dt = time.time() - t0
    print(res.summary())
    print(f"# {res.hands} hands in {dt:.1f}s; totals A {res.total_a:+d} B {res.total_b:+d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
