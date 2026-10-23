#!/usr/bin/env python
r"""Preflop off-tree audit (``pokerbot.eval.preflop_audit``): the bot's fold / call /
raise frequencies over its range against every open size from a min-raise to all-in,
odd 3-bet and 4-bet sizes after its own on-tree raises, and raises of its limps and
re-raises of its limp-raises, next to the minimum-defence bar. No search runs (preflop
is played from the blueprint or the preflop table), so it runs on the CPU in seconds.

The bot is agent ``a`` of a match config (production: configs/match_search_vn.yaml),
built with that config's ``agents`` section; ``--agent`` replaces the spec and
``--device`` the blueprint's device. Results go to ``--out-dir/preflop_audit.{json,md}``.
Run it in the main checkout: the blueprint finds its strength tables under ``data/``
relative to the working directory.

python scripts/audit_preflop.py --config configs/match_search_vn.yaml --device cpu \
    --out-dir runs/preflop_audit

From a worktree: the main checkout as the working directory, the worktree on
``PYTHONPATH``, and ``--agent search:preflop:<main checkout>/runs/preflop/preflop_r15.pt``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pokerbot.agents import make_agent  # noqa: E402
from pokerbot.config import game_config, load_yaml, run_info  # noqa: E402
from pokerbot.engine_select import get_engine  # noqa: E402
from pokerbot.eval.preflop_audit import audit_markdown, run_audit  # noqa: E402


def build_agent(spec: str, params: dict, device: str | None):
    """``spec`` with its prefix's ``agents`` section; ``device`` moves the
    blueprint (and the search) there."""
    kw = dict(params.get(spec.partition(":")[0]) or {})
    if device and spec.startswith("search"):
        kw["blueprint"] = {**dict(kw.get("blueprint") or {}), "device": device}
        kw["device"] = device
    elif device and "device" in kw:
        kw["device"] = device
    return make_agent(spec, **kw)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default="configs/match_search_vn.yaml", help="match config")
    ap.add_argument("--agent", help="bot spec (default: the config's match.a)")
    ap.add_argument("--device", help="blueprint device, e.g. cpu")
    ap.add_argument("--engine", choices=["auto", "reference", "rust"])
    ap.add_argument("--out-dir", default="runs/preflop_audit")
    args = ap.parse_args(argv)

    cfg = load_yaml(args.config)
    engine = get_engine(args.engine or cfg.get("engine", "auto"))
    spec = args.agent or (cfg.get("match") or {}).get("a")
    if not spec:
        ap.error("no agent: give --agent or a config with match.a")
    agent = build_agent(spec, dict(cfg.get("agents") or {}), args.device)
    config = game_config(cfg.get("game") or {}, engine)
    t0 = time.time()
    rows = run_audit(agent, config, engine=engine)
    dt = time.time() - t0

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    info = {**run_info(args.config, engine), "agent": spec, "seconds": round(dt, 1)}
    (out / "preflop_audit.json").write_text(json.dumps({"info": info, "rows": rows}, indent=1))
    md = (
        f"# Preflop off-tree audit\n\n`{spec}`, git {info['git']}, {len(rows)} spots in "
        f"{dt:.0f} s.\n\nFrequencies over the bot's range at each spot (its strategy "
        "weighted by its reach). `max fold` is the minimum-defence bar "
        "`bet / (pot + bet)`; `source` says whether the preflop table or the base "
        "blueprint plays the spot.\n\n" + audit_markdown(rows, int(config.big_blind))
    )
    (out / "preflop_audit.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"# wrote {out / 'preflop_audit.json'} and .md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
