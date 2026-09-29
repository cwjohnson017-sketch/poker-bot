"""Training driver for the tabular MCCFR blueprint.

    python scripts/train_mccfr.py --config configs/mccfr_small.yaml
    python scripts/train_mccfr.py --config configs/mccfr_hunl.yaml --threads 16
    python scripts/train_mccfr.py --config configs/mccfr_hunl.yaml \
        --resume runs/mccfr_hunl/checkpoint.bin

The solver itself is ``poker_engine.Trainer`` (Rust). This module turns a YAML
run config into the trainer's config dict, runs it in chunks with progress
logs, writes checkpoints (periodically from Rust, and at the end) and exports
the averaged strategy for ``BlueprintAgent``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ...config import REPO_ROOT, git_hash, load_yaml
from .interfaces import STREETS, normalize_action_spec

GAME_KEYS = ("num_players", "stacks", "small_blind", "big_blind", "ante")
SOLVER_KEYS = (
    "lcfr_discount_every",
    "lcfr_stop",
    "prune_start",
    "prune_prob",
    "prune_threshold",
    "regret_floor",
    "shards",
)


def _game_section(cfg: dict[str, Any]) -> dict[str, Any]:
    game = {"num_players": 2, "stacks": [20000, 20000], "small_blind": 50, "big_blind": 100}
    game["ante"] = 0
    game.update(cfg.get("game") or {})
    n = int(game["num_players"])
    stacks = game["stacks"]
    game["stacks"] = [int(stacks)] * n if isinstance(stacks, int) else [int(s) for s in stacks]
    return {k: game[k] for k in GAME_KEYS}


def _resolve(path: str | None) -> str | None:
    if path is None:
        return None
    p = Path(path)
    return str(p if p.is_absolute() else REPO_ROOT / p)


def solver_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """The ``poker_engine.Trainer`` config dict for a YAML run config."""
    m = dict(cfg.get("mccfr") or {})
    actions = dict(m.get("actions") or {})
    max_raises = int(actions.pop("max_raises", actions.pop("max_raises_per_street", 4)))
    streets = normalize_action_spec({k: v for k, v in actions.items() if k in STREETS})
    unknown = set(actions) - set(STREETS)
    if unknown:
        raise ValueError(f"unknown mccfr.actions keys: {sorted(unknown)}")
    cards = dict(m.get("cards") or {})
    buckets = cards.get("buckets", [169, 1000, 1000, 1000])
    if isinstance(buckets, dict):
        buckets = [int(buckets.get(s, 169 if s == "preflop" else 1000)) for s in STREETS]
    tables = cards.get("tables") or [None, None, None, None]
    if isinstance(tables, dict):
        tables = [tables.get(s) for s in STREETS]
    out: dict[str, Any] = {
        "game": _game_section(cfg),
        "actions": {"streets": streets, "max_raises": max_raises},
        "cards": {
            "buckets": [int(b) for b in buckets],
            "hs_samples": int(cards.get("hs_samples", 256)),
            "tables": [_resolve(t) for t in tables],
        },
        "seed": int(m.get("seed", cfg.get("seed", 0))),
    }
    for k in SOLVER_KEYS:
        if k in m:
            out[k] = m[k]
    ck = m.get("checkpoint") or {}
    if ck.get("path"):
        out["checkpoint_path"] = _resolve(ck["path"])
        out["checkpoint_interval"] = float(ck.get("interval_seconds", 3600))
    return out


def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def train(
    cfg: dict[str, Any],
    config_path: str | Path | None = None,
    iterations: int | None = None,
    threads: int | None = None,
    resume: str | Path | None = None,
    log: Callable[[str], None] = print,
) -> tuple[Any, dict[str, Any]]:
    """Train (or resume) per ``cfg``; returns ``(trainer, summary)``.

    ``iterations`` is the total target (a resumed run continues up to it).
    Writes the final checkpoint and the exported strategy to the paths in
    ``mccfr.checkpoint.path`` and ``mccfr.output.strategy`` when set."""
    import poker_engine as pe

    m = cfg.get("mccfr") or {}
    target = int(iterations if iterations is not None else m.get("iterations", 100_000))
    threads = int(threads if threads is not None else m.get("threads", 1))
    log_every = float(m.get("log_interval_seconds", 10))
    sc = solver_config(cfg)
    meta = {
        "git": git_hash(),
        "config_path": str(config_path) if config_path else None,
        "config": cfg,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if resume:
        trainer = pe.Trainer.load(str(resume))
        trainer.set_checkpoint(sc.get("checkpoint_path"), sc.get("checkpoint_interval", 0.0))
        log(f"# resumed {resume} at iteration {trainer.iterations}")
    else:
        trainer = pe.Trainer(sc)
    trainer.meta = json.dumps(meta, default=str)
    log(f"# git {meta['git']} | config {config_path} | threads {threads} | target {target} it")
    t0 = time.time()
    chunk = 1000
    interrupted = False
    try:
        while trainer.iterations < target:
            n = min(chunk, target - trainer.iterations)
            st = trainer.run(n, threads)
            rate = max(st["iterations_per_second"], 1.0)
            chunk = max(1000, int(rate * log_every))
            log(
                f"it {st['iterations']:>11,d} | {st['iterations_per_second']:>9,.0f} it/s "
                f"| {st['nodes_per_second']:>11,.0f} nodes/s | infosets {st['infosets']:>11,d} "
                f"| tables {_fmt_bytes(st['table_bytes'])} | rss {_fmt_bytes(st['rss_bytes'])} "
                f"| {time.time() - t0:,.0f}s"
            )
    except KeyboardInterrupt:
        # Keep the work done so far: fall through to checkpoint + export.
        interrupted = True
        log(f"# interrupted at iteration {trainer.iterations}; saving")
    summary = dict(trainer.stats(detailed=True))
    summary["wall_seconds"] = time.time() - t0
    ck = sc.get("checkpoint_path")
    if ck:
        Path(ck).parent.mkdir(parents=True, exist_ok=True)
        trainer.save(ck)
        log(f"# checkpoint {ck}")
    out = (m.get("output") or {}).get("strategy")
    if out:
        out = _resolve(out)
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        n = trainer.export_strategy(out)
        summary["strategy_path"] = out
        summary["strategy_infosets"] = n
        log(f"# strategy {out} ({n:,d} infosets)")
    log(
        f"# done: {summary['iterations']:,d} iterations, {summary['infosets']:,d} infosets "
        f"(per street {summary['infosets_per_street']}), tables "
        f"{_fmt_bytes(summary['table_bytes'])}, rss {_fmt_bytes(summary['rss_bytes'])}"
    )
    if interrupted:
        raise KeyboardInterrupt
    return trainer, summary


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", required=True, help="YAML run config (configs/mccfr_*.yaml)")
    ap.add_argument("--iterations", type=int, help="total iterations (overrides the config)")
    ap.add_argument("--threads", type=int, help="traversal threads (overrides the config)")
    ap.add_argument("--resume", help="checkpoint to continue from")
    ap.add_argument("--out-dir", help="write checkpoint.bin and strategy.bin here instead")
    ap.add_argument("--seed", type=int, help="override the seed")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_yaml(args.config)
    m = cfg.setdefault("mccfr", {})
    if args.seed is not None:
        m["seed"] = args.seed
    if args.out_dir:
        out = Path(args.out_dir)
        m.setdefault("checkpoint", {})["path"] = str(out.resolve() / "checkpoint.bin")
        m.setdefault("output", {})["strategy"] = str(out.resolve() / "strategy.bin")
    try:
        train(cfg, args.config, args.iterations, args.threads, args.resume)
    except KeyboardInterrupt:
        print("# interrupted", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
