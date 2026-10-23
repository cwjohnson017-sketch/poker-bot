#!/usr/bin/env python
"""Archetype suite: an agent against every scripted archetype, duplicate and
luck-adjusted (roadmap item 1.3, Gate 1).

For each archetype (``pokerbot.agents.archetypes.SUITE`` by default, or
``--archetypes``) the agent plays ``--deals`` duplicate deals (two hands
each) with ``run_duplicate_match(..., luck_adjust=True)``. With
``--reference`` a second agent (e.g. the blueprint next to the production
bot) plays the same deals with the same seeds, and the table adds the paired
difference agent - reference per deal, the Gate 1 regression check.

The table gives the luck-adjusted margin in mbb/hand with a 95% bootstrap CI
per archetype; ``--out`` writes everything as JSON after every chunk of
``--chunk`` deals (per-deal raw and adjusted results, the archetype's measured
HUD statistics and the agent's decisions per street), and ``--resume``
continues a run from that file. Chunk ``k`` of archetype ``i`` uses seed
``seed + 10000 i + k``.

CPU, against the blueprint::

    python scripts/run_archetypes.py --agent neural:runs/dcfr4_distilled_v2,device=cpu \\
        --deals 2000 --out runs/archetypes/blueprint.json

Production next to the blueprint (GPU; the config's ``agents:`` section gives
the search agent its config and the blueprint its device)::

    python scripts/run_archetypes.py --config configs/match_search_vn.yaml \\
        --agent search:preflop:runs/preflop/preflop_r15.pt \\
        --reference neural:runs/dcfr4_distilled_v2 \\
        --deals 10000 --chunk 250 --out runs/archetypes/production.json --resume
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pokerbot.agents import SUITE, Hud, make_agent  # noqa: E402
from pokerbot.config import game_config, load_yaml  # noqa: E402
from pokerbot.engine_select import get_engine  # noqa: E402
from pokerbot.eval.match import run_duplicate_match  # noqa: E402
from pokerbot.eval.stats import win_rate  # noqa: E402

ROLES = ("agent", "reference")


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--agent", required=True, help="agent spec under test")
    ap.add_argument("--reference", help="reference agent spec, played on the same deals")
    ap.add_argument(
        "--archetypes", help=f"comma-separated archetype specs (default: {','.join(SUITE)})"
    )
    ap.add_argument("--deals", type=int, default=1000, help="duplicate deals per archetype")
    ap.add_argument("--chunk", type=int, help="deals per saved chunk (default: --deals)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", help="match YAML: its game: and agents: sections are used")
    ap.add_argument(
        "--search-config",
        help="search YAML replacing agents.search.config (e.g. search_value_net_fixed.yaml)",
    )
    ap.add_argument("--stacks", type=int, default=100, help="stacks in big blinds (no --config)")
    ap.add_argument("--out", help="JSON results file")
    ap.add_argument("--resume", action="store_true", help="skip chunks already in --out")
    ap.add_argument("--no-adjust", action="store_true", help="skip the luck adjustment")
    ap.add_argument("--adjust-samples", type=int, default=2048)
    ap.add_argument("--adjust-device", default="cpu", help="device for the luck adjustment")
    ap.add_argument("--boot", type=int, default=2000, help="bootstrap resamples")
    ap.add_argument(
        "--threads", type=int, help="torch CPU threads (a few is faster on a busy machine)"
    )
    return ap


def build_agent(spec: str, params: dict[str, Any]) -> Any:
    return make_agent(spec, **(params.get(spec.partition(":")[0]) or {}))


def save(path: str, data: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, path)


def results(chunks: list[dict], key: str) -> np.ndarray:
    return np.concatenate([np.asarray(c[key], dtype=np.float64) for c in chunks])


def summarise(data: dict, bb: int, boot: int) -> dict[str, Any]:
    """Win rates per archetype and role, and the paired agent - reference
    difference over the chunks both played."""
    key = "adjusted" if data["luck_adjust"] else "a"
    out: dict[str, Any] = {}
    for name, entry in data["archetypes"].items():
        row: dict[str, Any] = {}
        for role in ROLES:
            chunks = entry.get(role, {}).get("chunks", [])
            if not chunks:
                continue
            hud = Hud()
            for c in chunks:
                if c.get("hud"):
                    hud.merge(c["hud"])
            r = win_rate(results(chunks, key), bb, 2, n_boot=boot)
            raw = win_rate(results(chunks, "a"), bb, 2, n_boot=boot)
            hands = max(1, hud.hands)
            row[role] = {
                "mbb": r.mbb_per_hand,
                "ci": [r.ci_low, r.ci_high],
                "raw_mbb": raw.mbb_per_hand,
                "raw_ci": [raw.ci_low, raw.ci_high],
                "hands": r.hands,
                "hud": hud.summary() if hud.hands else None,
                "decisions_per_hand": [x / hands for x in hud.opp_actions] if hud.hands else None,
                "seconds": sum(c["seconds"] for c in chunks),
            }
        if "agent" in row and "reference" in row:
            ref = {c["k"]: c for c in entry["reference"]["chunks"]}
            both = [(c, ref[c["k"]]) for c in entry["agent"]["chunks"] if c["k"] in ref]
            if both:
                diff = results([a for a, _ in both], key) - results([b for _, b in both], key)
                d = win_rate(diff, bb, 2, n_boot=boot)
                row["diff"] = {"mbb": d.mbb_per_hand, "ci": [d.ci_low, d.ci_high], "hands": d.hands}
        out[name] = row
    return out


def fmt(cell: dict | None) -> str:
    if not cell:
        return "-"
    lo, hi = cell["ci"]
    return f"{cell['mbb']:+7.0f} [{lo:+6.0f}, {hi:+6.0f}]"


def table(summary: dict, data: dict) -> list[str]:
    kind = "luck-adjusted" if data["luck_adjust"] else "raw"
    cols = ["agent"] + (["reference", "agent - reference"] if data.get("reference") else [])
    lines = [f"margin of the agent, {kind} mbb/hand, 95% CI", ""]
    head = f"| {'archetype':18s} | {'hands':>6s} | " + " | ".join(f"{c:24s}" for c in cols) + " |"
    lines += [head, "|" + "|".join("-" * len(p) for p in head.strip("|").split("|")) + "|"]
    for name, row in summary.items():
        cells = [fmt(row.get("agent")), fmt(row.get("reference")), fmt(row.get("diff"))]
        hands = row.get("agent", {}).get("hands", 0)
        lines.append(
            f"| {name:18s} | {hands:6d} | "
            + " | ".join(f"{c:24s}" for c in cells[: len(cols)])
            + " |"
        )
    lines += ["", "archetype HUD measured against the agent (vpip/pfr/3bet/cbet/fold-cbet/agg):"]
    for name, row in summary.items():
        hud = (row.get("agent") or {}).get("hud")
        if hud:
            vals = [hud[k] for k in ("vpip", "pfr", "three_bet", "cbet", "fold_to_cbet", "agg")]
            lines.append(f"  {name:18s} " + " / ".join(f"{v:.2f}" for v in vals))
    return lines


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.threads:
        import torch

        torch.set_num_threads(args.threads)
    cfg = load_yaml(args.config) if args.config else {}
    params = dict(cfg.get("agents") or {})
    if args.search_config:
        params["search"] = {**(params.get("search") or {}), "config": args.search_config}
    engine = get_engine(cfg.get("engine", "auto"))
    game = cfg.get("game") or {"stacks": args.stacks * 100, "small_blind": 50, "big_blind": 100}
    config = game_config(game, engine)
    names = args.archetypes.split(",") if args.archetypes else list(SUITE)
    chunk = args.chunk or args.deals
    n_chunks = -(-args.deals // chunk)
    adjust = not args.no_adjust

    data: dict[str, Any] = {
        "agent": args.agent,
        "reference": args.reference,
        "config": args.config,
        "search_config": args.search_config,
        "game": game,
        "deals": args.deals,
        "chunk": chunk,
        "seed": args.seed,
        "luck_adjust": adjust,
        "archetypes": {},
    }
    if args.resume and args.out and Path(args.out).exists():
        old = json.loads(Path(args.out).read_text())
        for k in ("agent", "reference", "seed", "chunk", "luck_adjust"):
            if old.get(k) != data[k]:
                raise SystemExit(f"--resume: {k} {old.get(k)!r} in {args.out} != {data[k]!r}")
        data["archetypes"] = old["archetypes"]

    specs = {"agent": args.agent, "reference": args.reference}
    players = {role: build_agent(s, params) for role, s in specs.items() if s}
    for i, name in enumerate(names):
        entry = data["archetypes"].setdefault(name, {})
        for role, player in players.items():
            done = {c["k"] for c in entry.setdefault(role, {"chunks": []})["chunks"]}
            for k in range(n_chunks):
                if k in done:
                    continue
                deals = min(chunk, args.deals - k * chunk)
                seed = args.seed + 10_000 * i + k
                opp = make_agent(name)
                t0 = time.time()
                res = run_duplicate_match(
                    player,
                    opp,
                    config,
                    deals,
                    seed,
                    engine,
                    n_boot=args.boot,
                    luck_adjust=adjust,
                    adjust_samples=args.adjust_samples,
                    adjust_device=args.adjust_device,
                )
                hud = getattr(opp, "hud", None)
                entry[role]["chunks"].append(
                    {
                        "k": k,
                        "seed": seed,
                        "deals": deals,
                        "a": res.samples.tolist(),
                        "adjusted": None if res.adjusted is None else res.adjusted.tolist(),
                        "hud": hud.counts() if hud is not None else None,
                        "seconds": time.time() - t0,
                    }
                )
                stats = res.adjusted_stats or res.stats
                log(f"{name} {role} chunk {k + 1}/{n_chunks}: {stats} ({time.time() - t0:.0f}s)")
                if args.out:
                    save(args.out, data)
    summary = summarise(data, config.big_blind, args.boot)
    data["summary"] = summary
    if args.out:
        save(args.out, data)
    print()
    print(f"agent {args.agent}" + (f" | reference {args.reference}" if args.reference else ""))
    print("\n".join(table(summary, data)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
