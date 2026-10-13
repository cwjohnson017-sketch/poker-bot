#!/usr/bin/env python
"""Head-to-head duplicate match of a search agent against a blueprint, in chunks.

Plays ``--deals`` duplicate deals (2 hands each) of ``--a`` (a ``search:...``
agent) against ``--b`` in chunks of ``--chunk`` deals. After every chunk the
per-deal results (raw and luck-adjusted) and the search agent's per-decision
stats are appended to ``--out`` (JSON), so a detached run can be followed and
resumed (``--resume`` skips the chunks already there; chunk ``k`` uses seed
``seed + k``). The summary has the win rate of A with a bootstrap CI and,
per street, how many decisions searched, fell back to the blueprint, and took
how long.

python scripts/match_search.py --config configs/match_search_vn.yaml \
    --a search:neural:runs/dcfr4_distilled_v2 --b neural:runs/dcfr4_distilled_v2 \
    --deals 1000 --chunk 50 --out runs/vn_match/match.json

Search against search: both agents ``search:...``, each with its own config, e.g.
``--b search:neural:runs/dcfr4_distilled_v2 --b-search-config
configs/search_value_net_r3.yaml``; B's per-decision stats go to ``b_search_stats``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pokerbot.agents import make_agent  # noqa: E402
from pokerbot.config import load_yaml  # noqa: E402
from pokerbot.engine_select import get_engine  # noqa: E402
from pokerbot.eval.match import run_duplicate_match  # noqa: E402
from pokerbot.eval.stats import win_rate  # noqa: E402

STREETS = {1: "flop", 2: "turn", 3: "river"}


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def summarise(data: dict, bb: int) -> list[str]:
    raw = np.concatenate([np.asarray(c["a"], dtype=np.float64) for c in data["chunks"]])
    adj = [c.get("adjusted") for c in data["chunks"]]
    lines = [f"{len(raw)} deals ({2 * len(raw)} hands)"]
    lines.append(f"  raw: {win_rate(raw, bb, 2)}")
    if all(a is not None for a in adj):
        a = np.concatenate([np.asarray(x, dtype=np.float64) for x in adj])
        lines.append(f"  luck-adjusted: {win_rate(a, bb, 2)}")
    for key, who in (("search_stats", ""), ("b_search_stats", "B ")):
        stats = [s for c in data["chunks"] for s in c.get(key, [])]
        for st, name in STREETS.items():
            xs = [s for s in stats if s.get("street") == st]
            if not xs:
                continue
            fb = sum(1 for s in xs if s.get("fallback"))
            t = np.array([s["total_seconds"] for s in xs if "total_seconds" in s])
            it = np.array([s["iterations"] for s in xs if "iterations" in s])
            lines.append(
                f"  {who}{name}: {len(xs)} decisions, {fb} fallbacks, "
                f"time mean {t.mean() if len(t) else float('nan'):.2f}s "
                f"p90 {np.quantile(t, 0.9) if len(t) else float('nan'):.2f}s "
                f"max {t.max() if len(t) else float('nan'):.2f}s, "
                f"iterations mean {it.mean() if len(it) else float('nan'):.0f}"
            )
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True, help="match YAML (game + agents sections)")
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--deals", type=int, default=500)
    ap.add_argument("--chunk", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-adjust", action="store_true")
    ap.add_argument("--out", default="runs/vn_match/match.json")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument(
        "--a-search-config",
        help="search config YAML of --a (default: the match YAML's agents.search.config)",
    )
    ap.add_argument("--b-search-config", help="search config YAML of --b (search against search)")
    args = ap.parse_args(argv)

    cfg = load_yaml(args.config)
    params = cfg.get("agents") or {}
    engine = get_engine()
    g = cfg.get("game") or {}
    config = engine.GameConfig(
        num_players=2,
        stacks=[int(s) for s in g.get("stacks", [10000, 10000])],
        small_blind=int(g.get("small_blind", 50)),
        big_blind=int(g.get("big_blind", 100)),
        ante=int(g.get("ante", 0)),
    )

    def agent_params(spec: str, search_config: str | None) -> dict:
        p = dict(params.get(spec.partition(":")[0]) or {})
        if search_config:
            p["config"] = search_config
        return p

    a = make_agent(args.a, **agent_params(args.a, args.a_search_config))
    b = make_agent(args.b, **agent_params(args.b, args.b_search_config))
    if a.name == b.name:
        a.name, b.name = a.name + "_A", b.name + "_B"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "a": args.a,
        "b": args.b,
        "config": args.config,
        "a_search_config": args.a_search_config,
        "b_search_config": args.b_search_config,
        "seed": args.seed,
        "chunks": [],
    }
    if args.resume and out.exists():
        data = json.loads(out.read_text())
    done = len(data["chunks"])
    n_chunks = (args.deals + args.chunk - 1) // args.chunk
    log(f"# match {args.a} vs {args.b}: {args.deals} deals in chunks of {args.chunk}, {done} done")
    for k in range(done, n_chunks):
        deals = min(args.chunk, args.deals - k * args.chunk)
        t0 = time.time()
        n0 = len(getattr(a, "stats", []))
        nb0 = len(getattr(b, "stats", []))
        res = run_duplicate_match(
            a, b, config, deals, seed=args.seed + k, engine=engine, luck_adjust=not args.no_adjust
        )

        def new_stats(agent, start: int) -> list[dict]:
            return [
                {key: v for key, v in s.items() if isinstance(v, int | float | bool | str | None)}
                for s in getattr(agent, "stats", [])[start:]
            ]

        chunk = {
            "seed": args.seed + k,
            "a": res.samples.tolist(),
            "adjusted": None if res.adjusted is None else res.adjusted.tolist(),
            "search_stats": new_stats(a, n0),
            "seconds": time.time() - t0,
        }
        if hasattr(b, "stats"):  # search against search: B's decisions too
            chunk["b_search_stats"] = new_stats(b, nb0)
        data["chunks"].append(chunk)
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps(data))
        tmp.replace(out)
        log(f"chunk {k + 1}/{n_chunks}: {res.summary()} ({time.time() - t0:.0f}s)")
        for line in summarise(data, config.big_blind):
            log(f"#   {line}")
    log("# done")
    for line in summarise(data, config.big_blind):
        print(line, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
