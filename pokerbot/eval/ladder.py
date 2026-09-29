"""Checkpoint ladder: duplicate matches between many agents, pairwise mbb/h
with bootstrap CIs, and ratings.

The ladder is a JSON results file that grows incrementally: every run
schedules pairs (round robin, or each checkpoint against its ``k``
predecessors plus fixed anchors), skips pairs already played with at least
the requested number of hands, plays the rest in parallel worker processes
and rewrites the JSON and a markdown table after every finished pair.

Ratings (see :func:`compute_ratings`):

* ``rating`` - mbb/h against an average field: the weighted least-squares
  solution of ``r_a - r_b = mbb(a vs b)`` with weights ``1 / se^2`` and
  ``sum(r) = 0``. For a transitive set of agents this reproduces every
  pairwise win rate; residuals measure intransitivity.
* ``elo`` - Bradley-Terry fitted to duplicate deals won / lost / tied
  (a deal is won when an agent's net over both seatings is positive),
  on the Elo scale (400 * log10) with the field mean at 1500. It ignores
  win size and is only a secondary, outlier-robust view.

Agents are identified by their spec strings (see
:mod:`pokerbot.agents.registry`), so results are reproducible from the
JSON file alone.
"""

from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import os
import re
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from .logging import RunLogger, write_json_atomic

SCHEDULES = ("all", "previous", "newest")
SEP = " || "


def pair_key(a: str, b: str) -> str:
    x, y = sorted((a, b))
    return f"{x}{SEP}{y}"


def schedule_pairs(
    specs: Sequence[str],
    schedule: str = "all",
    k: int = 5,
    anchors: Sequence[str] = (),
) -> list[tuple[str, str]]:
    """Pairs to play, each as ``(newer, older)`` / ``(agent, anchor)``.

    * ``all``: every pair of ``specs + anchors``;
    * ``previous``: each spec against its ``k`` predecessors and every anchor;
    * ``newest``: only the last spec against its ``k`` predecessors and anchors
      (the per-checkpoint job of DESIGN.md 5.7).
    """
    if schedule not in SCHEDULES:
        raise ValueError(f"schedule must be one of {SCHEDULES}")
    specs = list(dict.fromkeys(specs))
    anchors = [a for a in dict.fromkeys(anchors) if a not in specs]
    out: list[tuple[str, str]] = []
    if schedule == "all":
        pool = specs + anchors
        out = [(pool[j], pool[i]) for i, j in combinations(range(len(pool)), 2)]
    else:
        idx = range(len(specs)) if schedule == "previous" else range(len(specs) - 1, len(specs))
        for i in idx:
            if i < 0:
                continue
            for j in range(max(0, i - k), i):
                out.append((specs[i], specs[j]))
            for a in anchors:
                out.append((specs[i], a))
    seen: set[str] = set()
    uniq = []
    for a, b in out:
        if a == b or pair_key(a, b) in seen:
            continue
        seen.add(pair_key(a, b))
        uniq.append((a, b))
    return uniq


def _worker_init() -> None:
    # one BLAS / torch thread per worker; set before any torch import
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = "1"
    if "torch" in sys.modules:
        sys.modules["torch"].set_num_threads(1)


def play_pair(task: dict[str, Any]) -> dict[str, Any]:
    """Play one duplicate match. Top-level so worker processes can pickle it."""
    from ..agents.registry import make_agent, parse_spec
    from ..config import game_config, git_hash
    from ..engine_select import engine_name, get_engine
    from .match import run_duplicate_match

    t0 = time.time()
    engine = get_engine(task.get("engine") or None)
    config = game_config(task.get("game") or {}, engine)
    params = task.get("agent_params") or {}
    a_spec, b_spec = task["a"], task["b"]
    agents = []
    for spec in (a_spec, b_spec):
        base = parse_spec(spec).name
        ag = make_agent(spec, **(params.get(base) or {}))
        agents.append(ag)
    agents[0].name, agents[1].name = "A", "B"
    deals = max(1, int(task["hands"]) // 2)
    res = run_duplicate_match(
        agents[0],
        agents[1],
        config,
        deals,
        seed=int(task.get("seed", 0)),
        engine=engine,
        n_boot=int(task.get("n_boot", 2000)),
        on_illegal=task.get("on_illegal", "raise"),
    )
    s = res.samples
    return {
        "a": a_spec,
        "b": b_spec,
        "hands": res.hands,
        "deals": int(len(s)),
        "mbb": res.stats.mbb_per_hand,
        "ci_low": res.stats.ci_low,
        "ci_high": res.stats.ci_high,
        "confidence": res.stats.confidence,
        "wins": int((s > 0).sum()),
        "losses": int((s < 0).sum()),
        "ties": int((s == 0).sum()),
        "total_chips": int(s.sum()),
        "seed": int(task.get("seed", 0)),
        "engine": engine_name(engine),
        "git": git_hash(),
        "seconds": round(time.time() - t0, 3),
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def _oriented(m: dict[str, Any], a: str) -> dict[str, Any]:
    """Match record from ``a``'s point of view."""
    if m["a"] == a:
        return m
    flipped = dict(m)
    flipped.update(
        a=m["b"],
        b=m["a"],
        mbb=-m["mbb"],
        ci_low=-m["ci_high"],
        ci_high=-m["ci_low"],
        wins=m["losses"],
        losses=m["wins"],
        total_chips=-m.get("total_chips", 0),
    )
    return flipped


def compute_ratings(
    agents: Sequence[str], matches: Iterable[dict[str, Any]], elo_mean: float = 1500.0
) -> list[dict[str, Any]]:
    """Ratings for ``agents`` from match records (see module docstring)."""
    agents = list(agents)
    idx = {a: i for i, a in enumerate(agents)}
    ms = [m for m in matches if m["a"] in idx and m["b"] in idx]
    n = len(agents)
    rating = np.zeros(n)
    played = np.zeros(n, dtype=int)
    hands = np.zeros(n, dtype=int)
    if ms and n > 1:
        rows, target, weight = [], [], []
        for m in ms:
            r = np.zeros(n)
            r[idx[m["a"]]], r[idx[m["b"]]] = 1.0, -1.0
            se = max((m["ci_high"] - m["ci_low"]) / (2 * 1.96), 1.0)
            rows.append(r)
            target.append(m["mbb"])
            weight.append(1.0 / se)
        # sum-to-zero constraint with a large weight
        rows.append(np.ones(n))
        target.append(0.0)
        weight.append(1e3 * max(weight))
        X = np.array(rows) * np.array(weight)[:, None]
        y = np.array(target) * np.array(weight)
        rating = np.linalg.lstsq(X, y, rcond=None)[0]
    for m in ms:
        for s in (m["a"], m["b"]):
            played[idx[s]] += 1
            hands[idx[s]] += m["hands"]
    # Bradley-Terry by minorization-maximization on deal outcomes
    W = np.full((n, n), 0.0)
    for m in ms:
        i, j = idx[m["a"]], idx[m["b"]]
        W[i, j] += m["wins"] + 0.5 * m["ties"] + 0.5  # +0.5 each way: weak prior
        W[j, i] += m["losses"] + 0.5 * m["ties"] + 0.5
    N = W + W.T
    p = np.ones(n)
    has = N.sum(1) > 0
    for _ in range(2000):
        denom = (N / (p[:, None] + p[None, :])).sum(1)
        new = np.where(has, W.sum(1) / np.maximum(denom, 1e-300), 1.0)
        if has.any():
            new[has] /= np.exp(np.log(new[has]).mean())
        if np.max(np.abs(np.log(new) - np.log(p))) < 1e-10:
            p = new
            break
        p = new
    elo = elo_mean + 400.0 * np.log10(p)
    out = [
        {
            "agent": a,
            "rating": float(rating[i]),
            "elo": float(elo[i]),
            "matches": int(played[i]),
            "hands": int(hands[i]),
        }
        for i, a in enumerate(agents)
    ]
    out.sort(key=lambda r: -r["rating"])
    return out


class Ladder:
    """A ladder results file (JSON) plus its markdown rendering."""

    VERSION = 1

    def __init__(self, path: str | Path, game: dict[str, Any] | None = None) -> None:
        self.path = Path(path)
        self.data: dict[str, Any] = {
            "version": self.VERSION,
            "game": dict(game or {}),
            "agents": [],
            "chain": [],
            "matches": {},
        }
        if self.path.exists():
            with open(self.path) as fh:
                data = json.load(fh)
            if game is not None and data.get("game") and data["game"] != dict(game):
                raise ValueError(
                    f"{self.path} was built for game {data['game']}, not {dict(game)}; "
                    "use a different results file"
                )
            self.data.update(data)

    # ------------------------------------------------------------ bookkeeping
    @property
    def agents(self) -> list[str]:
        return list(self.data["agents"])

    @property
    def matches(self) -> dict[str, dict[str, Any]]:
        return self.data["matches"]

    def add_agents(self, specs: Iterable[str]) -> None:
        for s in specs:
            if s not in self.data["agents"]:
                self.data["agents"].append(s)

    def set_chain(self, specs: Sequence[str]) -> None:
        chain = self.data.setdefault("chain", [])
        for s in specs:
            if s not in chain:
                chain.append(s)

    def get(self, a: str, b: str) -> dict[str, Any] | None:
        m = self.matches.get(pair_key(a, b))
        return _oriented(m, a) if m else None

    def needs(self, a: str, b: str, hands: int) -> bool:
        m = self.matches.get(pair_key(a, b))
        return m is None or int(m["hands"]) < 2 * max(1, int(hands) // 2)

    def record(self, result: dict[str, Any]) -> None:
        self.add_agents([result["a"], result["b"]])
        self.matches[pair_key(result["a"], result["b"])] = result

    # ------------------------------------------------------------ running
    def run(
        self,
        pairs: Sequence[tuple[str, str]],
        hands: int,
        workers: int = 1,
        seed: int = 0,
        engine: str | None = None,
        agent_params: dict[str, Any] | None = None,
        on_illegal: str = "raise",
        n_boot: int = 2000,
        logger: RunLogger | None = None,
        progress: Callable[[str], None] | None = print,
    ) -> list[dict[str, Any]]:
        """Play every pair that is missing (or has fewer hands than ``hands``);
        saves after each pair. Returns the new results."""
        self.add_agents([s for p in pairs for s in p])
        todo = [(a, b) for a, b in pairs if self.needs(a, b, hands)]
        skipped = len(pairs) - len(todo)
        if progress and skipped:
            progress(f"# skipping {skipped} pair(s) already in {self.path}")
        tasks = [
            {
                "a": a,
                "b": b,
                "hands": hands,
                "seed": seed,
                "engine": engine,
                "game": self.data["game"],
                "agent_params": agent_params or {},
                "on_illegal": on_illegal,
                "n_boot": n_boot,
            }
            for a, b in todo
        ]
        done: list[dict[str, Any]] = []

        def finish(res: dict[str, Any]) -> None:
            self.record(res)
            done.append(res)
            self.save()
            if progress:
                hw = (res["ci_high"] - res["ci_low"]) / 2
                progress(
                    f"{res['a']} vs {res['b']}: {res['mbb']:+.1f} mbb/h ± {hw:.1f} "
                    f"({res['hands']} hands, {res['seconds']:.1f}s)"
                )
            if logger is not None:
                logger.scalar(f"ladder/{res['a']} vs {res['b']}", res["mbb"], len(self.matches))

        if workers <= 1 or len(tasks) <= 1:
            for t in tasks:
                finish(play_pair(t))
        else:
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=min(workers, len(tasks)), mp_context=ctx, initializer=_worker_init
            ) as pool:
                futs = [pool.submit(play_pair, t) for t in tasks]
                for f in as_completed(futs):
                    finish(f.result())
        self.save()
        if logger is not None:
            for r in self.ratings():
                logger.scalar(f"rating/{r['agent']}", r["rating"], len(self.matches))
                logger.scalar(f"elo/{r['agent']}", r["elo"], len(self.matches))
            logger.flush()
        return done

    # ------------------------------------------------------------ reporting
    def ratings(self) -> list[dict[str, Any]]:
        return compute_ratings(self.agents, self.matches.values())

    def regressions(self) -> list[dict[str, Any]]:
        """Consecutive chain entries where the newer agent loses to its
        predecessor with a CI that excludes zero (a red flag for the run)."""
        chain = self.data.get("chain") or []
        out = []
        for prev, new in zip(chain, chain[1:], strict=False):
            m = self.get(new, prev)
            if m and m["ci_high"] < 0:
                out.append(m)
        return out

    def save(self) -> None:
        self.data["ratings"] = self.ratings()
        self.data["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        write_json_atomic(self.path, self.data)
        self.markdown_path.write_text(self.to_markdown())

    @property
    def markdown_path(self) -> Path:
        return self.path.with_suffix(".md")

    def to_markdown(self, matrix_limit: int = 12) -> str:
        from ..agents.registry import parse_spec

        def label(s: str) -> str:
            try:
                return parse_spec(s).label
            except ValueError:
                return s

        ratings = self.ratings()
        lines = [
            "# Checkpoint ladder",
            "",
            f"{len(self.agents)} agents, {len(self.matches)} duplicate matches; "
            f"updated {self.data.get('updated', '-')}. Game: `{json.dumps(self.data['game'])}`.",
            "",
            "`rating` is mbb/h against the average of the field (weighted least squares on "
            "pairwise duplicate results). `Elo` is Bradley-Terry on duplicate deals won/lost, "
            "mean 1500.",
            "",
            "| # | agent | rating (mbb/h) | Elo | matches | hands |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for i, r in enumerate(ratings, 1):
            lines.append(
                f"| {i} | `{label(r['agent'])}` | {r['rating']:+.1f} | {r['elo']:.0f} "
                f"| {r['matches']} | {r['hands']} |"
            )
        order = [r["agent"] for r in ratings]
        if 1 < len(order) <= matrix_limit:
            lines += [
                "",
                "## Pairwise (row vs column, mbb/h, 95% CI half-width)",
                "",
                "| | " + " | ".join(f"`{label(a)}`" for a in order) + " |",
                "|---|" + "---:|" * len(order),
            ]
            for a in order:
                cells = []
                for b in order:
                    m = self.get(a, b) if a != b else None
                    if m is None:
                        cells.append("" if a == b else "-")
                    else:
                        hw = (m["ci_high"] - m["ci_low"]) / 2
                        star = "*" if m["ci_low"] > 0 or m["ci_high"] < 0 else ""
                        cells.append(f"{m['mbb']:+.0f} ± {hw:.0f}{star}")
                lines.append(f"| `{label(a)}` | " + " | ".join(cells) + " |")
            lines += ["", "`*`: the confidence interval excludes zero."]
        lines += [
            "",
            "## Matches",
            "",
            "| A | B | mbb/h (A) | 95% CI | hands | deals W/L/T |",
            "|---|---|---:|---|---:|---|",
        ]
        for m in sorted(self.matches.values(), key=lambda m: (m["a"], m["b"])):
            lines.append(
                f"| `{label(m['a'])}` | `{label(m['b'])}` | {m['mbb']:+.1f} "
                f"| [{m['ci_low']:+.1f}, {m['ci_high']:+.1f}] | {m['hands']} "
                f"| {m['wins']}/{m['losses']}/{m['ties']} |"
            )
        regs = self.regressions()
        if regs:
            lines += ["", "## Regressions", ""]
            for m in regs:
                lines.append(
                    f"- `{label(m['a'])}` loses to its predecessor `{label(m['b'])}`: "
                    f"{m['mbb']:+.1f} mbb/h [{m['ci_low']:+.1f}, {m['ci_high']:+.1f}]"
                )
        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- command line


def _natural_key(s: str) -> list[Any]:
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def checkpoint_specs(pattern: str, kind: str, sort: str = "name") -> list[str]:
    paths = glob.glob(pattern)
    if sort == "mtime":
        paths.sort(key=os.path.getmtime)
    else:
        paths.sort(key=_natural_key)
    return [f"{kind}:{p}" for p in paths]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Checkpoint ladder: duplicate matches, pairwise mbb/h and ratings.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument("--agents", nargs="*", default=[], help="agent specs, oldest first")
    ap.add_argument("--checkpoints", help="glob of checkpoint files (added after --agents)")
    ap.add_argument("--kind", default=None, help="agent kind for --checkpoints (blueprint)")
    ap.add_argument("--sort", choices=["name", "mtime"], default="name")
    ap.add_argument("--anchor", action="append", default=[], help="fixed opponent spec")
    ap.add_argument("--schedule", choices=SCHEDULES, default=None)
    ap.add_argument("-k", type=int, default=None, help="predecessors per checkpoint")
    ap.add_argument("--hands", type=int, default=None, help="hands per pair (duplicate)")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--config", help="YAML with game:, agents: and ladder: sections")
    ap.add_argument("--engine", choices=["auto", "reference", "rust"], default=None)
    ap.add_argument("--out", default=None, help="results JSON (markdown written next to it)")
    ap.add_argument("--log-dir", default=None)
    ap.add_argument("--on-illegal", choices=["raise", "fold"], default=None)
    ap.add_argument("--boot", type=int, default=2000)
    return ap


def main(argv: list[str] | None = None) -> int:
    from ..config import GAME_DEFAULTS, load_yaml

    args = build_parser().parse_args(argv)
    cfg = load_yaml(args.config) if args.config else {}
    lcfg = dict(cfg.get("ladder") or {})

    def opt(name: str, default: Any) -> Any:
        v = getattr(args, name)
        return v if v is not None else lcfg.get(name, default)

    specs = list(args.agents or lcfg.get("agents") or [])
    chain: list[str] = []
    pattern = args.checkpoints or lcfg.get("checkpoints")
    if pattern:
        chain = checkpoint_specs(pattern, args.kind or lcfg.get("kind", "blueprint"), args.sort)
        specs += chain
    anchors = list(args.anchor or lcfg.get("anchors") or [])
    schedule = opt("schedule", "all")
    if not chain and schedule != "all":
        chain = list(specs)
    if len(specs) + len(anchors) < 2:
        print("need at least two agents", file=sys.stderr)
        return 2
    game = dict(GAME_DEFAULTS)
    game.update(cfg.get("game") or {})
    out = opt("out", "results/ladder.json")
    ladder = Ladder(out, game)
    ladder.add_agents(specs + anchors)
    ladder.set_chain(chain)
    pairs = schedule_pairs(specs, schedule, int(opt("k", 5)), anchors)
    engine = args.engine or cfg.get("engine", "auto")
    logger = RunLogger(args.log_dir, config={"argv": argv or sys.argv[1:], **cfg})
    print(f"# ladder {out}: {len(pairs)} scheduled pair(s), schedule={schedule}")
    t0 = time.time()
    ladder.run(
        pairs,
        hands=int(opt("hands", 2000)),
        workers=int(opt("workers", 1)),
        seed=int(opt("seed", 0)),
        engine=engine,
        agent_params=cfg.get("agents") or {},
        on_illegal=opt("on_illegal", "raise"),
        n_boot=args.boot,
        logger=logger,
    )
    logger.close()
    print(f"# done in {time.time() - t0:.1f}s; wrote {ladder.path} and {ladder.markdown_path}")
    print(ladder.to_markdown())
    return 0


if __name__ == "__main__":
    sys.exit(main())
