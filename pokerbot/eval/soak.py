"""Soak test: many hands of the bot against a mix of opponents, for robustness.

Roadmap 1.5 and Gate 1's soak bar: 100,000 hands with 0 crashes, 0 illegal
actions, search fallbacks under 0.1% of decisions and a p99 decision time under
8 s; process and GPU memory must not grow. This is not a strength measurement
(the match runner and the archetype matches are): opponents rotate every two
hands and nothing is duplicated.

* **Schedule.** Hand ``h`` is played by the bot in seat 0 against opponent
  ``(h // 2) % len(opponents)``, with the button on seat ``h % 2``; its deck and
  agent RNG come from ``(seed, h)`` alone. So any hand replays on its own, and a
  run split into chunks plays the same hands as one that is not.
* **Chunks.** Hands go in chunks of ``chunk`` (:func:`run_chunk`); each writes
  ``chunk_NNNNN.json`` atomically when done. :func:`run_soak` skips the chunks on
  disk, so a stopped run resumes where it stopped, losing at most one chunk.
* **Per hand:** an exception anywhere in the hand is caught and recorded (who
  raised it, the traceback) and the next hand starts; illegal actions are
  replaced by fold or check and recorded (``play_hand``'s ``illegal`` list).
* **Per decision of the bot:** the wall time of ``act`` by street (decision
  times include the preflop table lookups), and from ``SearchAgent.stats`` the
  searched decisions and fallbacks. ``stats`` is emptied after every chunk: it
  grows by one entry per searched decision and is not the bot's to keep here.
* **Per chunk:** process RSS and, when CUDA is in use, allocated, reserved and
  peak GPU memory, at its end.

:func:`summarize` reads every chunk file and returns the totals, per-street
p50 / p99 / max decision times, per-opponent win rates, the memory trend
(MB per 10k hands, fitted over the chunks after the first) and the gate.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from ..engine_select import get_engine
from .match import play_hand
from .stats import win_rate

STREETS = ("preflop", "flop", "turn", "river")
GATE = {"crashes": 0, "illegal": 0, "fallback_rate": 0.001, "p99_seconds": 8.0}


class TimedAgent:
    """Wraps the bot: records ``(street, seconds)`` of every ``act`` and whether
    the last exception of a hand came from inside it. Other attributes pass
    through to the wrapped agent."""

    def __init__(self, agent: Any) -> None:
        self.agent = agent
        self.name = getattr(agent, "name", type(agent).__name__)
        self.times: list[tuple[int, float]] = []
        self.raised = False

    def new_hand(self, seat: int, config: Any) -> None:
        self.raised = False
        self.agent.new_hand(seat, config)

    def act(self, state: Any, seat: int, rng: np.random.Generator) -> Any:
        t0 = time.perf_counter()
        try:
            return self.agent.act(state, seat, rng)
        except BaseException:
            self.raised = True
            raise
        finally:
            self.times.append((int(state.street), time.perf_counter() - t0))

    def observe_end(self, state: Any) -> None:
        try:
            self.agent.observe_end(state)
        except BaseException:
            self.raised = True
            raise

    def __getattr__(self, name: str) -> Any:
        return getattr(self.agent, name)


def process_rss_mb() -> float | None:
    """Resident set size of this process in MB (psutil, else the OS directly)."""
    try:
        import psutil

        return psutil.Process().memory_info().rss / 2**20
    except ImportError:
        pass
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        k32 = ctypes.windll.kernel32
        psapi = ctypes.windll.psapi
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        if psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
            return pmc.WorkingSetSize / 2**20
        return None
    try:
        with open("/proc/self/statm") as fh:
            return int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20
    except (OSError, ValueError):
        return None


def gpu_memory_mb() -> dict | None:
    """Allocated, reserved and peak-allocated CUDA memory in MB, or ``None`` when
    CUDA was never initialised in this process."""
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available() or not torch.cuda.is_initialized():
        return None
    return {
        "allocated": torch.cuda.memory_allocated() / 2**20,
        "reserved": torch.cuda.memory_reserved() / 2**20,
        "max_allocated": torch.cuda.max_memory_allocated() / 2**20,
    }


def hand_setup(h: int, seed: int, num_opponents: int) -> tuple[int, int, np.ndarray, Any]:
    """``(opponent index, button, deck, agent rng)`` of hand ``h``."""
    deck = np.random.default_rng((seed, h, 0)).permutation(52)
    return (h // 2) % num_opponents, h % 2, deck, np.random.default_rng((seed, h, 1))


def _short_tb(exc: BaseException, lines: int = 12) -> str:
    tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
    return "".join(tb)[-4000:] if len(tb) <= lines else "".join(tb[-lines:])[-4000:]


def run_chunk(
    bot: Any,
    opponents: Sequence[Any],
    config: Any,
    first: int,
    count: int,
    seed: int = 0,
    engine: ModuleType | None = None,
    labels: Sequence[str] | None = None,
) -> dict:
    """Play hands ``first .. first + count - 1`` (see the module doc) and return the
    chunk record. ``bot`` may be a :class:`TimedAgent` (one is made otherwise)."""
    engine = engine or get_engine()
    timed = bot if isinstance(bot, TimedAgent) else TimedAgent(bot)
    labels = list(labels or [getattr(o, "name", type(o).__name__) for o in opponents])
    stats = getattr(timed.agent, "stats", None)
    n_stats = len(stats) if isinstance(stats, list) else 0
    timed.times = []
    payoffs: list[int | None] = []
    crashes: list[dict] = []
    illegal: list[dict] = []
    t0 = time.time()
    for h in range(first, first + count):
        o, button, deck, rng = hand_setup(h, seed, len(opponents))
        bad: list = []
        try:
            state = play_hand(config, [timed, opponents[o]], button, deck, rng, engine, "fold", bad)
            payoffs.append(int(state.payoffs()[0]))
        except Exception as e:  # a crash: record it and play on
            payoffs.append(None)
            crashes.append(
                {
                    "hand": h,
                    "opponent": labels[o],
                    "agent": "bot" if timed.raised else "opponent or engine",
                    "error": f"{type(e).__name__}: {e}"[:500],
                    "traceback": _short_tb(e),
                }
            )
        for seat, name, action, err in bad:
            illegal.append(
                {
                    "hand": h,
                    "opponent": labels[o],
                    "bot": seat == 0,
                    "agent": name,
                    "action": action,
                    "error": err[:300],
                }
            )
    seconds = time.time() - t0
    times: dict[str, list[float]] = {s: [] for s in STREETS}
    for street, dt in timed.times:
        times[STREETS[min(street, 3)]].append(round(dt, 5))
    fallbacks: dict[str, int] = {s: 0 for s in STREETS}
    searched: dict[str, int] = {s: 0 for s in STREETS}
    if isinstance(stats, list):
        for s in stats[n_stats:]:
            st = STREETS[min(int(s.get("street", 0)), 3)]
            if s.get("fallback"):
                fallbacks[st] += 1
            elif "nodes" in s:
                searched[st] += 1
        del stats[:]  # grows by one entry per searched decision
    return {
        "first_hand": first,
        "hands": count,
        "seconds": seconds,
        "opponents": labels,
        "payoffs": payoffs,
        "crashes": crashes,
        "illegal": illegal,
        "fallbacks": fallbacks,
        "searched": searched,
        "times": times,
        "rss_mb": process_rss_mb(),
        "gpu_mb": gpu_memory_mb(),
    }


def chunk_path(out_dir: str | Path, c: int) -> Path:
    return Path(out_dir) / f"chunk_{c:05d}.json"


def _write_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data))
    os.replace(tmp, path)


def run_soak(
    bot: Any,
    opponents: Sequence[Any],
    config: Any,
    hands: int,
    out_dir: str | Path,
    chunk: int = 100,
    seed: int = 0,
    engine: ModuleType | None = None,
    labels: Sequence[str] | None = None,
    meta: dict | None = None,
    log: Any = None,
) -> dict:
    """Play ``hands`` hands in chunks, writing ``chunk_NNNNN.json`` per chunk and
    ``soak.json`` / ``soak.md`` (:func:`summarize`) after each; chunks already on
    disk are skipped. ``meta`` (e.g. the agent specs) goes to ``soak_meta.json``;
    a resumed run must have the same ``hands``-independent settings."""
    engine = engine or get_engine()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    labels = list(labels or [getattr(o, "name", type(o).__name__) for o in opponents])
    settings = {"seed": seed, "chunk": chunk, "opponents": labels, **(meta or {})}
    meta_path = out / "soak_meta.json"
    if meta_path.exists():
        old = json.loads(meta_path.read_text())
        diff = {k: (old.get(k), v) for k, v in settings.items() if old.get(k) != v}
        if diff:
            raise ValueError(f"{out} holds a soak run with other settings: {diff}")
    else:
        _write_json(meta_path, settings)
    timed = TimedAgent(bot)
    summary = summarize(out, hands)
    for c in range((hands + chunk - 1) // chunk):
        if chunk_path(out, c).exists():
            continue
        first = c * chunk
        rec = run_chunk(
            timed, opponents, config, first, min(chunk, hands - first), seed, engine, labels
        )
        rec["chunk"] = c
        _write_json(chunk_path(out, c), rec)
        summary = summarize(out, hands)
        if log is not None:
            log(_progress(rec, summary))
    return summary


def _progress(rec: dict, s: dict) -> str:
    fb = sum(rec["fallbacks"].values())
    return (
        f"chunk {rec['chunk']}: hands {rec['first_hand']}-{rec['first_hand'] + rec['hands'] - 1} "
        f"in {rec['seconds']:.0f}s, crashes {len(rec['crashes'])}, illegal "
        f"{len(rec['illegal'])}, fallbacks {fb}, rss {rec['rss_mb'] or 0:.0f} MB | "
        f"total {s['hands']} hands, gate {'PASS' if s['gate']['pass'] else 'FAIL'}"
    )


def _percentiles(x: list[float]) -> dict:
    if not x:
        return {"n": 0, "p50": None, "p99": None, "max": None, "mean": None}
    a = np.asarray(x)
    return {
        "n": int(a.size),
        "p50": float(np.percentile(a, 50)),
        "p99": float(np.percentile(a, 99)),
        "max": float(a.max()),
        "mean": float(a.mean()),
    }


def _trend(hands_done: list[int], values: list[float | None]) -> float | None:
    """Least-squares slope in MB per 10,000 hands, over the chunks after the first."""
    pts = [(h, v) for h, v in zip(hands_done, values, strict=True) if v is not None][1:]
    if len(pts) < 2:
        return None
    h, v = np.asarray(pts, dtype=np.float64).T
    if np.ptp(h) == 0:
        return None
    return float(np.polyfit(h, v, 1)[0] * 10_000)


def summarize(out_dir: str | Path, target_hands: int | None = None, big_blind: int = 100) -> dict:
    """Aggregate every chunk file in ``out_dir`` into ``soak.json`` and ``soak.md``
    (both written) and return the summary."""
    out = Path(out_dir)
    recs = sorted(
        (json.loads(p.read_text()) for p in out.glob("chunk_*.json")),
        key=lambda r: r["first_hand"],
    )
    hands = sum(r["hands"] for r in recs)
    crashes = [c for r in recs for c in r["crashes"]]
    illegal = [i for r in recs for i in r["illegal"]]
    times = {s: [t for r in recs for t in r["times"][s]] for s in STREETS}
    all_times = [t for s in STREETS for t in times[s]]
    fallbacks = {s: sum(r["fallbacks"][s] for r in recs) for s in STREETS}
    searched = {s: sum(r["searched"][s] for r in recs) for s in STREETS}
    n_fb = sum(fallbacks.values())
    n_post = sum(len(times[s]) for s in STREETS[1:])
    by_opp: dict[str, list[int]] = {}
    for r in recs:
        labels = r["opponents"]
        for i, p in enumerate(r["payoffs"]):
            if p is not None:
                h = r["first_hand"] + i
                by_opp.setdefault(labels[(h // 2) % len(labels)], []).append(p)
    opponents = {}
    for name, xs in by_opp.items():
        wr = win_rate(xs, big_blind, n_boot=1000)
        opponents[name] = {
            "hands": len(xs),
            "mbb_per_hand": wr.mbb_per_hand,
            "ci": [wr.ci_low, wr.ci_high],
        }
    done = np.cumsum([r["hands"] for r in recs]).tolist()
    rss = [r.get("rss_mb") for r in recs]
    gpu = [r.get("gpu_mb") for r in recs]
    gpu_alloc = [g["allocated"] if g else None for g in gpu]
    gpu_res = [g["reserved"] if g else None for g in gpu]
    bot_crashes = [c for c in crashes if c["agent"] == "bot"]
    bot_illegal = [i for i in illegal if i["bot"]]
    p99 = _percentiles(all_times)["p99"]
    fb_rate = n_fb / n_post if n_post else 0.0
    checks = {
        "crashes": len(crashes) <= GATE["crashes"],
        "illegal": len(bot_illegal) <= GATE["illegal"],
        "fallback_rate": fb_rate < GATE["fallback_rate"],
        "p99_seconds": p99 is None or p99 < GATE["p99_seconds"],
    }
    summary = {
        "hands": hands,
        "target_hands": target_hands,
        "chunks": len(recs),
        "seconds": sum(r["seconds"] for r in recs),
        "crashes": len(crashes),
        "bot_crashes": len(bot_crashes),
        "crash_examples": crashes[:5],
        "illegal": len(illegal),
        "bot_illegal": len(bot_illegal),
        "illegal_examples": illegal[:5],
        "decisions": {s: len(times[s]) for s in STREETS},
        "searched": searched,
        "fallbacks": fallbacks,
        "fallback_rate": fb_rate,
        "times": {s: _percentiles(times[s]) for s in STREETS} | {"all": _percentiles(all_times)},
        "opponents": opponents,
        "memory": {
            "rss_mb_first": next((v for v in rss if v is not None), None),
            "rss_mb_last": next((v for v in reversed(rss) if v is not None), None),
            "rss_mb_max": max((v for v in rss if v is not None), default=None),
            "rss_mb_per_10k_hands": _trend(done, rss),
            "gpu_allocated_mb_last": next((v for v in reversed(gpu_alloc) if v), None),
            "gpu_reserved_mb_max": max((v for v in gpu_res if v), default=None),
            "gpu_allocated_mb_per_10k_hands": _trend(done, gpu_alloc),
        },
        "gate": {
            "checks": checks,
            "pass": all(checks.values()),
            "complete": target_hands is None or hands >= target_hands,
        },
    }
    _write_json(out / "soak.json", summary)
    (out / "soak.md").write_text(soak_markdown(summary), encoding="utf-8")
    return summary


def _f(x: float | None, fmt: str = ".2f") -> str:
    return "-" if x is None else format(x, fmt)


def soak_markdown(s: dict) -> str:
    """The summary as a short markdown report."""
    g = s["gate"]
    target = f" of {s['target_hands']}" if s.get("target_hands") else ""
    lines = [
        "# Soak test\n",
        f"{s['hands']}{target} hands in {s['chunks']} chunks, {s['seconds'] / 3600:.2f} h. "
        f"Gate 1 soak bar: **{'PASS' if g['pass'] else 'FAIL'}**"
        + ("" if g["complete"] else " (incomplete run)")
        + ".\n",
        "| check | value | bar | ok |",
        "|---|---:|---:|---|",
        f"| crashes (bot {s['bot_crashes']}) | {s['crashes']} | 0 | {g['checks']['crashes']} |",
        f"| illegal actions by the bot (all: {s['illegal']}) | {s['bot_illegal']} | 0 "
        f"| {g['checks']['illegal']} |",
        f"| fallbacks / postflop decisions | {100 * s['fallback_rate']:.3f}% | < 0.1% "
        f"| {g['checks']['fallback_rate']} |",
        f"| p99 decision time | {_f(s['times']['all']['p99'])} s | < 8 s "
        f"| {g['checks']['p99_seconds']} |",
        "",
        "| street | decisions | searched | fallbacks | p50 s | p99 s | max s |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for st in STREETS:
        t = s["times"][st]
        lines.append(
            f"| {st} | {t['n']} | {s['searched'][st]} | {s['fallbacks'][st]} | "
            f"{_f(t['p50'], '.3f')} | {_f(t['p99'], '.3f')} | {_f(t['max'], '.3f')} |"
        )
    m = s["memory"]
    lines += [
        "",
        f"Memory: RSS {_f(m['rss_mb_first'], '.0f')} -> {_f(m['rss_mb_last'], '.0f')} MB "
        f"(max {_f(m['rss_mb_max'], '.0f')}, trend {_f(m['rss_mb_per_10k_hands'], '+.1f')} "
        f"MB per 10k hands); GPU allocated {_f(m['gpu_allocated_mb_last'], '.0f')} MB "
        f"(trend {_f(m['gpu_allocated_mb_per_10k_hands'], '+.1f')} MB per 10k hands), "
        f"reserved max {_f(m['gpu_reserved_mb_max'], '.0f')} MB.",
        "",
        "| opponent | hands | bot mbb/hand | 95% CI |",
        "|---|---:|---:|---|",
    ]
    for name, o in s["opponents"].items():
        lines.append(
            f"| {name} | {o['hands']} | {o['mbb_per_hand']:+.0f} | "
            f"[{o['ci'][0]:+.0f}, {o['ci'][1]:+.0f}] |"
        )
    for title, key in (("Crashes", "crash_examples"), ("Illegal actions", "illegal_examples")):
        if s[key]:
            lines += ["", f"## {title} (first {len(s[key])})", ""]
            for e in s[key]:
                lines.append(f"* hand {e['hand']} vs {e['opponent']}: {e.get('error', '')}")
    return "\n".join(lines) + "\n"


__all__ = [
    "GATE",
    "TimedAgent",
    "gpu_memory_mb",
    "hand_setup",
    "process_rss_mb",
    "run_chunk",
    "run_soak",
    "soak_markdown",
    "summarize",
]
