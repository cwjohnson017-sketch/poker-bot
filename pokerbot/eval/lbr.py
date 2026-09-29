"""Local Best Response (Lisý & Bowling, 2017), heads-up.

LBR is a cheap lower bound on exploitability. It plays real hands against
the opponent and, at each of its decisions:

1. keeps the opponent's range, a distribution over the 1,326 hole-card
   pairs, updated by Bayes' rule with the opponent's policy after every
   opponent action (``range[h] *= pi(observed action | h)``) and with
   card removal for LBR's own cards and the board;
2. computes ``wp``, its equity against that range, with the batched torch
   evaluator (exact on the river and turn, sampled runouts on the flop);
3. values each candidate action assuming that after it the opponent checks
   or calls to showdown, except that a raise can make them fold:

   * fold:  ``-c_me`` (what LBR has already put in),
   * call:  ``(2 wp - 1) * stake_after_call``,
   * raise: ``fp * c_opp + (1 - fp) * (2 wp - 1) * stake_after_call_of_raise``,
     where ``fp = sum_h range[h] * pi(fold | h, after the raise)``,

   and plays the best one. Chip values are the final net result of the hand
   (a heads-up showdown moves ``min(contributions)``), so the call rule
   equals the paper's ``wp * pot - (1 - wp) * asked``.

Preflop, LBR just calls by default (``preflop="call"``, as in the paper's
experiments); ``"fc"`` lets it fold by the same rule and ``"full"`` also
raises. Candidate raises are the abstract spec's raises (``raises="spec"``)
or pot and all-in only (``raises="fcpa"``).

The opponent must be a :class:`~pokerbot.agents.policy.PolicyAgent`; agents
that only ``act`` are wrapped by sampling (``SampledPolicyAgent``, warns).

Reported: LBR's win rate in mbb/h with a bootstrap CI, the same with
all-in adjustment (when the money went in before the river, the realized
result is replaced by its expectation over the remaining board, which cuts
variance a lot and is unbiased), and a per-street breakdown.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from itertools import combinations
from math import comb
from types import ModuleType
from typing import Any

import numpy as np
import torch

from ..agents.base import BaseAgent
from ..agents.policy import (
    CHECK_CALL,
    FOLD,
    RAISE,
    AbstractChoice,
    abstract_actions,
    as_policy_agent,
    default_spec,
    hypothetical_state,
    nearest_abstract_index,
    range_policy,
)
from ..engine_select import get_engine
from ..env.cards import make_generator
from ..env.equity import sample_unknown
from ..env.evaluator import evaluate_batch
from .logging import RunLogger
from .stats import WinRate, win_rate

STREETS = ("preflop", "flop", "turn", "river")
BOARD_LEN = (0, 3, 4, 5)

COMBOS = np.array(list(combinations(range(52), 2)), dtype=np.int64)  # [1326, 2]
NUM_COMBOS = len(COMBOS)
_COMBOS_T: dict[str, torch.Tensor] = {}


def combos_tensor(device: torch.device | str) -> torch.Tensor:
    key = str(device)
    if key not in _COMBOS_T:
        _COMBOS_T[key] = torch.as_tensor(COMBOS, device=device)
    return _COMBOS_T[key]


def blocked_mask(cards: list[int] | np.ndarray) -> np.ndarray:
    """``[1326]`` bool: combos that share a card with ``cards``."""
    dead = np.zeros(53, dtype=bool)
    dead[np.asarray(list(cards), dtype=np.int64)] = True
    return dead[COMBOS[:, 0]] | dead[COMBOS[:, 1]]


def combo_index(c1: int, c2: int) -> int:
    a, b = sorted((int(c1), int(c2)))
    # index of (a, b) in lexicographic combinations(range(52), 2)
    return a * 51 - a * (a - 1) // 2 + (b - a - 1)


@torch.no_grad()
def range_equity(
    hero: list[int],
    board: list[int],
    weights: np.ndarray,
    max_runouts: int | None = 64,
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
    max_rows: int = 1 << 20,
) -> float:
    """Equity (win + tie / 2) of ``hero`` against a weighted range of opponent
    combos, over the runouts of the rest of the board.

    Runouts are enumerated when there are at most ``max_runouts`` of them
    (always on the river and turn with the default), otherwise
    ``max_runouts`` are sampled. Combos that collide with a runout are
    dropped from that runout, which gives the exact joint distribution of
    (runout, opponent hand)."""
    dev = torch.device(device)
    w_all = np.asarray(weights, dtype=np.float64)
    live = np.nonzero(w_all > 0)[0]
    if live.size == 0:
        raise ValueError("empty range")
    combos = combos_tensor(dev)[torch.as_tensor(live, device=dev)]
    wl = torch.as_tensor(w_all[live], dtype=torch.float64, device=dev)
    known = [int(c) for c in hero] + [int(c) for c in board]
    k = 5 - len(board)
    if k == 0:
        runouts = torch.zeros(1, 0, dtype=torch.long, device=dev)
    else:
        remaining = [c for c in range(52) if c not in set(known)]
        n_exact = comb(len(remaining), k)
        if (max_runouts is None and n_exact <= 200_000) or (
            max_runouts is not None and n_exact <= max_runouts
        ):
            runouts = torch.tensor(list(combinations(remaining, k)), dtype=torch.long, device=dev)
        else:
            R = int(max_runouts or 2000)
            kn = torch.tensor(known, dtype=torch.long, device=dev).expand(R, len(known))
            runouts = sample_unknown(kn, k, generator)
    R = runouts.shape[0]
    b = torch.tensor([int(c) for c in board], dtype=torch.long, device=dev)
    full = torch.cat([b.expand(R, len(board)), runouts], 1)  # [R, 5]
    hero_t = torch.tensor([int(c) for c in hero], dtype=torch.long, device=dev)
    hr = evaluate_batch(torch.cat([hero_t.expand(R, 2), full], 1))  # [R]
    K = combos.shape[0]
    num = torch.zeros((), dtype=torch.float64, device=dev)
    den = torch.zeros((), dtype=torch.float64, device=dev)
    step = max(1, max_rows // max(1, K))
    for s in range(0, R, step):
        e = min(R, s + step)
        f = full[s:e]
        r = e - s
        opp = torch.cat([combos[None].expand(r, K, 2), f[:, None, :].expand(r, K, 5)], 2)
        orank = evaluate_batch(opp)  # [r, K]
        dead = torch.zeros(r, 53, dtype=torch.bool, device=dev)
        if k > 0:
            dead.scatter_(1, runouts[s:e], True)
        bad = dead[:, combos[:, 0]] | dead[:, combos[:, 1]]
        ww = wl[None, :] * (~bad).double()
        h = hr[s:e, None]
        score = (h > orank).double() + 0.5 * (h == orank).double()
        num += (ww * score).sum()
        den += ww.sum()
    return float(num / den)


class LBRAgent(BaseAgent):
    """Local best response against a policy agent (see module docstring).

    ``range`` holds the opponent range (``[1326]``, sums to one) after the
    last call to ``act``; ``decisions`` counts LBR's actions per street."""

    name = "lbr"

    def __init__(
        self,
        opponent: Any,
        spec: Any = None,
        preflop: str = "call",
        raises: str = "spec",
        max_runouts: int = 64,
        likelihood_floor: float = 1e-4,
        device: torch.device | str = "cpu",
        seed: int = 0,
        engine: ModuleType | None = None,
    ) -> None:
        super().__init__()
        if preflop not in ("call", "fc", "full"):
            raise ValueError("preflop must be 'call', 'fc' or 'full'")
        if raises not in ("spec", "fcpa"):
            raise ValueError("raises must be 'spec' or 'fcpa'")
        self.opponent = opponent
        self.spec = spec or getattr(opponent, "spec", None) or default_spec()
        self.preflop = preflop
        self.raises = raises
        self.max_runouts = max_runouts
        self.floor = float(likelihood_floor)
        self.device = torch.device(device)
        self.generator = make_generator(seed, self.device)
        self.engine = engine or get_engine()
        self.range: np.ndarray | None = None
        self.decisions: dict[str, dict[str, int]] = {s: {} for s in STREETS}
        self.range_resets = 0
        self.last_values: dict[str, float] = {}
        self._processed = 0
        self._hole: list[int] = []

    # ---------------------------------------------------------------- range
    def new_hand(self, seat: int, config: Any) -> None:
        super().new_hand(seat, config)
        self.range = None
        self._processed = 0
        self._hole = []

    def _uniform(self, dead: list[int]) -> np.ndarray:
        w = (~blocked_mask(dead)).astype(np.float64)
        return w / w.sum()

    def sync_range(self, state: Any, seat: int) -> np.ndarray:
        """Bring the opponent range up to date with ``state`` (card removal and
        every opponent action not yet processed)."""
        hole = list(state.hole_cards(seat))
        if self.range is None or hole != self._hole:
            self._hole = hole
            self.range = self._uniform(hole)
            self._processed = 0
        opp = 1 - seat
        hist = state.history
        known = {seat: hole}
        config = self.config if self.config is not None else state.config
        for k in range(self._processed, len(hist)):
            _s, p, action = hist[k]
            if p != opp:
                continue
            # card removal first so every queried hand is consistent with the board
            self.range[blocked_mask(state.board)] = 0.0
            live = np.nonzero(self.range > 0)[0]
            if live.size == 0:
                break
            probs = range_policy(
                self.opponent,
                state,
                opp,
                COMBOS[live],
                config,
                known,
                self.engine,
                self.spec,
                upto=k,
            )
            pre = hypothetical_state(state, opp, COMBOS[live[0]], config, known, self.engine, k)
            idx = nearest_abstract_index(pre, action, self.spec)
            self.range[live] *= np.maximum(probs[:, idx], self.floor)
            self.range /= self.range.sum()
        self._processed = len(hist)
        self.range[blocked_mask(hole + list(state.board))] = 0.0
        total = self.range.sum()
        if total <= 0:
            self.range_resets += 1
            self.range = self._uniform(hole + list(state.board))
        else:
            self.range /= total
        return self.range

    # ---------------------------------------------------------------- acting
    def _candidates(self, state: Any, seat: int) -> list[AbstractChoice]:
        choices = abstract_actions(state, self.spec)
        street = state.street
        if street == 0 and self.preflop != "full":
            return [c for c in choices if c.kind != RAISE]
        if self.raises == "spec":
            return choices
        out = [c for c in choices if c.kind != RAISE]
        legal = state.legal_actions()
        if legal.min_raise_to > 0:
            bets = list(state.street_bets)
            to_call = max(bets) - bets[seat]
            pot_to = max(bets) + state.pot + to_call
            hi = legal.max_raise_to
            pot_to = min(max(pot_to, legal.min_raise_to), hi)
            if pot_to < hi:
                out.append(AbstractChoice(-1, RAISE, pot_to, "raise 1"))
            out.append(AbstractChoice(-2, RAISE, hi, "allin"))
        return out

    def _record(self, street: int, label: str) -> None:
        d = self.decisions[STREETS[min(street, 3)]]
        d[label] = d.get(label, 0) + 1

    def act(self, state: Any, seat: int, rng: np.random.Generator) -> Any:
        rng_range = self.sync_range(state, seat)
        street = state.street
        cands = self._candidates(state, seat)
        if street == 0 and self.preflop == "call":
            self._record(street, "check_call")
            return self.check_call()
        config = self.config if self.config is not None else state.config
        opp = 1 - seat
        start = list(config.stacks)
        stacks = list(state.stacks)
        c_me = start[seat] - stacks[seat]
        c_opp = start[opp] - stacks[opp]
        bets = list(state.street_bets)
        wp = range_equity(
            self._hole, list(state.board), rng_range, self.max_runouts, self.generator, self.device
        )
        known = {seat: self._hole}
        live = np.nonzero(rng_range > 0)[0]
        values: dict[int, float] = {}
        base = None  # full state with a placeholder opponent hand, built on demand
        for i, c in enumerate(cands):
            if c.kind == FOLD:
                values[i] = -float(c_me)
            elif c.kind == CHECK_CALL:
                call = min(max(bets) - bets[seat], stacks[seat])
                values[i] = (2 * wp - 1) * min(c_me + call, c_opp)
            else:
                add_me = c.amount - bets[seat]
                add_opp = min(c.amount - bets[opp], stacks[opp])
                stake = min(c_me + add_me, c_opp + add_opp)
                if base is None:
                    base = hypothetical_state(
                        state, opp, COMBOS[live[0]], config, known, self.engine
                    )
                child = base.child(c.action(self.engine))
                fp = 0.0
                if not child.is_terminal and child.current_player == opp:
                    probs = range_policy(
                        self.opponent,
                        child,
                        opp,
                        COMBOS[live],
                        config,
                        known,
                        self.engine,
                        self.spec,
                    )
                    fold_idx = next(
                        j
                        for j, a in enumerate(self.spec.streets[min(child.street, 3)])
                        if a[0] == "fold"
                    )
                    fp = float((rng_range[live] * probs[:, fold_idx]).sum())
                values[i] = fp * c_opp + (1 - fp) * (2 * wp - 1) * stake
        best = max(values, key=lambda i: (values[i], -i))
        choice = cands[best]
        self.last_values = {cands[i].label: v for i, v in values.items()}
        self.last_values["wp"] = wp
        self._record(street, choice.label)
        return choice.action(self._engine)


# ------------------------------------------------------------------ running


def allin_adjusted(state: Any, seat: int, config: Any, generator: torch.Generator | None) -> float:
    """``seat``'s expected result when the money went in before the river
    (expectation over the remaining board), else the realized payoff."""
    pay = float(state.payoffs()[seat])
    if any(state.folded) or not state.history:
        return pay
    last_street = state.history[-1][0]
    if last_street >= 3:
        return pay
    opp = 1 - seat
    board = list(state.board)[: BOARD_LEN[last_street]]
    hero, vill = list(state.hole_cards(seat)), list(state.hole_cards(opp))
    if len(vill) != 2:
        return pay
    w = np.zeros(NUM_COMBOS)
    w[combo_index(*vill)] = 1.0
    eq = range_equity(
        hero, board, w, max_runouts=None if last_street > 0 else 4000, generator=generator
    )
    start = list(config.stacks)
    stake = min(start[p] - state.stacks[p] for p in (0, 1))
    return stake * (2 * eq - 1)


@dataclass
class LBRResult:
    opponent: str
    big_blind: int
    payoffs: np.ndarray
    adjusted: np.ndarray
    end_street: np.ndarray
    actions: dict[str, dict[str, int]]
    seconds: float = 0.0
    range_resets: int = 0
    confidence: float = 0.95
    n_boot: int = 2000
    seed: int = 0
    raw: WinRate = field(init=False)
    adj: WinRate = field(init=False)

    def __post_init__(self) -> None:
        self.raw = win_rate(
            self.payoffs, self.big_blind, 1, self.confidence, self.n_boot, self.seed
        )
        self.adj = win_rate(
            self.adjusted, self.big_blind, 1, self.confidence, self.n_boot, self.seed
        )

    @property
    def hands(self) -> int:
        return int(len(self.payoffs))

    def by_street(self) -> list[dict[str, Any]]:
        n = max(1, self.hands)
        rows = []
        for s, name in enumerate(STREETS):
            m = self.end_street == s
            k = int(m.sum())
            rows.append(
                {
                    "street": name,
                    "hands_ended": k,
                    "share": k / n,
                    "mbb_when_ended": (
                        1000.0 * float(self.adjusted[m].mean()) / self.big_blind if k else 0.0
                    ),
                    "contribution_mbb": 1000.0 * float(self.adjusted[m].sum()) / n / self.big_blind,
                    "lbr_actions": dict(self.actions.get(name, {})),
                }
            )
        return rows

    def summary(self) -> str:
        lines = [
            f"LBR vs {self.opponent}: {self.adj} (all-in adjusted)",
            f"  raw: {self.raw}",
            "  street   ended  share  mbb/h|ended  contrib  LBR actions",
        ]
        for r in self.by_street():
            acts = ", ".join(f"{k} {v}" for k, v in sorted(r["lbr_actions"].items()))
            lines.append(
                f"  {r['street']:<8} {r['hands_ended']:>5}  {r['share']:5.1%}  "
                f"{r['mbb_when_ended']:+10.0f}  {r['contribution_mbb']:+7.0f}  {acts}"
            )
        if self.range_resets:
            lines.append(f"  range reset {self.range_resets}x (observed action had ~0 probability)")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        def wr(w: WinRate) -> dict[str, float]:
            return {"mbb": w.mbb_per_hand, "ci_low": w.ci_low, "ci_high": w.ci_high}

        return {
            "opponent": self.opponent,
            "hands": self.hands,
            "adjusted": wr(self.adj),
            "raw": wr(self.raw),
            "by_street": self.by_street(),
            "range_resets": self.range_resets,
            "seconds": self.seconds,
            "seed": self.seed,
        }


def _play_lbr_hands(
    opponent: Any,
    config: Any,
    hands: int,
    seed: int,
    chunk: int,
    engine: ModuleType,
    lbr_kwargs: dict[str, Any],
    samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, dict[str, int]], int]:
    from .match import play_hand

    opp = as_policy_agent(opponent, samples=samples)
    lbr = LBRAgent(opp, engine=engine, seed=seed * 1_000_003 + chunk, **lbr_kwargs)
    deck_rng = np.random.default_rng((seed, chunk))
    gen = make_generator(seed * 7919 + chunk, "cpu")
    pay = np.zeros(hands)
    adj = np.zeros(hands)
    end = np.zeros(hands, dtype=np.int64)
    for h in range(hands):
        deck = deck_rng.permutation(52)
        rng = np.random.default_rng((seed, chunk, h))
        state = play_hand(config, [lbr, opp], h % 2, deck, rng, engine)
        pay[h] = state.payoffs()[0]
        adj[h] = allin_adjusted(state, 0, config, gen)
        end[h] = state.history[-1][0] if state.history else 0
    return pay, adj, end, lbr.decisions, lbr.range_resets


def _lbr_chunk(task: dict[str, Any]) -> tuple:
    from ..agents.registry import make_agent, parse_spec
    from ..config import game_config

    torch.set_num_threads(1)
    engine = get_engine(task.get("engine") or None)
    config = game_config(task.get("game") or {}, engine)
    spec = task["opponent"]
    params = (task.get("agent_params") or {}).get(parse_spec(spec).name) or {}
    opponent = make_agent(spec, **params)
    return _play_lbr_hands(
        opponent,
        config,
        task["hands"],
        task["seed"],
        task["chunk"],
        engine,
        task["lbr_kwargs"],
        task["samples"],
    )


def run_lbr(
    opponent: Any,
    config: Any = None,
    hands: int = 1000,
    seed: int = 0,
    engine: ModuleType | str | None = None,
    workers: int = 1,
    samples: int = 16,
    logger: RunLogger | None = None,
    game: dict[str, Any] | None = None,
    agent_params: dict[str, Any] | None = None,
    confidence: float = 0.95,
    n_boot: int = 2000,
    **lbr_kwargs: Any,
) -> LBRResult:
    """Play ``hands`` hands of LBR (seat 0, button alternating) against
    ``opponent`` (an agent object, or a registry spec string).

    ``workers > 1`` needs a spec string: hands are split into chunks played
    in separate processes, each with its own opponent instance."""
    from ..agents.registry import make_agent, parse_spec
    from ..config import game_config

    t0 = time.time()
    eng = engine if isinstance(engine, ModuleType) else get_engine(engine)
    if config is None:
        config = game_config(game or {}, eng)
    if isinstance(opponent, str):
        name = opponent
    else:
        name = str(getattr(opponent, "name", type(opponent).__name__))
    if workers > 1 and isinstance(opponent, str) and hands >= 2 * workers:
        sizes = [hands // workers + (1 if i < hands % workers else 0) for i in range(workers)]
        tasks = [
            {
                "opponent": opponent,
                "hands": n,
                "seed": seed,
                "chunk": i,
                "engine": engine if isinstance(engine, str) else None,
                "game": game or {},
                "agent_params": agent_params or {},
                "lbr_kwargs": lbr_kwargs,
                "samples": samples,
            }
            for i, n in enumerate(sizes)
        ]
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
            parts = list(pool.map(_lbr_chunk, tasks))
    else:
        if isinstance(opponent, str):
            params = (agent_params or {}).get(parse_spec(opponent).name) or {}
            opponent = make_agent(opponent, **params)
        parts = [_play_lbr_hands(opponent, config, hands, seed, 0, eng, lbr_kwargs, samples)]
    pay = np.concatenate([p[0] for p in parts])
    adj = np.concatenate([p[1] for p in parts])
    end = np.concatenate([p[2] for p in parts])
    actions: dict[str, dict[str, int]] = {s: {} for s in STREETS}
    for p in parts:
        for s, d in p[3].items():
            for k, v in d.items():
                actions[s][k] = actions[s].get(k, 0) + v
    res = LBRResult(
        opponent=name,
        big_blind=int(config.big_blind),
        payoffs=pay,
        adjusted=adj,
        end_street=end,
        actions=actions,
        seconds=time.time() - t0,
        range_resets=sum(p[4] for p in parts),
        confidence=confidence,
        n_boot=n_boot,
        seed=seed,
    )
    if logger is not None:
        logger.scalars(
            {
                "lbr/mbb": res.adj.mbb_per_hand,
                "lbr/ci_low": res.adj.ci_low,
                "lbr/ci_high": res.adj.ci_high,
                "lbr/raw_mbb": res.raw.mbb_per_hand,
            },
            res.hands,
        )
        for r in res.by_street():
            logger.scalar(f"lbr/contribution/{r['street']}", r["contribution_mbb"], res.hands)
        logger.write_json("lbr.json", res.to_dict())
    return res


# ------------------------------------------------------------------ command line


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Local best response against an agent.")
    ap.add_argument("--opponent", help="agent spec (see pokerbot.agents.registry)")
    ap.add_argument("--config", help="YAML with game:, agents: and lbr: sections")
    ap.add_argument("--hands", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--workers", type=int)
    ap.add_argument("--preflop", choices=["call", "fc", "full"])
    ap.add_argument("--raises", choices=["spec", "fcpa"])
    ap.add_argument("--runouts", type=int, help="max flop runouts (sampled above this)")
    ap.add_argument(
        "--samples", type=int, help="act() samples per policy query for act-only agents"
    )
    ap.add_argument("--engine", choices=["auto", "reference", "rust"])
    ap.add_argument("--device", help="torch device for the equity kernels")
    ap.add_argument("--threads", type=int, help="torch CPU threads per process (default 1)")
    ap.add_argument("--out", help="write the result JSON here")
    ap.add_argument("--log-dir")
    return ap


def main(argv: list[str] | None = None) -> int:
    from ..config import load_yaml, run_info
    from .logging import write_json_atomic

    args = build_parser().parse_args(argv)
    cfg = load_yaml(args.config) if args.config else {}
    lc = dict(cfg.get("lbr") or {})

    def opt(name: str, default: Any) -> Any:
        v = getattr(args, name)
        return v if v is not None else lc.get(name, default)

    opponent = opt("opponent", None)
    if not opponent:
        print("--opponent is required", file=sys.stderr)
        return 2
    torch.set_num_threads(int(opt("threads", 1)))
    engine = args.engine or cfg.get("engine", "auto")
    info = run_info(args.config, get_engine(engine))
    print(f"# git {info['git']} | engine {info['engine']} | opponent {opponent}")
    logger = RunLogger(args.log_dir, config={"argv": argv or sys.argv[1:], **cfg})
    res = run_lbr(
        opponent,
        hands=int(opt("hands", 1000)),
        seed=int(opt("seed", 0)),
        engine=engine,
        workers=int(opt("workers", 1)),
        samples=int(opt("samples", 16)),
        logger=logger,
        game=cfg.get("game") or {},
        agent_params=cfg.get("agents") or {},
        preflop=opt("preflop", "call"),
        raises=opt("raises", "spec"),
        max_runouts=int(opt("runouts", 64)),
        device=opt("device", "cpu"),
    )
    logger.close()
    print(res.summary())
    print(f"# {res.hands} hands in {res.seconds:.1f}s")
    out = opt("out", None)
    if out:
        write_json_atomic(out, {**res.to_dict(), "run": info})
    return 0


if __name__ == "__main__":
    sys.exit(main())
