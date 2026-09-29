"""Best response in the abstract game, over a fixed sample of deals.

For small abstract games (short stacks, few actions) this walks the whole
abstract betting tree once per seat, vectorized over ``num_deals`` sampled
deals, and computes a best response to a fixed strategy that sees the same
information as the blueprint: (street, own bucket, betting sequence). The
response is built bottom-up, choosing at every infoset the action with the
highest counterfactual value summed over the sampled deals (weighted by the
opponent's reach). Its value is an exact expectation over the sampled deals,
so ``exploitability`` is a lower bound on the exploitability of the strategy
in the card-abstracted game with that deal distribution. It is a proxy for
tracking training progress, not a real-game best response.

Values are in big blinds per hand; seat 0 is the button (small blind).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

StrategyFn = Callable[[int], "np.ndarray | None"]


@dataclass
class BestResponseResult:
    br_value: tuple[float, float]  # value of the best responder in seat 0 / seat 1
    exploitability: float  # mean of the two, big blinds per hand
    infosets: int  # best-response infosets decided

    @property
    def mbb_per_hand(self) -> float:
        return 1000.0 * self.exploitability


class _Deals:
    def __init__(self, pe: Any, cards: Any, n: int, rng: np.random.Generator) -> None:
        deck = np.argsort(rng.random((n, 52)), axis=1)[:, :9].astype(np.uint8)
        self.hole = [deck[:, 0:2], deck[:, 2:4]]
        board = deck[:, 4:9]
        self.buckets = []
        for p in range(2):
            per = []
            for street, nb in enumerate((0, 3, 4, 5)):
                rows = np.ascontiguousarray(np.concatenate([self.hole[p], board[:, :nb]], axis=1))
                per.append(np.asarray(cards.buckets_batch(street, rows), dtype=np.int64))
            self.buckets.append(per)
        self.ranks = [
            np.asarray(
                pe.evaluate_batch(np.ascontiguousarray(np.concatenate([self.hole[p], board], 1))),
                dtype=np.int64,
            )
            for p in range(2)
        ]
        self.n = n


def best_response(
    strategy: StrategyFn,
    game: Any,
    actions: Any,
    cards: Any,
    num_deals: int = 20000,
    seed: int = 0,
) -> BestResponseResult:
    """Best-response values against ``strategy`` (a function from infoset key to
    probabilities over ``actions.legal(state)``, or None for uniform).

    ``game``: ``poker_engine.GameConfig`` (heads-up); ``actions``:
    ``poker_engine.ActionAbstraction``; ``cards``: an object with
    ``buckets_batch(street, cards)`` and ``num_buckets(street)`` such as
    ``poker_engine.CardAbstraction``."""
    import poker_engine as pe

    if game.num_players != 2:
        raise ValueError("heads-up only")
    deals = _Deals(pe, cards, num_deals, np.random.default_rng(seed))
    bb = float(game.big_blind)
    nbuckets = [int(cards.num_buckets(s)) for s in range(4)]
    cache: dict[int, np.ndarray] = {}
    decided = 0

    def probs_matrix(street: int, seq: list[int], n_actions: int) -> np.ndarray:
        m = np.empty((nbuckets[street], n_actions))
        for b in range(nbuckets[street]):
            key = pe.make_infoset_key(street, b, seq)
            p = cache.get(key)
            if p is None:
                q = strategy(key)
                if q is None or len(q) != n_actions:
                    p = np.full(n_actions, 1.0 / n_actions)
                else:
                    p = np.asarray(q, dtype=np.float64)
                    s = p.sum()
                    p = p / s if s > 0 else np.full(n_actions, 1.0 / n_actions)
                cache[key] = p
            m[b] = p
        return m

    def terminal(state: Any, br: int) -> np.ndarray:
        c = state.contributed
        folded = state.folded
        o = 1 - br
        if folded[br]:
            return np.full(deals.n, -c[br] / bb)
        if folded[o]:
            return np.full(deals.n, c[o] / bb)
        m = min(c[0], c[1]) / bb
        return m * np.sign(deals.ranks[br] - deals.ranks[o]).astype(np.float64)

    def value(state: Any, seq: list[int], reach: np.ndarray, br: int) -> np.ndarray:
        nonlocal decided
        if state.is_terminal:
            return terminal(state, br)
        p = state.current_player
        street = state.street
        legal = actions.legal(state)
        n = len(legal)
        bucket = deals.buckets[p][street]
        if p == br:
            vals = [value(state.child(a), seq + [i], reach, br) for i, a in legal]
            vals_m = np.stack(vals, axis=1)  # (D, n)
            weighted = vals_m * reach[:, None]
            score = np.zeros((nbuckets[street], n))
            np.add.at(score, bucket, weighted)
            best = np.argmax(score, axis=1)
            decided += int((np.bincount(bucket, minlength=nbuckets[street]) > 0).sum())
            return vals_m[np.arange(deals.n), best[bucket]]
        sigma = probs_matrix(street, seq, n)[bucket]  # (D, n)
        out = np.zeros(deals.n)
        for k, (i, a) in enumerate(legal):
            w = sigma[:, k]
            if not np.any(w > 0):
                continue
            out += w * value(state.child(a), seq + [i], reach * w, br)
        return out

    root = pe.GameState.new_hand(game, 0, list(range(52)))
    brv = []
    for br in (0, 1):
        brv.append(float(value(root, [], np.ones(deals.n), br).mean()))
    return BestResponseResult(
        br_value=(brv[0], brv[1]), exploitability=0.5 * (brv[0] + brv[1]), infosets=decided
    )


def strategy_from_file(sf: Any) -> StrategyFn:
    """Strategy function for a ``StrategyFile`` (from ``read_strategy``)."""
    table = dict(sf.items())
    return table.get


def uniform_strategy(_key: int) -> None:
    return None
