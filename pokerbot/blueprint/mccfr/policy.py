"""Stateless, batched queries of an exported MCCFR strategy.

:class:`TabularPolicy` answers "what does the blueprint play here, for this
hand?" from a contract ``GameState`` alone (a ``poker_engine`` or reference
state, the match runner's masked view, or the search's ``CardView``), with no
per-hand bookkeeping, and for many hands at once:

* **Betting.** The public action history is replayed into two
  ``poker_engine`` shadow states on a dummy deck, as :class:`BlueprintAgent`
  does: ``real`` (the actual game) and ``abs`` (the training game), with
  every action mapped into ``abs`` by ``ActionAbstraction.translate``. The
  mapping is deterministic (``u = 0.5`` by default: the more likely side of
  the pseudo-harmonic split), because a policy has to be a function of the
  state. Both players' actions are translated, including the queried
  player's own. Replays are cached per history prefix, so a state that
  extends a cached one costs one ``apply``.
* **Cards.** The infoset key is ``(street, bucket, abstract sequence)``
  (``export.make_key``). For a batch of hands the buckets come from one
  ``CardAbstraction.buckets_batch`` call per board (built from the
  strategy's own card config and cached per board), and each distinct bucket
  is looked up once. The single-hand path, :meth:`TabularPolicy.probs`, goes
  through ``BlueprintStrategy.action_probs`` instead (the Rust key and bucket
  code), so the two paths check each other.

Probabilities are over the indices of :attr:`TabularPolicy.spec`, the
blueprint's action list as a :class:`~pokerbot.env.actions.ActionSpec`. The
row holds the stored strategy on the entries of ``ActionAbstraction.legal``
in the abstract state (uniform over them when the infoset is missing). When
the abstract game no longer tracks the hand (a re-raise past the raise cap,
a mapped all-in that was not one), the row is check/call, as the agent's
fallback. Mass on abstract actions that are not legal in the real state
(when the real stacks differ from the training game) moves to check/call,
which is what the agent plays when an abstract action is unavailable. Hands
that share a card with the board get a zero row.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from collections.abc import Sequence
from itertools import combinations
from typing import Any

import numpy as np

from .export import make_key

RAISE = 2
BOARD_LEN = (0, 3, 4, 5)
ALL_COMBOS = np.array(list(combinations(range(52), 2)), dtype=np.int64)  # [1326, 2]


def spec_from_solver_config(cfg: dict[str, Any]) -> Any:
    """The ``ActionSpec`` of a solver config (``BlueprintStrategy.config_json``).

    The MCCFR action abstraction and ``pokerbot.env.actions`` use the same
    ``raise`` rule (``max_bet + f * (pot + to_call)``, clamped, all-in sizes
    left to ``allin``, duplicates merged into the earliest entry)."""
    from ...env.actions import spec_from_lists

    acts = cfg["actions"]
    streets = []
    for st in acts["streets"]:
        row = []
        for a in st:
            a = [a] if isinstance(a, str) else list(a)
            if len(a) == 1:
                row.append((str(a[0]),))
            else:
                row.append((str(a[0]), float(a[1]), *(str(x) for x in a[2:])))
        streets.append(row)
    return spec_from_lists(streets, max_raises=int(acts.get("max_raises", 4)), dedupe=True)


def _combo_index(holes: np.ndarray) -> np.ndarray:
    a = np.minimum(holes[:, 0], holes[:, 1])
    b = np.maximum(holes[:, 0], holes[:, 1])
    return a * (103 - a) // 2 + (b - a - 1)


class _Node:
    """Replayed betting: the real and abstract shadow states after a history."""

    __slots__ = ("real", "abs", "_info")

    def __init__(self, real: Any, abs_state: Any) -> None:
        self.real = real
        self.abs = abs_state  # None once the abstract game stopped tracking
        self._info: tuple | None = None

    def info(self, abstraction: Any) -> tuple[list[int], tuple[int, ...]]:
        """(legal abstract indices, abstract sequence) at the abstract state."""
        if self._info is None:
            legal = [int(i) for i, _a in abstraction.legal(self.abs)]
            seq = tuple(int(i) for i in abstraction.sequence(self.abs))
            self._info = (legal, seq)
        return self._info


class TabularPolicy:
    """Batched, stateless policy of a ``poker_engine.BlueprintStrategy``."""

    def __init__(
        self,
        strategy: Any,
        mapping_u: float = 0.5,
        cache_size: int = 200_000,
        board_cache_size: int = 4096,
    ) -> None:
        import poker_engine as pe

        self._pe = pe
        self.strategy = strategy
        self.abstraction = strategy.action_abstraction()
        self.solver_config = json.loads(strategy.config_json)
        self.spec = spec_from_solver_config(self.solver_config)
        self.mapping_u = float(mapping_u)
        self.cache_size = int(cache_size)
        self.board_cache_size = int(board_cache_size)
        self._nodes: dict[tuple, _Node] = {}
        self._buckets: OrderedDict[tuple, np.ndarray] = OrderedDict()
        self._cards: Any = None
        self._call = [
            next(i for i, a in enumerate(st) if a[0] == "check_call") for st in self.spec.streets
        ]

    @classmethod
    def load(cls, path: str, **kwargs: Any) -> TabularPolicy:
        import poker_engine as pe

        return cls(pe.BlueprintStrategy(str(path)), **kwargs)

    # ------------------------------------------------------------ card buckets
    @property
    def card_abstraction(self) -> Any:
        """``poker_engine.CardAbstraction`` with the strategy's card config
        (built on first use; the tests check it buckets like the strategy)."""
        if self._cards is None:
            c = self.solver_config.get("cards") or {}
            self._cards = self._pe.CardAbstraction(
                [int(b) for b in c["buckets"]], int(c.get("hs_samples", 256)), c.get("tables")
            )
        return self._cards

    def board_buckets(self, street: int, board: Sequence[int]) -> np.ndarray:
        """``[1326]`` int64 bucket of every combo on ``board`` (``-1`` when the
        combo shares a card with it), cached per board."""
        key = (int(street), tuple(int(c) for c in board))
        out = self._buckets.get(key)
        if out is not None:
            self._buckets.move_to_end(key)
            return out
        board = list(key[1])
        out = np.full(len(ALL_COMBOS), -1, dtype=np.int64)
        ok = ~np.isin(ALL_COMBOS, board).any(1)
        rows = ALL_COMBOS[ok]
        cards = np.concatenate(
            [rows, np.tile(np.asarray(board, dtype=np.int64), (len(rows), 1))], 1
        )
        out[ok] = self.card_abstraction.buckets_batch(int(street), cards.astype(np.uint8))
        self._buckets[key] = out
        if len(self._buckets) > self.board_cache_size:
            self._buckets.popitem(last=False)
        return out

    # ------------------------------------------------------------ betting
    def _config_key(self, state: Any) -> tuple:
        cfg = state.config
        return (
            int(state.button),
            tuple(int(s) for s in cfg.stacks),
            int(cfg.small_blind),
            int(cfg.big_blind),
            int(getattr(cfg, "ante", 0)),
        )

    def _root(self, base: tuple) -> _Node:
        pe = self._pe
        button, stacks, sb, bb, ante = base
        real_cfg = pe.GameConfig(
            num_players=2, stacks=list(stacks), small_blind=sb, big_blind=bb, ante=ante
        )
        deck = list(range(52))
        return _Node(
            pe.GameState.new_hand(real_cfg, button, deck),
            pe.GameState.new_hand(self.strategy.game_config, button, deck),
        )

    def _extend(self, node: _Node, kind: int, amount: int) -> _Node:
        a = self._pe.action_from(kind, amount)
        abs_next = None
        ab = node.abs
        if ab is not None and not ab.is_terminal:
            idx = self.abstraction.translate(ab, node.real, a, self.mapping_u)
            abs_a = None if idx is None else self.abstraction.to_concrete(ab, idx)
            if abs_a is not None:
                abs_next = ab.child(abs_a)
        return _Node(node.real.child(a), abs_next)

    def node(self, state: Any) -> _Node:
        """The replayed shadow states for ``state``'s public history."""
        base = self._config_key(state)
        hist = tuple(
            (int(a.kind), int(a.amount) if int(a.kind) == RAISE else 0)
            for _s, _p, a in state.history
        )
        got = self._nodes.get((base, hist))
        if got is not None:
            return got
        n = len(hist)
        while n > 0 and (base, hist[:n]) not in self._nodes:
            n -= 1
        node = self._nodes.get((base, hist[:n])) or self._root(base)
        if len(self._nodes) + len(hist) - n > self.cache_size:
            self._nodes.clear()
        for i in range(n, len(hist)):
            node = self._extend(node, *hist[i])
            self._nodes[(base, hist[: i + 1])] = node
        if not hist:
            self._nodes[(base, hist)] = node
        return node

    def tracked(self, state: Any, player: int) -> _Node | None:
        """The node when the abstract game tracks ``state`` with ``player`` to
        act on the same street, else None (the agent's check/call fallback)."""
        node = self.node(state)
        ab = node.abs
        if (
            ab is None
            or ab.is_terminal
            or int(ab.current_player) != int(player)
            or int(ab.street) != int(state.street)
        ):
            return None
        return node

    def _fallback(self, street: int) -> np.ndarray:
        v = np.zeros(self.spec.num_actions)
        v[self._call[min(int(street), 3)]] = 1.0
        return v

    def _to_real(self, rows: np.ndarray, state: Any) -> np.ndarray:
        """Move the mass of abstract actions that are not legal in the real
        state (other stacks than the training game) to check/call, as the
        agent plays check/call when an abstract action is unavailable."""
        from ...agents.policy import legal_vector

        legal = legal_vector(state, self.spec)
        if legal.all():
            return rows
        call = self._call[min(int(state.street), 3)]
        lost = rows[..., ~legal].sum(-1)
        rows = np.where(legal, rows, 0.0)
        rows[..., call] += lost
        return rows

    # ------------------------------------------------------------ queries
    def probs(self, state: Any, player: int, hole: Sequence[int] | None = None) -> np.ndarray:
        """``[A]`` probabilities for one hand (``state.hole_cards(player)`` by
        default), through ``BlueprintStrategy.action_probs``."""
        hole = [int(c) for c in (state.hole_cards(player) if hole is None else hole)]
        board = [int(c) for c in state.board]
        if set(hole) & set(board):
            return np.zeros(self.spec.num_actions)
        node = self.tracked(state, player)
        if node is None:
            return self._fallback(state.street)
        out = np.zeros(self.spec.num_actions)
        idxs, _acts, probs, _found = self.strategy.action_probs(node.abs, hole, board)
        out[np.asarray(idxs, dtype=np.int64)] = np.asarray(probs, dtype=np.float64)
        s = out.sum()
        out = out / s if s > 0 else self._fallback(state.street)
        return self._to_real(out, state)

    def probs_batch(self, state: Any, player: int, holes: Any = None) -> np.ndarray:
        """``[K, A]`` probabilities for the hands ``holes`` (``[K, 2]``; all
        1326 combos in canonical order when None). Rows of hands that share a
        card with the board are zero."""
        holes = ALL_COMBOS if holes is None else np.asarray(holes, dtype=np.int64).reshape(-1, 2)
        K, A = len(holes), self.spec.num_actions
        street = int(state.street)
        board = [int(c) for c in state.board][: BOARD_LEN[min(street, 3)]]
        valid = ~np.isin(holes, board).any(1) & (holes[:, 0] != holes[:, 1])
        node = self.tracked(state, player)
        if node is None:
            return np.where(valid[:, None], self._fallback(street)[None, :], 0.0)
        bk = self.board_buckets(street, board)[_combo_index(holes)]
        legal, seq = node.info(self.abstraction)
        uniq, inv = np.unique(bk[valid], return_inverse=True)
        table = np.zeros((len(uniq), A))
        cols = np.asarray(legal, dtype=np.int64)
        for j, b in enumerate(uniq.tolist()):
            row = self.strategy.lookup(make_key(street, int(b), seq))
            table[j, cols] = 1.0 / len(cols) if row is None else np.asarray(row, dtype=np.float64)
        s = table.sum(1, keepdims=True)
        table = np.where(s > 0, table / np.where(s > 0, s, 1.0), self._fallback(street)[None, :])
        table = self._to_real(table, state)
        out = np.zeros((K, A))
        out[valid] = table[inv]
        return out
