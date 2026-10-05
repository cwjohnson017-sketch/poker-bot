"""Turn-end counterfactual value net (DeepStack's auxiliary net) for depth-limit leaves.

A flop search with ``depth_streets: 1`` has its ``VALUE`` leaves at the end of
turn betting, on a 4-card board ``b4``. :class:`~.value_leaf.ValueLeafEvaluator`
values them with the river net ``N_R`` averaged over the 48 river cards (48 net
rows per leaf); the turn-end net ``N_TE`` predicts that average directly, one
row per leaf:

    ev_TE_p(c) = v_p(c) / (m_-p(c) * pot)
               = sum_x [c avoids x] * m^x_-p(c) / (44 * m_-p(c)) * N_R(b4 + x)_p(c)

(``m_-p = blocked_sum(r_-p)`` on ``b4``, ``m^x_-p`` the same with the opponent
range masked by ``x``; the weights sum to 1). Targets are bootstrapped from
``N_R`` without any solving (:mod:`.turn_data`).

**Features.** The model is :class:`~.value_net.RiverValueNet` unchanged
(bucketed ranges -> bucket values -> combos, zero-sum layer); only the
board-relative buckets differ. For every combo ``c`` valid on ``b4`` and every
river card ``x`` it can see (46 of the 48), let ``rank2^x(c)`` be its river
strength rank on ``b4 + x`` (:func:`~.value_net.strength_rank2`, percentile
``rank2 / 2160``). Per combo:

* ``S(c) = sum_x rank2^x(c)``: the mean river percentile (equity against a
  random hand, ties half), an exact integer;
* ``V(c) = 46 * sum_x rank2^x(c)^2 - S(c)^2``: ``46^2`` times its variance
  over the river cards (draws high, made hands low), also exact.

``mrank2`` / ``vrank2`` are the tie-averaged ranks of ``S`` / ``V`` among the
1128 valid combos (:func:`~.value_net.tie_rank2`), stored as int16 per board.
With ``K`` buckets and ``spread_buckets = Ks``:

* ``Ks = 1`` (1-D): bucket ``floor(K * mrank2 / 2254)``, the turn analogue of
  the river percentile buckets (nuts on top, ties shared);
* ``Ks > 1`` (2-D): ``Km = K / Ks`` mean-percentile buckets, each split into
  ``Ks`` equal-count sub-buckets by ``V`` (quantiles within the mean bucket),
  bucket ``mean_bucket * Ks + sub``. Made hands and draws of similar equity
  then get separate inputs and outputs.

Measured on check-down targets of blueprint self-play, perturbed and random
turn-end states (16k training samples, K = 256), 2-D ``32 x 8`` buckets have 8%
lower held-out MAE than 1-D ``256`` buckets (``64 x 4``: 5%), so training
defaults to ``spread_buckets = 8`` (:data:`DEFAULT_SPREAD_BUCKETS`).

``pct`` (the residual head's input) is ``mrank2 / 2254``. A turn board's
features need the rank tables of its 48 river boards (``48 * 1326`` hand
evaluations), computed once per board by :class:`TurnFeatureCache`.

**Checkpoints** are :func:`~.value_net.save_value_net` files whose ``meta``
has ``kind: turn_end`` and ``turn_features: {spread_buckets}``.
:func:`load_leaf_predictor` loads either kind (river nets have no ``kind``) as
the matching predictor; :class:`TurnEndPredictor` has the river predictor's
interface with ``boards [n, 4]``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from .combos import NUM_CARDS, NUM_COMBOS
from .value_net import (
    RiverValueNet,
    ValueNetConfig,
    ValueNetPredictor,
    board_onehot,
    features_from_rank2,
    load_value_net,
    save_value_net,
    strength_rank2,
    tie_rank2,
)

C = NUM_COMBOS
TURN_LEN = 4
RIVERS = NUM_CARDS - TURN_LEN  # 48
SEEN_RIVERS = RIVERS - 2  # river cards a valid combo can see (46)
TURN_VALID = 1128  # combos disjoint from a 4-card board: C(48, 2)
KIND = "turn_end"
# 2-D buckets by default: 32 x 8 beat 1-D 256 by 8% held-out MAE on check-down targets
DEFAULT_SPREAD_BUCKETS = 8
_KEY_BASE = torch.tensor([NUM_CARDS**i for i in range(TURN_LEN)], dtype=torch.long)


# --------------------------------------------------------------------------- features


def river_boards(boards4: torch.Tensor) -> torch.Tensor:
    """``[n, 4]`` turn boards -> ``[n, 48, 5]``: each board plus each river card
    not on it (ascending)."""
    b = boards4.long()
    n = b.shape[0]
    present = torch.zeros(n, NUM_CARDS, dtype=torch.bool, device=b.device)
    present.scatter_(1, b, True)
    rivers = (~present).nonzero()[:, 1].view(n, RIVERS)
    return torch.cat([b[:, None, :].expand(n, RIVERS, TURN_LEN), rivers[:, :, None]], 2)


@torch.no_grad()
def turn_river_sums(boards4: torch.Tensor, chunk: int = 128) -> dict[str, torch.Tensor]:
    """River-strength sums over the river cards each combo can see, ``[n, 1326]``
    long: ``S = sum_x rank2^x``, ``Q = sum_x (rank2^x)^2`` and ``count`` (46 for
    combos valid on the turn board, 0 otherwise)."""
    b = boards4.long()
    n = b.shape[0]
    dev = b.device
    out = {k: torch.zeros(n, C, dtype=torch.long, device=dev) for k in ("S", "Q", "count")}
    for s in range(0, n, chunk):
        bb = b[s : s + chunk]
        k = bb.shape[0]
        r2 = strength_rank2(river_boards(bb).reshape(k * RIVERS, 5)).view(k, RIVERS, C)
        ok = r2 >= 0
        rr = r2.clamp(min=0)
        out["S"][s : s + k] = rr.sum(1)
        out["Q"][s : s + k] = (rr * rr).sum(1)
        out["count"][s : s + k] = ok.sum(1)
    return out


@torch.no_grad()
def turn_rank_tables(boards4: torch.Tensor, chunk: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    """``[n, 4]`` boards -> ``(mrank2, vrank2)`` long ``[n, 1326]``: the
    tie-averaged ranks (``2 * smaller + ties - 1``) of the mean river strength
    ``S`` and of its spread ``V`` among the valid combos, ``-1`` on combos that
    hit the board."""
    st = turn_river_sums(boards4, chunk)
    valid = st["count"] > 0
    S, Q, n = st["S"], st["Q"], st["count"]
    V = n * Q - S * S  # n^2 * variance >= 0, exact
    mrank2 = tie_rank2(torch.where(valid, S, -1))
    vrank2 = tie_rank2(torch.where(valid, V, -1))
    return mrank2, vrank2


def turn_buckets(
    mrank2: torch.Tensor, vrank2: torch.Tensor, buckets: int, spread_buckets: int = 1
) -> torch.Tensor:
    """Turn-end buckets ``[n, 1326]`` long (``buckets`` = invalid) from the rank
    tables: 1-D mean-percentile buckets, or ``buckets / spread_buckets`` mean
    buckets split into ``spread_buckets`` equal-count spread sub-buckets."""
    K, Ks = int(buckets), int(spread_buckets)
    if Ks <= 1:
        return features_from_rank2(mrank2, K)["bucket"]
    if K % Ks:
        raise ValueError(f"buckets ({K}) must be a multiple of spread_buckets ({Ks})")
    Km = K // Ks
    f = features_from_rank2(mrank2, Km)
    valid, mb = f["valid"], f["bucket"]  # mb = Km on invalid combos
    vr = vrank2.long().clamp(min=0)
    key = torch.where(valid, mb * (4 * C) + vr, torch.full_like(mb, -1))  # order by (mb, V)
    r = tie_rank2(key)
    cnt = torch.zeros(mb.shape[0], Km + 1, dtype=torch.long, device=mb.device)
    cnt.scatter_add_(1, mb, valid.long())
    below = (cnt.cumsum(1) - cnt).gather(1, mb)  # valid combos in lower mean buckets
    nk = cnt.gather(1, mb).clamp(min=1)
    within = r - 2 * below  # rank2 inside the mean bucket: 0 .. 2 (nk - 1)
    sub = torch.clamp((Ks * (within + 1)) // (2 * nk), 0, Ks - 1)
    return torch.where(valid, mb * Ks + sub, torch.full_like(mb, K))


def turn_board_features(
    boards4: torch.Tensor, buckets: int = 256, spread_buckets: int = 1
) -> dict[str, torch.Tensor]:
    """Per-combo turn-end features of ``[n, 4]`` boards (reference / analysis
    path): ``valid``, ``pct`` (mean-strength percentile), ``bucket``, ``mrank2``,
    ``vrank2``, ``equity`` (mean river percentile ``S / (46 * 2160)``),
    ``spread`` (its standard deviation over the river cards) and ``onehot``."""
    st = turn_river_sums(boards4)
    valid = st["count"] > 0
    S, Q, n = st["S"].double(), st["Q"].double(), st["count"].double().clamp(min=1)
    den = 2.0 * (1081 - 1)
    mean = S / n
    var = (Q / n - mean * mean).clamp(min=0)
    V = st["count"] * st["Q"] - st["S"] * st["S"]
    mrank2 = tie_rank2(torch.where(valid, st["S"], -1))
    vrank2 = tie_rank2(torch.where(valid, V, -1))
    f = features_from_rank2(mrank2, buckets)
    return {
        "valid": valid,
        "pct": f["pct"],
        "bucket": turn_buckets(mrank2, vrank2, buckets, spread_buckets),
        "mrank2": mrank2,
        "vrank2": vrank2,
        "equity": torch.where(valid, mean / den, 0.0).float(),
        "spread": torch.where(valid, var.sqrt() / den, 0.0).float(),
        "onehot": board_onehot(boards4),
    }


class TurnFeatureCache:
    """Turn-end feature tables of 4-card boards on ``device``, keyed by the
    sorted board: the :class:`~.value_net.BoardFeatureCache` interface (``ids``,
    ``features``, ``generation``) with ``spread_buckets`` fixed per cache.

    Stores ``mrank2`` and ``vrank2`` (int16) per board; with ``tables`` an int16
    bucket table per ``K`` is kept too (one gather per query), without it the
    buckets are derived per query (a row sort for 2-D buckets). Adding boards
    beyond ``max_boards`` clears the cache first (``generation`` is incremented).
    """

    def __init__(
        self,
        device: torch.device | str = "cpu",
        spread_buckets: int = 1,
        max_boards: int | None = 1 << 14,
        tables: bool = True,
    ):
        self.device = torch.device(device)
        self.spread_buckets = int(spread_buckets)
        self.max_boards = max_boards
        self.tables = tables
        self.generation = 0
        self.clear()

    def clear(self) -> None:
        self._index: dict[int, int] = {}
        self._mrank2 = torch.empty(0, C, dtype=torch.int16, device=self.device)
        self._vrank2 = torch.empty(0, C, dtype=torch.int16, device=self.device)
        self._boards = torch.empty(0, TURN_LEN, dtype=torch.long, device=self.device)
        self._buckets: dict[int, list] = {}  # K -> [int16 [cap, C] table, rows filled]
        self.generation += 1

    def __len__(self) -> int:
        return len(self._index)

    @property
    def boards(self) -> torch.Tensor:
        return self._boards[: len(self)]

    def _reserve(self, n: int) -> None:
        cap = self._mrank2.shape[0]
        if n <= cap:
            return
        new = max(n, 2 * cap, 256)

        def grow(t: torch.Tensor) -> torch.Tensor:
            g = torch.empty(new, *t.shape[1:], dtype=t.dtype, device=self.device)
            g[:cap] = t
            return g

        self._mrank2, self._vrank2, self._boards = map(
            grow, (self._mrank2, self._vrank2, self._boards)
        )
        for t in self._buckets.values():
            t[0] = grow(t[0])

    def _add(self, boards: torch.Tensor, keys: list[int]) -> None:
        start, k = len(self), len(keys)
        self._reserve(start + k)
        mr, vr = turn_rank_tables(boards)
        self._mrank2[start : start + k] = mr.to(torch.int16)
        self._vrank2[start : start + k] = vr.to(torch.int16)
        self._boards[start : start + k] = boards
        for i, key in enumerate(keys):
            self._index[key] = start + i

    def ids(self, boards: torch.Tensor) -> torch.Tensor:
        """``[n, 4]`` boards (any card order) -> ``[n]`` long cache ids."""
        b = boards.to(self.device).long().sort(1).values
        if b.shape[0] == 0:
            return torch.empty(0, dtype=torch.long, device=self.device)
        if b.shape[1] != TURN_LEN:
            raise ValueError(f"turn-end boards must have 4 cards, got {b.shape[1]}")
        key = (b * _KEY_BASE.to(b.device)).sum(1)
        ukey, inv = torch.unique(key, return_inverse=True)
        keys = ukey.tolist()
        new = [i for i, k in enumerate(keys) if k not in self._index]
        if new:
            if self.max_boards is not None and len(self) + len(new) > self.max_boards:
                self.clear()
                new = list(range(len(keys)))
            sel = torch.tensor(new, dtype=torch.long, device=self.device)
            first = torch.full((len(keys),), b.shape[0], dtype=torch.long, device=self.device)
            first.scatter_reduce_(0, inv, torch.arange(b.shape[0], device=self.device), "amin")
            self._add(b[first[sel]], [keys[i] for i in new])
        uid = torch.tensor([self._index[k] for k in keys], dtype=torch.long)
        return uid.to(self.device)[inv]

    def _bucket_table(self, K: int) -> torch.Tensor:
        t = self._buckets.get(K)
        if t is None:
            t = self._buckets[K] = [
                torch.empty(self._mrank2.shape[0], C, dtype=torch.int16, device=self.device),
                0,
            ]
        n = len(self)
        if t[1] < n:
            t[0][t[1] : n] = turn_buckets(
                self._mrank2[t[1] : n], self._vrank2[t[1] : n], K, self.spread_buckets
            ).to(torch.int16)
            t[1] = n
        return t[0]

    def features(
        self, ids: torch.Tensor, buckets: int, pct: bool = True
    ) -> dict[str, torch.Tensor]:
        """Per-row ``valid`` bool, ``bucket`` int32 (``buckets`` = invalid), the
        ``onehot`` [n, 52] board and, when ``pct``, ``pct`` float [n, 1326]."""
        ids = ids.to(self.device)
        mr = self._mrank2[ids]
        if self.tables:
            bucket = self._bucket_table(buckets)[ids].to(torch.int32)
        else:
            bucket = turn_buckets(mr, self._vrank2[ids], buckets, self.spread_buckets)
            bucket = bucket.to(torch.int32)
        out = {"bucket": bucket, "valid": bucket < buckets}
        if pct:
            out["pct"] = mr.clamp(min=0).float() / (2.0 * (TURN_VALID - 1))
        out["onehot"] = board_onehot(self._boards[ids])
        return out


# --------------------------------------------------------------------------- predictor


class TurnEndPredictor(ValueNetPredictor):
    """Batched turn-end net inference: :class:`~.value_net.ValueNetPredictor`
    with a :class:`TurnFeatureCache`, so ``predict(boards [n, 4], ranges [n, 2,
    1326], c, stack) -> ev [n, 2, 1326]`` (pot units per unit of disjoint
    opponent mass on the 4-card board, ``(OOP, IP)``, 0 on invalid combos)."""

    kind = KIND

    def __init__(
        self,
        net: RiverValueNet,
        device: torch.device | str | None = None,
        cache: TurnFeatureCache | None = None,
        head_chunk: int = 2048,
        spread_buckets: int | None = None,
    ):
        dev = torch.device(device) if device is not None else next(net.parameters()).device
        if spread_buckets is None:
            meta = getattr(net, "meta", None) or {}
            spread_buckets = meta.get("turn_features", {}).get("spread_buckets")
            if spread_buckets is None:
                raise ValueError(
                    "the net's meta has no turn_features.spread_buckets: pass spread_buckets "
                    "or load a turn-end checkpoint"
                )
        if cache is None:
            cache = TurnFeatureCache(dev, spread_buckets)
        elif cache.spread_buckets != int(spread_buckets):
            raise ValueError("the cache's spread_buckets differ from the net's")
        if net.cfg.buckets % int(spread_buckets):
            raise ValueError("the net's buckets must be a multiple of spread_buckets")
        self.spread_buckets = int(spread_buckets)
        super().__init__(net, dev, cache, head_chunk)

    @classmethod
    def from_path(cls, path: str | Path, device: torch.device | str = "cpu") -> TurnEndPredictor:
        net = load_value_net(path, device)
        kind = net.meta.get("kind", "river")
        if kind != KIND:
            raise ValueError(f"{path} is a {kind} value net, not a turn-end net")
        return cls(net, device)


# --------------------------------------------------------------------------- checkpoints


def turn_meta(spread_buckets: int, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Checkpoint ``meta`` of a turn-end net."""
    return {**(extra or {}), "kind": KIND, "turn_features": {"spread_buckets": int(spread_buckets)}}


def save_turn_net(
    path: str | Path, net: RiverValueNet, spread_buckets: int, meta: dict[str, Any] | None = None
) -> None:
    save_value_net(path, net, turn_meta(spread_buckets, meta))


def checkpoint_kind(path: str | Path) -> str:
    """``"turn_end"`` or ``"river"`` (the default when ``meta`` has no ``kind``)."""
    ck = torch.load(path, map_location="cpu", weights_only=True)
    return str((ck.get("meta") or {}).get("kind", "river"))


def load_leaf_predictor(
    path: str | Path, device: torch.device | str = "cpu"
) -> ValueNetPredictor | TurnEndPredictor:
    """The predictor of a value-net checkpoint, by its ``meta`` kind: a
    :class:`TurnEndPredictor` for a turn-end net, else a river
    :class:`~.value_net.ValueNetPredictor`."""
    net = load_value_net(path, device)
    kind = net.meta.get("kind", "river")
    if kind == KIND:
        return TurnEndPredictor(net, device)
    if kind == "river":
        return ValueNetPredictor(net, device)
    raise ValueError(f"{path}: unknown value-net kind {kind!r}")


# --------------------------------------------------------------------------- training


def train_turn_net(
    data: str | Path | Sequence[str | Path] | None,
    out: str | Path | None = None,
    net_cfg: ValueNetConfig | None = None,
    cfg: Any = None,
    spread_buckets: int = DEFAULT_SPREAD_BUCKETS,
    heldout: str | Path | Sequence[str | Path] | None = None,
    device: str | torch.device = "cuda",
    log: Any = print,
    raw: dict[str, torch.Tensor] | None = None,
    raw_heldout: dict[str, torch.Tensor] | None = None,
) -> tuple[RiverValueNet, dict[str, Any]]:
    """:func:`~.value_train.train_value_net` on turn-end shards (boards
    ``[n, 4]``) with :class:`TurnFeatureCache` features; the checkpoint's meta
    records ``kind: turn_end`` and ``spread_buckets``."""
    from .value_train import train_value_net

    net_cfg = net_cfg or ValueNetConfig()
    sb = int(spread_buckets)
    if net_cfg.buckets % sb:
        raise ValueError(f"buckets ({net_cfg.buckets}) must be a multiple of spread_buckets ({sb})")

    def factory(dev: torch.device) -> TurnFeatureCache:
        return TurnFeatureCache(dev, sb, max_boards=None, tables=True)

    return train_value_net(
        data,
        out,
        net_cfg,
        cfg,
        heldout,
        device,
        log,
        raw,
        raw_heldout,
        cache_factory=factory,
        meta=turn_meta(sb),
    )


__all__ = [
    "DEFAULT_SPREAD_BUCKETS",
    "KIND",
    "TurnEndPredictor",
    "TurnFeatureCache",
    "checkpoint_kind",
    "load_leaf_predictor",
    "river_boards",
    "save_turn_net",
    "train_turn_net",
    "turn_board_features",
    "turn_buckets",
    "turn_meta",
    "turn_rank_tables",
    "turn_river_sums",
]
