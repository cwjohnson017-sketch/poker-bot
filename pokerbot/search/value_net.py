"""River-start counterfactual value network (``docs/value_net.md``, sections 1-2).

A state is a 5-card board, ``c`` (chips each player has committed), ``stack``
(chips behind each player) and both players' ranges over the 1326 combos of
:mod:`.combos`, in ``(OOP, IP)`` order. For player ``p`` and combo ``c`` the net
predicts

    ev_p(c) = v_p(c) / m_{-p}(c) / pot,      pot = 2 c,

with ``v_p`` the counterfactual value in chips at the river root and
``m_{-p}(c) = blocked_sum(r_{-p})(c)`` the opponent mass disjoint from ``c``:
the expected chips per hand against the opponent's card-removed range, in pot
units.

* **Strength buckets.** Every valid combo gets its board-relative strength
  percentile ``(valid weaker + (ties - 1) / 2) / (valid - 1)`` and the bucket
  ``min(K - 1, floor(K * pct))`` (index ``K`` for combos that hit the board).
  Ties share a bucket and a lone nut combo is in bucket ``K - 1``. Boards are
  stored as ``rank2 = 2 * weaker + ties - 1`` (int16, independent of ``K``),
  so ``pct = rank2 / (2 * (valid - 1))`` and the bucket is an integer division.
* **Inputs.** Each player's bucketed range ``R_p[k] = K * sum_{c in k} r_p(c)``
  (mean 1), ``c / (c + stack)``, ``log(stack / pot)`` (clamped to
  ``[log 0.01, log 100]``), ``1 / (1 + stack / pot)`` and the board one-hot.
* **Trunk.** An MLP (plain, or pre-LayerNorm residual blocks) to ``2K``
  bucket values, decoded to combos by each combo's bucket.
* **Residual head** (optional). A small per-combo MLP on the decoded value,
  ``pct``, the opponent mass the combo blocks, the strength-weighted blocked
  opponent mass, the own reach and the pot geometry, adding a correction.
* **Zero-sum layer** (DeepStack). With ``w_p = r_p * m_{-p}`` and the pair mass
  ``Z = sum w_0 = sum w_1``, ``delta = (sum w_0 ev_0 + sum w_1 ev_1) / (2 Z)``
  is subtracted from every ``ev``, so the range-weighted game values of the two
  players sum to zero.

:class:`ValueNetPredictor` is the inference entry point for the solver.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ..env.evaluator import evaluate7_batch
from .combos import NUM_CARDS, NUM_COMBOS, blocked_sum, combo_table, incidence

C = NUM_COMBOS
NUM_CTX = 3  # context scalars (see context_features)
MASS_EPS = 1e-4  # opponent disjoint mass below this: the combo has no target
LOG_SPR_MIN, LOG_SPR_MAX = math.log(1e-2), math.log(1e2)
_KEY_BASE = torch.tensor([NUM_CARDS**i for i in range(5)], dtype=torch.long)


# --------------------------------------------------------------------------- board features


def board_onehot(boards: torch.Tensor) -> torch.Tensor:
    """``[n, 5]`` cards -> ``[n, 52]`` float one-hot."""
    b = boards.long()
    return torch.zeros(b.shape[0], NUM_CARDS, device=b.device).scatter_(1, b, 1.0)


@torch.no_grad()
def board_strengths(boards: torch.Tensor, chunk: int = 256) -> torch.Tensor:
    """``[n, 5]`` boards -> ``[n, 1326]`` long hand strengths, ``-1`` on board
    conflicts. Same values as :func:`.showdown.combo_strengths`, with the
    validity mask computed by one matmul instead of a Python loop over boards
    (that loop dominates on CUDA)."""
    b = boards.long()
    dev = b.device
    cards = combo_table(dev)
    n = b.shape[0]
    out = torch.empty(n, C, dtype=torch.long, device=dev)
    for s in range(0, n, chunk):
        bb = b[s : s + chunk]
        k = bb.shape[0]
        seven = torch.cat([bb[:, None, :].expand(k, C, 5), cards[None].expand(k, C, 2)], 2)
        out[s : s + k] = evaluate7_batch(seven.reshape(-1, 7)).view(k, C)
    hit = board_onehot(b) @ incidence(dev, torch.float32).t()
    return out.masked_fill_(hit > 0, -1)


@torch.no_grad()
def strength_rank2(boards: torch.Tensor) -> torch.Tensor:
    """``[n, 5]`` boards -> ``[n, 1326]`` long ``2 * (valid weaker) + ties - 1``
    (ties count the combo itself), ``-1`` for combos that hit the board."""
    s = board_strengths(boards)
    valid = s >= 0
    n_bad = (~valid).sum(1, keepdim=True)
    ss = s.sort(1).values  # the -1 entries come first
    lo = torch.searchsorted(ss, s, right=False) - n_bad  # valid combos strictly weaker
    hi = torch.searchsorted(ss, s, right=True) - n_bad  # ... weaker or tied (incl. itself)
    return torch.where(valid, lo + hi - 1, torch.full_like(s, -1))


def features_from_rank2(rank2: torch.Tensor, buckets: int) -> dict[str, torch.Tensor]:
    """``rank2 [n, 1326]`` -> ``valid`` bool, ``pct`` float, ``bucket`` long
    (``buckets`` for invalid combos)."""
    r = rank2.long()
    valid = r >= 0
    den = (2 * (valid.sum(1, keepdim=True) - 1)).clamp(min=1)
    rc = r.clamp(min=0)
    pct = rc.float() / den
    bucket = torch.clamp((buckets * rc) // den, max=buckets - 1)
    bucket = torch.where(valid, bucket, torch.full_like(bucket, buckets))
    return {"valid": valid, "pct": pct, "bucket": bucket}


def river_board_features(boards: torch.Tensor, buckets: int = 256) -> dict[str, torch.Tensor]:
    """Per-combo strength features of ``[n, 5]`` river boards: ``valid`` [n, 1326]
    bool, ``pct`` [n, 1326] float (board-relative strength percentile among
    valid combos, ties averaged), ``bucket`` [n, 1326] long (``K`` = invalid),
    ``rank2`` and the ``onehot`` [n, 52] board."""
    rank2 = strength_rank2(boards)
    out = features_from_rank2(rank2, buckets)
    out["rank2"] = rank2
    out["onehot"] = board_onehot(boards)
    return out


class BoardFeatureCache:
    """``rank2`` tables of river boards on ``device``, keyed by the sorted board.

    :meth:`ids` maps ``[n, 5]`` boards to row ids into the cache (computing new
    boards in one batch); :meth:`features` gathers per-row features by id. The
    cache is ``K``-independent. When adding boards would exceed ``max_boards``
    the cache is cleared first (``generation`` is incremented), so ids from an
    earlier call stay valid until a later :meth:`ids` call triggers a reset.
    """

    def __init__(self, device: torch.device | str = "cpu", max_boards: int | None = 1 << 17):
        self.device = torch.device(device)
        self.max_boards = max_boards
        self.generation = 0
        self.clear()

    def clear(self) -> None:
        self._index: dict[int, int] = {}  # board key -> row
        self._rank2 = torch.empty(0, C, dtype=torch.int16, device=self.device)
        self._boards = torch.empty(0, 5, dtype=torch.long, device=self.device)
        self.generation += 1

    def __len__(self) -> int:
        return len(self._index)

    @property
    def rank2(self) -> torch.Tensor:
        return self._rank2[: len(self)]

    @property
    def boards(self) -> torch.Tensor:
        return self._boards[: len(self)]

    def _reserve(self, n: int) -> None:
        cap = self._rank2.shape[0]
        if n <= cap:
            return
        new = max(n, 2 * cap, 1024)
        r = torch.empty(new, C, dtype=torch.int16, device=self.device)
        b = torch.empty(new, 5, dtype=torch.long, device=self.device)
        r[:cap] = self._rank2
        b[:cap] = self._boards
        self._rank2, self._boards = r, b

    def _add(self, boards: torch.Tensor, keys: list[int]) -> None:
        start, k = len(self), len(keys)
        self._reserve(start + k)
        step = 8192
        for s in range(0, k, step):
            e = min(k, s + step)
            self._rank2[start + s : start + e] = strength_rank2(boards[s:e])
        self._boards[start : start + k] = boards
        for i, key in enumerate(keys):
            self._index[key] = start + i

    def ids(self, boards: torch.Tensor) -> torch.Tensor:
        """``[n, 5]`` boards (any card order) -> ``[n]`` long cache ids."""
        b = boards.to(self.device).long().sort(1).values
        if b.shape[0] == 0:
            return torch.empty(0, dtype=torch.long, device=self.device)
        key = (b * _KEY_BASE.to(b.device)).sum(1)  # one int64 per sorted board
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

    def features(self, ids: torch.Tensor, buckets: int) -> dict[str, torch.Tensor]:
        """Per-row ``valid``, ``pct``, ``bucket`` [n, 1326] and ``onehot`` [n, 52]."""
        ids = ids.to(self.device)
        out = features_from_rank2(self._rank2[ids], buckets)
        out["onehot"] = board_onehot(self._boards[ids])
        return out


# --------------------------------------------------------------------------- model


@dataclass
class ValueNetConfig:
    buckets: int = 256  # K strength buckets
    width: int = 1024
    layers: int = 4  # hidden layers of the trunk
    dropout: float = 0.0
    block: str = "mlp"  # "mlp" (Linear [+ LayerNorm] + act) or "resnet" (pre-LN residual)
    layer_norm: bool = False  # LayerNorm in the "mlp" trunk
    activation: str = "gelu"  # gelu | relu
    count_input: bool = False  # also feed the valid-combo count per bucket (board shape)
    residual_head: bool = False
    head_width: int = 32

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> ValueNetConfig:
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in names})

    def input_dim(self) -> int:
        K = self.buckets
        return 2 * K + (K if self.count_input else 0) + NUM_CTX + NUM_CARDS


def context_features(c: torch.Tensor, stack: torch.Tensor) -> torch.Tensor:
    """``[n]`` committed chips and stacks -> ``[n, 3]``: ``c / (c + stack)``,
    clamped ``log(stack / pot)`` and ``1 / (1 + stack / pot)``."""
    c = c.float()
    stack = stack.float()
    spr = stack / (2 * c).clamp(min=1.0)
    return torch.stack(
        [
            c / (c + stack).clamp(min=1.0),
            torch.log(spr.clamp(min=1e-2)).clamp(LOG_SPR_MIN, LOG_SPR_MAX),
            1.0 / (1.0 + spr),
        ],
        -1,
    )


def normalise_ranges(ranges: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """``[n, 2, 1326]`` non-negative reaches (any scale) -> ranges summing to 1
    over the valid combos; an all-zero range becomes uniform over them."""
    v = valid[:, None, :].to(torch.float32)
    r = ranges.float().clamp(min=0) * v
    s = r.sum(-1, keepdim=True)
    uni = v / v.sum(-1, keepdim=True).clamp(min=1.0)
    return torch.where(s > 0, r / s.clamp(min=1e-30), uni)


def opponent_mass(ranges: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """``m[:, p] = blocked_sum(r_{-p})``: opponent mass disjoint from each combo
    (zero on invalid combos)."""
    return blocked_sum(ranges.flip(1).float()) * valid[:, None, :]


def zero_sum(ev: torch.Tensor, ranges: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """Shift ``ev [n, 2, 1326]`` so that ``sum w_0 ev_0 + sum w_1 ev_1 = 0`` with
    ``w_p = r_p * m_{-p}``."""
    w = ranges * m
    tot = w.sum((1, 2))  # = 2 Z
    s = (w * ev).sum((1, 2))
    delta = torch.where(tot > 1e-12, s / tot.clamp(min=1e-12), torch.zeros_like(s))
    return ev - delta[:, None, None]


def _act(name: str) -> nn.Module:
    return {"gelu": nn.GELU, "relu": nn.ReLU}[name]()


class _ResBlock(nn.Module):
    def __init__(self, width: int, act: str, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.lin = nn.Linear(width, width)
        self.act = _act(act)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop(self.lin(self.act(self.norm(x))))


class RiverValueNet(nn.Module):
    """Bucketed river value net; :meth:`forward` returns per-combo ``ev`` [n, 2, 1326]."""

    def __init__(self, cfg: ValueNetConfig | None = None):
        super().__init__()
        self.cfg = cfg = cfg or ValueNetConfig()
        K, W = cfg.buckets, cfg.width
        if cfg.block == "mlp":
            mods: list[nn.Module] = []
            d = cfg.input_dim()
            for _ in range(cfg.layers):
                mods.append(nn.Linear(d, W))
                if cfg.layer_norm:
                    mods.append(nn.LayerNorm(W))
                mods.append(_act(cfg.activation))
                if cfg.dropout > 0:
                    mods.append(nn.Dropout(cfg.dropout))
                d = W
            self.trunk = nn.Sequential(*mods)
        elif cfg.block == "resnet":
            self.trunk = nn.Sequential(
                nn.Linear(cfg.input_dim(), W),
                *[_ResBlock(W, cfg.activation, cfg.dropout) for _ in range(cfg.layers - 1)],
                nn.LayerNorm(W),
                _act(cfg.activation),
            )
        else:
            raise ValueError(f"unknown block {cfg.block!r}")
        self.out = nn.Linear(W, 2 * K)
        nn.init.zeros_(self.out.weight)  # start at the zero baseline
        nn.init.zeros_(self.out.bias)
        self.head: nn.Module | None = None
        if cfg.residual_head:
            H = cfg.head_width
            self.head = nn.Sequential(
                nn.Linear(5 + NUM_CTX, H),
                _act(cfg.activation),
                nn.Linear(H, H),
                _act(cfg.activation),
                nn.Linear(H, 1),
            )
            nn.init.zeros_(self.head[-1].weight)
            nn.init.zeros_(self.head[-1].bias)

    def forward(
        self,
        ranges: torch.Tensor,
        bucket: torch.Tensor,
        pct: torch.Tensor,
        onehot: torch.Tensor,
        c: torch.Tensor,
        stack: torch.Tensor,
        m: torch.Tensor | None = None,
        amp: bool = False,
    ) -> torch.Tensor:
        """``ranges [n, 2, 1326]`` normalised to sum 1 over the valid combos
        (:func:`normalise_ranges`), ``bucket``/``pct`` [n, 1326] and ``onehot``
        [n, 52] from the board features, ``c``/``stack`` [n]; ``m`` is
        :func:`opponent_mass` (computed when omitted). ``amp`` runs the MLPs in
        bf16 autocast; everything else is fp32. Returns ``ev [n, 2, 1326]`` fp32
        in pot units after the zero-sum layer, zero on invalid combos."""
        cfg = self.cfg
        K = cfg.buckets
        n = ranges.shape[0]
        dev = ranges.device
        r = ranges.float()
        valid = bucket < K
        with torch.autocast(dev.type, enabled=False):
            if m is None:
                m = opponent_mass(r, valid)
            idx = bucket[:, None, :].expand(n, 2, C)
            R = r.new_zeros(n, 2, K + 1).scatter_add_(2, idx, r)[..., :K] * K
            ctx = context_features(c.to(dev), stack.to(dev))
            parts = [R.reshape(n, 2 * K)]
            if cfg.count_input:
                cnt = r.new_zeros(n, K + 1).scatter_add_(1, bucket, valid.float())[:, :K]
                parts.append(cnt * (K / valid.sum(1, keepdim=True).clamp(min=1)))
            x = torch.cat([*parts, ctx, onehot.float()], 1)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
            vals = self.out(self.trunk(x))
        with torch.autocast(dev.type, enabled=False):
            vals = vals.float().view(n, 2, K)
            ev = vals.gather(2, bucket.clamp(max=K - 1)[:, None, :].expand(n, 2, C))
        if self.head is not None:
            ev = ev + self._head(ev, r, pct, valid, m, ctx, amp)
        with torch.autocast(dev.type, enabled=False):
            ev = zero_sum(ev.float(), r, m)
            return ev * valid[:, None, :]

    def _head(
        self,
        ev: torch.Tensor,
        r: torch.Tensor,
        pct: torch.Tensor,
        valid: torch.Tensor,
        m: torch.Tensor,
        ctx: torch.Tensor,
        amp: bool,
    ) -> torch.Tensor:
        n = r.shape[0]
        dev = r.device
        with torch.autocast(dev.type, enabled=False):
            vf = valid[:, None, :].float()
            nvalid = vf.sum(-1, keepdim=True)
            # opponent mass sharing a card with c, and the same weighted by its pct
            blocked = (1.0 - m) * vf
            opp = r.flip(1)
            sw = (opp * pct[:, None, :]).sum(-1, keepdim=True) - blocked_sum(opp * pct[:, None])
            sw = sw * vf
            feats = torch.stack(
                [
                    ev,
                    pct[:, None, :].expand(n, 2, C),
                    10.0 * blocked,
                    10.0 * sw,
                    r * nvalid,
                    *[ctx[:, i, None, None].expand(n, 2, C) for i in range(NUM_CTX)],
                ],
                -1,
            )
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
            out = self.head(feats)
        return out.float().squeeze(-1)


# --------------------------------------------------------------------------- checkpoints


def save_value_net(path: str | Path, net: RiverValueNet, meta: dict[str, Any] | None = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in net.state_dict().items()}
    torch.save({"config": net.cfg.to_dict(), "state_dict": state, "meta": meta or {}}, path)


def load_value_net(path: str | Path, device: torch.device | str = "cpu") -> RiverValueNet:
    """The saved net in eval mode on ``device``; its ``meta`` dict is ``net.meta``."""
    ck = torch.load(path, map_location="cpu", weights_only=True)
    net = RiverValueNet(ValueNetConfig.from_dict(ck["config"]))
    net.load_state_dict(ck["state_dict"])
    net.meta = ck.get("meta", {})
    return net.to(device).eval()


# --------------------------------------------------------------------------- inference


class ValueNetPredictor:
    """Batched inference for the solver: ``ev [n, 2, 1326]`` in pot units.

    Typical use, once per solve and then every CFR iteration::

        ids = pred.board_ids(boards_of_rows)     # [n] (one dict lookup per distinct board)
        ev = pred.predict_ids(ids, ranges, c, stack)

    or simply ``pred.predict(boards, ranges, c, stack)``.
    """

    def __init__(
        self,
        net: RiverValueNet,
        device: torch.device | str | None = None,
        cache: BoardFeatureCache | None = None,
    ):
        self.device = torch.device(device) if device is not None else next(net.parameters()).device
        self.net = net.to(self.device).eval()
        self.cache = cache or BoardFeatureCache(self.device)
        self.buckets = net.cfg.buckets

    @classmethod
    def from_path(cls, path: str | Path, device: torch.device | str = "cpu") -> ValueNetPredictor:
        return cls(load_value_net(path, device), device)

    def board_ids(self, boards: torch.Tensor) -> torch.Tensor:
        return self.cache.ids(boards)

    @torch.no_grad()
    def predict(
        self,
        boards: torch.Tensor,
        ranges: torch.Tensor,
        c: torch.Tensor,
        stack: torch.Tensor,
        chunk: int = 8192,
        bf16: bool | None = None,
    ) -> torch.Tensor:
        """``boards [n, 5]``, ``ranges [n, 2, 1326]`` (OOP, IP; any positive
        scale), ``c``/``stack`` [n] -> ``ev [n, 2, 1326]`` fp32 (0 on invalid)."""
        return self.predict_ids(self.board_ids(boards), ranges, c, stack, chunk, bf16)

    @torch.no_grad()
    def predict_ids(
        self,
        board_ids: torch.Tensor,
        ranges: torch.Tensor,
        c: torch.Tensor,
        stack: torch.Tensor,
        chunk: int = 8192,
        bf16: bool | None = None,
    ) -> torch.Tensor:
        """As :meth:`predict` with boards given as :meth:`board_ids` ids."""
        dev = self.device
        amp = (dev.type == "cuda") if bf16 is None else bool(bf16)
        n = ranges.shape[0]
        out = torch.empty(n, 2, C, dtype=torch.float32, device=dev)
        board_ids = board_ids.to(dev)
        c = torch.as_tensor(c, device=dev).reshape(n)
        stack = torch.as_tensor(stack, device=dev).reshape(n)
        for lo in range(0, n, chunk):
            hi = min(n, lo + chunk)
            f = self.cache.features(board_ids[lo:hi], self.buckets)
            r = normalise_ranges(ranges[lo:hi].to(dev), f["valid"])
            out[lo:hi] = self.net(
                r, f["bucket"], f["pct"], f["onehot"], c[lo:hi], stack[lo:hi], amp=amp
            )
        return out
