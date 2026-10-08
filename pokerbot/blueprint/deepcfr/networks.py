"""Advantage network (DESIGN.md 5.5) and the regret-matching strategy head.

``AdvantageNet(cfg)(feats) -> [n, A]`` advantages, one per abstract action.
The legal mask is *not* applied inside the network; callers mask (see
:func:`regret_matching`). Inputs are the canonical feature dict of
:mod:`.features`.

Architecture:

* **Card branch.** Each of the 7 card slots gets ``rank_emb + suit_emb +
  slot_emb`` (plus ``card_emb``, one of 52, with ``card_embedding=True`` as
  in the Deep CFR paper; without it the sum per group cannot tell which rank
  carries which suit) (zeroed for undealt cards); the slots are summed per
  group (hole, flop, turn, river), the four group vectors concatenated and
  fed through a 2-layer MLP.
* **History branch.** Token embedding + position embedding + a linear
  projection of the chips-added amount, then a GRU (packed by length, the
  final hidden state is the summary), with ``hist_type="transformer"`` a
  small transformer encoder with a learned summary token, or with
  ``hist_type="mlp"`` a 2-layer MLP over the flattened fixed slots (the
  paper's bet encoding: no sequential loop, runs under bf16 autocast).
* **Scalars.** Pot/stack/bet fractions, street one-hot etc. (plus optional
  equity features) through one linear layer.
* **Trunk.** 3-layer MLP of width ``width`` (residual + LayerNorm after the
  first layer) and a linear output head.

The GRU always runs in fp32 (autocast disabled around it) and needs a host
sync for its packed lengths; everything else, including the ``"mlp"``
history branch, follows the caller's autocast (bf16 on CUDA in training and
traversal).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from ...env.cards import NO_CARD
from ...env.obs import NUM_SCALARS
from ...env.vec_env import HISTORY_LEN


@dataclass(frozen=True)
class NetConfig:
    num_actions: int = 6
    vocab_size: int = 49  # env.vocab_size = 1 + 8 * A
    history_len: int = HISTORY_LEN
    num_scalars: int = NUM_SCALARS
    card_dim: int = 64
    card_hidden: int = 768
    hist_type: str = "gru"  # "gru" | "transformer" | "mlp"
    # The GRU runs in fp32 once per token: it is the most expensive part of a
    # forward pass per row, so it is kept narrower than the MLPs.
    hist_dim: int = 128
    hist_hidden: int = 256
    hist_layers: int = 1  # GRU layers or transformer layers
    tf_heads: int = 4
    scalar_hidden: int = 64
    width: int = 512
    trunk_layers: int = 3
    card_embedding: bool = False  # add a per-card (52-way) embedding (Deep CFR)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> NetConfig:
        return NetConfig(**d)


class CardBranch(nn.Module):
    # slot -> group: hole (0, 1), flop (2, 3, 4), turn (5), river (6)
    GROUPS = (0, 0, 1, 1, 1, 2, 3)

    def __init__(self, dim: int, hidden: int, card_embedding: bool = False) -> None:
        super().__init__()
        self.rank = nn.Embedding(14, dim)  # 13 = undealt
        self.suit = nn.Embedding(5, dim)  # 4 = undealt
        self.slot = nn.Embedding(7, dim)
        self.card = nn.Embedding(NO_CARD + 1, dim) if card_embedding else None
        self.register_buffer("group", torch.tensor(self.GROUPS), persistent=False)
        self.mlp = nn.Sequential(
            nn.Linear(4 * dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU()
        )
        self.dim = dim

    def forward(self, cards: torch.Tensor, card_mask: torch.Tensor) -> torch.Tensor:
        dealt = card_mask & (cards < NO_CARD)
        c = torch.where(dealt, cards, 0)
        rank = torch.where(dealt, torch.div(c, 4, rounding_mode="floor"), 13)
        suit = torch.where(dealt, c % 4, 4)
        slots = torch.arange(7, device=cards.device)
        e = self.rank(rank) + self.suit(suit) + self.slot(slots)[None]
        if self.card is not None:
            e = e + self.card(torch.where(dealt, c, NO_CARD))
        e = e * dealt[..., None].to(e.dtype)  # [n, 7, d]
        n = cards.shape[0]
        g = e.new_zeros(n, 4, self.dim).index_add_(1, self.group, e)
        return self.mlp(g.reshape(n, 4 * self.dim))


class HistoryBranch(nn.Module):
    def __init__(self, cfg: NetConfig) -> None:
        super().__init__()
        d = cfg.hist_dim
        self.kind = cfg.hist_type
        self.tok = nn.Embedding(cfg.vocab_size, d, padding_idx=0)
        self.pos = nn.Embedding(cfg.history_len + 1, d)
        self.amt = nn.Linear(1, d)
        if self.kind == "gru":
            self.rnn = nn.GRU(d, cfg.hist_hidden, num_layers=cfg.hist_layers, batch_first=True)
            self.out = nn.Identity()
        elif self.kind == "transformer":
            layer = nn.TransformerEncoderLayer(
                d, cfg.tf_heads, dim_feedforward=2 * d, dropout=0.0, batch_first=True
            )
            self.tf = nn.TransformerEncoder(layer, cfg.hist_layers, enable_nested_tensor=False)
            self.cls = nn.Parameter(torch.zeros(1, 1, d))
            self.out = nn.Sequential(nn.Linear(d, cfg.hist_hidden), nn.ReLU())
        elif self.kind == "mlp":
            self.mlp = nn.Sequential(
                nn.Linear(cfg.history_len * d, cfg.hist_hidden),
                nn.ReLU(),
                nn.Linear(cfg.hist_hidden, cfg.hist_hidden),
                nn.ReLU(),
            )
        else:
            raise ValueError(f"unknown hist_type {self.kind!r}")

    def forward(self, hist: torch.Tensor, hist_amt: torch.Tensor) -> torch.Tensor:
        n, T = hist.shape
        mask = hist != 0
        pos = torch.arange(T, device=hist.device)
        x = self.tok(hist) + self.pos(pos + 1)[None] + self.amt(hist_amt[..., None].float())
        x = x * mask[..., None].to(x.dtype)
        if self.kind == "mlp":
            return self.mlp(x.reshape(n, T * x.shape[-1]))
        if self.kind == "gru":
            lengths = mask.sum(1)
            with torch.autocast(device_type=hist.device.type, enabled=False):
                # packed: final hidden state at each row's true length. Length-0
                # rows (no action yet) run one padding step and are zeroed.
                packed = pack_padded_sequence(
                    x.float(), lengths.clamp(min=1).cpu(), batch_first=True, enforce_sorted=False
                )
                _, h = self.rnn(packed)
            out = h[-1] * (lengths > 0)[:, None].to(h.dtype)
            return self.out(out)
        cls = self.cls.expand(n, 1, -1).to(x.dtype)
        seq = torch.cat([cls, x], 1)
        pad = torch.cat([torch.zeros(n, 1, dtype=torch.bool, device=hist.device), ~mask], 1)
        y = self.tf(seq, src_key_padding_mask=pad)
        return self.out(y[:, 0])


class AdvantageNet(nn.Module):
    def __init__(self, cfg: NetConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.cards = CardBranch(cfg.card_dim, cfg.card_hidden, cfg.card_embedding)
        self.history = HistoryBranch(cfg)
        self.scalars = nn.Sequential(nn.Linear(cfg.num_scalars, cfg.scalar_hidden), nn.ReLU())
        w = cfg.width
        self.inp = nn.Linear(cfg.card_hidden + cfg.hist_hidden + cfg.scalar_hidden, w)
        self.hidden = nn.ModuleList(nn.Linear(w, w) for _ in range(cfg.trunk_layers - 1))
        self.norms = nn.ModuleList(nn.LayerNorm(w) for _ in range(cfg.trunk_layers - 1))
        self.head = nn.Linear(w, cfg.num_actions)

    def forward(self, feats: dict[str, torch.Tensor]) -> torch.Tensor:
        c = self.cards(feats["cards"], feats["card_mask"])
        h = self.history(feats["hist"], feats["hist_amt"])
        s = self.scalars(feats["scalars"].float())
        x = torch.relu(self.inp(torch.cat([c, h.to(c.dtype), s.to(c.dtype)], 1)))
        for lin, norm in zip(self.hidden, self.norms, strict=True):
            x = norm(x + torch.relu(lin(x)))
        return self.head(x)


def num_params(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters())


class StackedAdvantageNets:
    """``T`` advantage nets of one ``hist_type="mlp"`` architecture evaluated
    together: ``nets(feats) -> [T, n, A]``, equal to stacking each net's
    output (up to float rounding).

    Evaluating many nets one by one is bound by kernel launches (hundreds per
    net). Here every layer runs once for all nets as a batched matmul over
    stacked weights. Two shortcuts use the structure of range queries, where
    all rows share one public state and differ only in the hole cards: the
    history branch is evaluated once when every row has the same history,
    and the board slots of the card branch are summed once when every row has
    the same board (only the hole-card group is per row).
    """

    def __init__(self, nets: list[AdvantageNet]) -> None:
        cfg = nets[0].cfg
        if cfg.hist_type != "mlp":
            raise ValueError("StackedAdvantageNets supports hist_type='mlp' only")
        if any(n.cfg != cfg for n in nets):
            raise ValueError("all nets must share one configuration")
        self.cfg = cfg
        self.T = len(nets)
        self.card_embedding = cfg.card_embedding

        def st(f):  # stack one parameter (detached) over the nets
            return torch.stack([f(n).detach() for n in nets])

        self.rank = st(lambda n: n.cards.rank.weight)  # [T, 14, d]
        self.suit = st(lambda n: n.cards.suit.weight)  # [T, 5, d]
        self.slot = st(lambda n: n.cards.slot.weight)  # [T, 7, d]
        self.card = st(lambda n: n.cards.card.weight) if cfg.card_embedding else None
        self.cw1 = st(lambda n: n.cards.mlp[0].weight.T)  # [T, 4d, H]
        self.cb1 = st(lambda n: n.cards.mlp[0].bias)[:, None]
        self.cw2 = st(lambda n: n.cards.mlp[2].weight.T)
        self.cb2 = st(lambda n: n.cards.mlp[2].bias)[:, None]
        self.tok = st(lambda n: n.history.tok.weight)  # [T, V, d]
        self.pos = st(lambda n: n.history.pos.weight[1 : cfg.history_len + 1])  # [T, L, d]
        self.amt_w = st(lambda n: n.history.amt.weight[:, 0])  # [T, d]
        self.amt_b = st(lambda n: n.history.amt.bias)
        self.hw1 = st(lambda n: n.history.mlp[0].weight.T)  # [T, L*d, hh]
        self.hb1 = st(lambda n: n.history.mlp[0].bias)[:, None]
        self.hw2 = st(lambda n: n.history.mlp[2].weight.T)
        self.hb2 = st(lambda n: n.history.mlp[2].bias)[:, None]
        self.sw = st(lambda n: n.scalars[0].weight.T)
        self.sb = st(lambda n: n.scalars[0].bias)[:, None]
        self.iw = st(lambda n: n.inp.weight.T)
        self.ib = st(lambda n: n.inp.bias)[:, None]
        self.hidden = [
            (
                st(lambda n, j=j: n.hidden[j].weight.T),
                st(lambda n, j=j: n.hidden[j].bias)[:, None],
                st(lambda n, j=j: n.norms[j].weight)[:, None],
                st(lambda n, j=j: n.norms[j].bias)[:, None],
            )
            for j in range(cfg.trunk_layers - 1)
        ]
        self.eps = nets[0].norms[0].eps if len(nets[0].norms) else 1e-5
        self.ow = st(lambda n: n.head.weight.T)
        self.ob = st(lambda n: n.head.bias)[:, None]

    def _slot_embeddings(
        self, cards: torch.Tensor, dealt: torch.Tensor, slots: slice
    ) -> torch.Tensor:
        """``[T, n, k, d]`` embeddings of the card slots ``slots`` (zero when undealt)."""
        c = torch.where(dealt, cards, 0).long()
        rank = torch.where(dealt, torch.div(c, 4, rounding_mode="floor"), 13)
        suit = torch.where(dealt, c % 4, 4)
        e = self.rank[:, rank] + self.suit[:, suit] + self.slot[:, slots][:, None]
        if self.card is not None:
            e = e + self.card[:, torch.where(dealt, c, NO_CARD)]
        return e * dealt[None, ..., None].to(e.dtype)

    def _cards(self, cards: torch.Tensor, card_mask: torch.Tensor) -> torch.Tensor:
        n = cards.shape[0]
        d = self.rank.shape[-1]
        dealt = card_mask & (cards < NO_CARD)
        shared_board = n > 1 and bool(
            (cards[:, 2:] == cards[:1, 2:]).all() and (dealt[:, 2:] == dealt[:1, 2:]).all()
        )
        if shared_board:
            hole = self._slot_embeddings(cards[:, :2], dealt[:, :2], slice(0, 2)).sum(
                2
            )  # [T, n, d]
            b = self._slot_embeddings(cards[:1, 2:], dealt[:1, 2:], slice(2, 7))  # [T, 1, 5, d]
            board = torch.cat([b[:, :, :3].sum(2), b[:, :, 3], b[:, :, 4]], 2)  # flop, turn, river
            x = torch.baddbmm(self.cb1, hole, self.cw1[:, :d])  # [T, n, H]
            x = x + torch.bmm(board, self.cw1[:, d:])  # the board's part, once
        else:
            e = self._slot_embeddings(cards, dealt, slice(0, 7))  # [T, n, 7, d]
            g = torch.cat([e[:, :, :2].sum(2), e[:, :, 2:5].sum(2), e[:, :, 5], e[:, :, 6]], 2)
            x = torch.baddbmm(self.cb1, g, self.cw1)
        return torch.relu(torch.baddbmm(self.cb2, torch.relu(x), self.cw2))

    def _history(self, hist: torch.Tensor, amt: torch.Tensor) -> torch.Tensor:
        n, L = hist.shape
        shared = n > 1 and bool((hist == hist[:1]).all() and (amt == amt[:1]).all())
        if shared:
            hist, amt = hist[:1], amt[:1]
        mask = (hist != 0).to(self.tok.dtype)
        x = (
            self.tok[:, hist.long()]
            + self.pos[:, None]
            + amt.float()[None, ..., None] * self.amt_w[:, None, None]
        )
        x = (x + self.amt_b[:, None, None]) * mask[None, ..., None]  # [T, m, L, d]
        x = x.reshape(self.T, x.shape[1], -1)
        h = torch.relu(
            torch.baddbmm(self.hb2, torch.relu(torch.baddbmm(self.hb1, x, self.hw1)), self.hw2)
        )
        return h.expand(self.T, n, h.shape[-1]) if shared else h

    @torch.no_grad()
    def __call__(self, feats: dict[str, torch.Tensor]) -> torch.Tensor:
        n = feats["cards"].shape[0]
        c = self._cards(feats["cards"], feats["card_mask"])
        h = self._history(feats["hist"], feats["hist_amt"])
        s = feats["scalars"].float()[None].expand(self.T, n, -1)
        s = torch.relu(torch.baddbmm(self.sb, s, self.sw))
        z = torch.relu(
            torch.baddbmm(self.ib, torch.cat([c, h.to(c.dtype), s.to(c.dtype)], 2), self.iw)
        )
        for w, b, g, beta in self.hidden:
            y = z + torch.relu(torch.baddbmm(b, z, w))
            mu = y.mean(-1, keepdim=True)
            var = y.var(-1, unbiased=False, keepdim=True)
            z = (y - mu) * torch.rsqrt(var + self.eps) * g + beta
        return torch.baddbmm(self.ob, z, self.ow)


def regret_matching(
    adv: torch.Tensor, legal: torch.Tensor, fallback: str = "uniform"
) -> torch.Tensor:
    """Regret-matching policy ``[n, A]`` from advantages and a legal mask.

    Positive parts of the legal advantages, normalized. When no legal action
    has a positive advantage: uniform over the legal actions (``"uniform"``,
    the default) or all mass on the best legal action (``"argmax"``, the
    Deep CFR paper's choice). Illegal actions always get 0.
    """
    legal = legal.bool()
    lf = legal.float()
    pos = adv.float().clamp(min=0) * lf
    s = pos.sum(1, keepdim=True)
    if fallback == "uniform":
        fb = lf / lf.sum(1, keepdim=True).clamp(min=1)
    elif fallback == "argmax":
        best = adv.float().masked_fill(~legal, float("-inf")).argmax(1)
        fb = torch.nn.functional.one_hot(best, adv.shape[1]).float()
    else:
        raise ValueError(f"unknown fallback {fallback!r}")
    return torch.where(s > 0, pos / s.clamp(min=1e-30), fb)


class StrategyHead(nn.Module):
    """Turns advantages into a regret-matching policy (see :func:`regret_matching`)."""

    def __init__(self, fallback: str = "uniform") -> None:
        super().__init__()
        self.fallback = fallback

    def forward(self, adv: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
        return regret_matching(adv, legal, self.fallback)
