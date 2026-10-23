"""Tabular preflop regrets: 169 lossless hand classes x the abstract preflop nodes.

Preflop has few information sets (the default 100bb abstraction has 92
decision nodes, 15,548 infosets with 169 classes) and they are the ones the
advantage network fits worst: on the 100bb run it explained about 2% of the
preflop regret-target variance, and its raise outputs at the root were
uncorrelated with the class means it was regressing onto. A table has no
approximation error and keeps every sample, where the reservoir keeps a
shrinking fraction of them.

* :class:`PreflopTree` enumerates the abstract preflop decision nodes of a
  game and action spec by stepping a :class:`~pokerbot.env.VecNLHE`, and maps
  history tokens to node ids.
* :class:`PreflopRegrets` accumulates linear-CFR weighted regret samples,
  ``R[node, class, a] += t * r``, and returns the regret-matching strategy
  table (uniform over the legal actions when no regret is positive).
* :class:`TablePolicy` wraps a policy: rows at a known preflop node play the
  table, every other row the wrapped (network) policy.

The traversal keeps recording preflop samples, so the advantage net still
learns preflop; it answers for histories the table does not know (off-tree
opponent raises mapped onto token sequences the abstract tree never reaches).
The strategy table of every iteration is saved in that iteration's
checkpoint, so :class:`~.policy.SDCFRPolicy` averages it like the nets.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch

from ...env.obs import SCALAR_NAMES
from ...env.vec_env import VecNLHE
from .features import index_features

NUM_CLASSES = 169
_STREET_COL = SCALAR_NAMES.index("preflop")
_BASE_BITS = 6  # history tokens < 64 (vocab = 1 + 8 * A with A <= 7)
_MAX_TOKENS = 10  # 10 * 6 = 60 bits of an int64 key; preflop lines are shorter


def hand_class(hole: torch.Tensor) -> torch.Tensor:
    """``[n]`` lossless preflop class (0..168) of ``hole [n, 2]``: pairs on the
    diagonal of the 13 x 13 rank grid, suited hands below it, offsuit above."""
    r, s = torch.div(hole, 4, rounding_mode="floor"), hole % 4
    hi = torch.maximum(r[:, 0], r[:, 1])
    lo = torch.minimum(r[:, 0], r[:, 1])
    suited = s[:, 0] == s[:, 1]
    return torch.where(suited & (hi != lo), hi * 13 + lo, lo * 13 + hi)


def history_keys(hist: torch.Tensor) -> torch.Tensor:
    """``[n]`` int64 key of each row's history tokens (-1 when longer than any
    preflop line can be)."""
    hist = hist.long()
    L = min(hist.shape[1], _MAX_TOKENS)
    shift = torch.arange(L, device=hist.device) * _BASE_BITS
    key = (hist[:, :L] << shift).sum(1)
    if hist.shape[1] > L:
        key = torch.where((hist[:, L:] != 0).any(1), -1, key)
    return key


class PreflopTree:
    """The abstract preflop decision nodes: node ``i`` is the betting line
    ``tokens[i]`` (env history tokens) with legal mask ``legal[i]``."""

    def __init__(
        self,
        tokens: Sequence[Sequence[int]],
        legal: Sequence[Sequence[bool]],
        device: torch.device | str = "cpu",
    ) -> None:
        self.tokens = [list(map(int, t)) for t in tokens]
        if any(len(t) > _MAX_TOKENS or max(t, default=0) >= 1 << _BASE_BITS for t in self.tokens):
            raise ValueError("preflop line too long or token out of range for the key")
        self.device = torch.device(device)
        self.legal = torch.tensor(legal, dtype=torch.bool, device=self.device)
        hist = torch.zeros(len(self.tokens), _MAX_TOKENS, dtype=torch.long)
        for i, t in enumerate(self.tokens):
            hist[i, : len(t)] = torch.tensor(t, dtype=torch.long)
        keys = history_keys(hist)
        order = torch.argsort(keys)
        self._keys = keys[order].to(self.device)
        self._node = order.to(self.device)

    def __len__(self) -> int:
        return len(self.tokens)

    @property
    def num_actions(self) -> int:
        return int(self.legal.shape[1])

    @classmethod
    def build(cls, game_config: Any, spec: Any, device: torch.device | str = "cpu") -> PreflopTree:
        """Enumerate by stepping every legal abstract action from the deal until
        the flop or the end of the hand (the betting structure does not depend
        on the cards)."""
        env = VecNLHE(1, game_config, "cpu", 0, spec, validate=False)
        tokens: list[list[int]] = []
        legal: list[list[bool]] = []
        while env.n > 0:
            mask = env.legal_mask()
            for i in range(env.n):
                tokens.append(env.hist_tok[i, : int(env.hist_len[i])].tolist())
                legal.append(mask[i].tolist())
            r, a = mask.nonzero(as_tuple=True)
            child = env.select(r)
            child.step(a)
            keep = (~child.done & (child.street == 0)).nonzero().squeeze(1)
            env = child.select(keep)
        return cls(tokens, legal, device)

    def to_meta(self) -> dict[str, Any]:
        return {"tokens": self.tokens, "legal": self.legal.cpu().tolist()}

    @classmethod
    def from_meta(cls, meta: dict[str, Any], device: torch.device | str = "cpu") -> PreflopTree:
        return cls(meta["tokens"], meta["legal"], device)

    def to(self, device: torch.device | str) -> PreflopTree:
        return PreflopTree(self.tokens, self.legal.cpu().tolist(), device)

    def node_index(self, hist: torch.Tensor) -> torch.Tensor:
        """``[n]`` node id of each row's history, -1 when it is not a node."""
        key = history_keys(hist.to(self.device))
        pos = torch.searchsorted(self._keys, key).clamp(max=len(self._keys) - 1)
        hit = (self._keys[pos] == key) & (key >= 0)
        return torch.where(hit, self._node[pos], -1)

    def locate(self, feats: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Rows of ``feats`` at a known preflop node: ``(rows, node * 169 + class)``."""
        street0 = feats["scalars"][:, _STREET_COL].to(self.device) > 0.5
        node = self.node_index(feats["hist"])
        rows = (street0 & (node >= 0)).nonzero().squeeze(1)
        cls = hand_class(feats["cards"][:, :2].to(self.device)[rows].long())
        return rows, node[rows] * NUM_CLASSES + cls


def table_probs(strategy: torch.Tensor, cells: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
    """``[..., m, A]`` probabilities of the flat ``cells`` of ``strategy
    [..., nodes * 169, A]`` restricted to ``legal [m, A]`` (uniform over it when
    the table puts no mass there)."""
    p = strategy[..., cells, :] * legal
    s = p.sum(-1, keepdim=True)
    uni = legal / legal.sum(-1, keepdim=True).clamp(min=1)
    return torch.where(s > 0, p / s.clamp(min=1e-30), uni.expand_as(p))


class PreflopRegrets:
    """Cumulative linear-CFR regrets of one seat's preflop infosets."""

    def __init__(self, tree: PreflopTree, device: torch.device | str = "cpu") -> None:
        self.tree = tree.to(device)
        self.device = torch.device(device)
        A = tree.num_actions
        self.regret = torch.zeros(len(tree) * NUM_CLASSES, A, dtype=torch.float64, device=device)
        self._legal = self.tree.legal.repeat_interleave(NUM_CLASSES, 0)  # [cells, A]

    def pending(self) -> torch.Tensor:
        return torch.zeros_like(self.regret)

    def add_samples(
        self, acc: torch.Tensor, samples: dict[str, torch.Tensor], weight: float
    ) -> int:
        """Add ``weight * target`` of the preflop rows of ``samples`` (traversal
        rows: features and regret targets) into ``acc``. Returns the count."""
        rows, cells = self.tree.locate(samples)
        if rows.numel():
            target = samples["target"].to(self.device)[rows].double()
            acc.index_add_(0, cells, target * float(weight))
        return int(rows.numel())

    def strategy(self) -> torch.Tensor:
        """``[nodes * 169, A]`` float32 regret-matching strategy."""
        pos = self.regret.clamp(min=0) * self._legal
        s = pos.sum(1, keepdim=True)
        uni = self._legal / self._legal.sum(1, keepdim=True).clamp(min=1)
        return torch.where(s > 0, pos / s.clamp(min=1e-300), uni).float()

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"regret": self.regret.cpu()}

    def load_state_dict(self, d: dict[str, torch.Tensor]) -> None:
        self.regret.copy_(d["regret"].to(self.device))


PolicyFn = Callable[[dict[str, torch.Tensor]], torch.Tensor]


class TablePolicy:
    """Rows at a known preflop node play ``strategy``, the rest ``base``."""

    def __init__(self, base: PolicyFn, tree: PreflopTree, strategy: torch.Tensor) -> None:
        self.base = base
        self.tree = tree
        self.strategy = strategy

    @torch.no_grad()
    def __call__(self, feats: dict[str, torch.Tensor]) -> torch.Tensor:
        legal = feats["legal"]
        n = legal.shape[0]
        rows, cells = self.tree.locate(feats)
        out = torch.zeros(n, legal.shape[1], dtype=torch.float32, device=legal.device)
        other = torch.ones(n, dtype=torch.bool, device=legal.device)
        other[rows.to(legal.device)] = False
        o = other.nonzero().squeeze(1)
        if o.numel():
            out[o] = self.base(index_features(feats, o)).float()
        if rows.numel():
            lg = legal.to(self.tree.device)[rows].float()
            p = table_probs(self.strategy.to(self.tree.device), cells, lg)
            out[rows.to(legal.device)] = p.to(legal.device)
        return out
