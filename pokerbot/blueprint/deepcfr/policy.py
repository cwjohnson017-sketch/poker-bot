"""SD-CFR average strategy from the sequence of advantage networks.

Single Deep CFR (Steinberger, 2019) never trains an average-policy network.
The average strategy of CFR is

    avg(I)(a) = sum_t w_t * pi_t(I) * sigma_t(I)(a) / sum_t w_t * pi_t(I)

where ``sigma_t`` is the regret-matching policy of the advantage net of
iteration ``t``, ``pi_t(I)`` the player's *own* reach probability of ``I``
under ``sigma_t`` (the product of ``sigma_t`` over the player's earlier
actions in the hand) and ``w_t = t`` (linear CFR). :class:`SDCFRPolicy`
computes this exactly at play time: at each own decision it evaluates every
net, mixes with the current weights, and after the action is chosen it
multiplies each net's reach by that net's probability of the action.

``reach_weighted=False`` drops ``pi_t`` (a plain iteration-weighted mixture
of the per-iteration policies, cheaper to reason about but not the CFR
average). ``last_n`` keeps only the most recent nets for speed.

Playing one hand with a single net drawn with probability ``w_t / sum w``
(:meth:`SDCFRPolicy.sample_index`, :meth:`SDCFRPolicy.net_probs`) realizes
the reach-weighted average exactly: the reach weights above are the
posterior over the drawn net given the player's own actions. It costs one
forward pass per decision instead of one per net.

Indexing: the net saved after iteration ``t`` has weight ``t``. The uniform
strategy of iteration 1 (no net yet) is not included.

With tabular preflop regrets (``preflop``: the :class:`~.preflop.PreflopTree`
and each iteration's strategy table, saved in its checkpoint), rows at a
known preflop node take iteration ``t``'s probabilities from its table
instead of its net, in :meth:`SDCFRPolicy.net_policies` and
:meth:`SDCFRPolicy.net_probs`; the reach weights and the average follow.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from .checkpoint import list_checkpoints, load_checkpoint, read_meta
from .networks import AdvantageNet, StackedAdvantageNets, regret_matching
from .preflop import PreflopTree, table_probs


class SDCFRPolicy:
    def __init__(
        self,
        nets: Sequence[tuple[int, AdvantageNet]],
        last_n: int | None = None,
        reach_weighted: bool = True,
        fallback: str = "uniform",
        device: torch.device | str = "cpu",
        preflop: tuple[PreflopTree, dict[int, torch.Tensor]] | None = None,
        policy_head: str = "regret",
    ) -> None:
        if policy_head not in ("regret", "softmax"):
            raise ValueError(f"policy_head must be 'regret' or 'softmax', got {policy_head!r}")
        # "regret": the nets output advantages (regret matching); "softmax": they
        # output policy logits (a distilled average-strategy net)
        self.policy_head = policy_head
        nets = sorted(nets, key=lambda x: x[0])
        if last_n:
            nets = nets[-int(last_n) :]
        if not nets:
            raise ValueError("SDCFRPolicy needs at least one network")
        self.device = torch.device(device)
        self.iterations = [t for t, _ in nets]
        self.nets = [net.to(self.device).eval() for _, net in nets]
        self.preflop_tree: PreflopTree | None = None
        self.preflop_tables: torch.Tensor | None = None  # [T, nodes * 169, A]
        if preflop is not None:
            tree, tables = preflop
            self.preflop_tree = tree.to(self.device)
            self.preflop_tables = torch.stack(
                [tables[t].to(self.device, torch.float32) for t in self.iterations]
            )
        self.log_w = torch.log(torch.tensor(self.iterations, dtype=torch.float64))
        # all nets share one architecture: evaluate them in one vmapped call
        # (one kernel per layer instead of one per layer and net)
        self.batched = len(self.nets) > 1 and all(isinstance(n, AdvantageNet) for n in self.nets)
        self._stack: StackedAdvantageNets | None = None
        self.reach_weighted = reach_weighted
        self.fallback = fallback
        self.new_hand()

    @classmethod
    def from_dir(
        cls,
        path: str | Path,
        player: int,
        last_n: int | None = None,
        max_iter: int | None = None,
        device: torch.device | str = "cpu",
        stride: int | None = None,
        **kwargs,
    ) -> SDCFRPolicy:
        cks = list_checkpoints(path, player, max_iter)
        if stride and int(stride) > 1 and cks:
            # every stride-th iteration plus the newest: with linear iteration
            # weights this keeps the shape of the full average at a fraction of the nets
            newest = cks[-1][0]
            cks = [(t, f) for t, f in cks if t % int(stride) == 0 or t == newest]
        if last_n:
            cks = cks[-int(last_n) :]
        if not cks:
            raise FileNotFoundError(f"no checkpoints for player {player} under {path}")
        loaded = [load_checkpoint(f, device) for _, f in cks]
        preflop = None
        meta = read_meta(path)
        if meta.get("preflop") and all(pre is not None for _, _, pre in loaded):
            tree = PreflopTree.from_meta(meta["preflop"], device)
            preflop = (tree, {t: pre for t, _, pre in loaded})
        nets = [(t, net) for t, net, _ in loaded]
        return cls(nets, device=device, preflop=preflop, **kwargs)

    def __len__(self) -> int:
        return len(self.nets)

    # ------------------------------------------------------------ stateless
    MAX_STACKED_ROWS = 1 << 17  # nets x rows per stacked call (bounds activation memory)

    def _advantages_all(self, feats: dict[str, torch.Tensor]) -> torch.Tensor:
        """``[T, n, A]`` advantages of every net (stacked evaluation when the
        architecture allows it, else one net at a time)."""
        if self.batched and self._stack is None:
            try:
                self._stack = StackedAdvantageNets(self.nets)
            except ValueError:  # a GRU / transformer history branch
                self.batched = False
        if not self.batched:
            return torch.stack([net(feats) for net in self.nets])
        n = feats["legal"].shape[0]
        step = max(1, self.MAX_STACKED_ROWS // len(self.nets))
        if n <= step:
            return self._stack(feats)
        return torch.cat(
            [
                self._stack({k: v[lo : lo + step] for k, v in feats.items()})
                for lo in range(0, n, step)
            ],
            1,
        )

    @torch.no_grad()
    def net_policies(self, feats: dict[str, torch.Tensor]) -> torch.Tensor:
        """``[T, n, A]`` policy of every iteration (its net, or its preflop table)."""
        feats = {k: v.to(self.device) for k, v in feats.items()}
        legal = feats["legal"]
        adv = self._advantages_all(feats).float()
        T, n, A = adv.shape
        if self.policy_head == "softmax":
            P = torch.softmax(adv.masked_fill(~legal.bool()[None], float("-inf")), -1)
        else:
            P = regret_matching(
                adv.reshape(T * n, A), legal[None].expand(T, n, A).reshape(T * n, A), self.fallback
            ).reshape(T, n, A)
        return self._with_tables(P, feats, None)

    def _with_tables(
        self, P: torch.Tensor, feats: dict[str, torch.Tensor], index: int | None
    ) -> torch.Tensor:
        """Overwrite the rows of ``P`` at known preflop nodes with the tables."""
        if self.preflop_tree is None or self.preflop_tables is None:
            return P
        rows, cells = self.preflop_tree.locate(feats)
        if rows.numel() == 0:
            return P
        legal = feats["legal"][rows].float()
        if index is None:
            P[:, rows] = table_probs(self.preflop_tables, cells, legal)
        else:
            P[rows] = table_probs(self.preflop_tables[index], cells, legal)
        return P

    @torch.no_grad()
    def average(
        self, feats: dict[str, torch.Tensor], log_reach: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Average policy ``[n, A]`` given per-net own-reach logs ``[T, n]``
        (None = the root / no reach weighting). Also returns ``[T, n, A]``."""
        P = self.net_policies(feats)
        T, n, _ = P.shape
        logw = self.log_w.to(P.device)[:, None].expand(T, n)
        if log_reach is not None and self.reach_weighted:
            lw = logw + log_reach.to(P.device, torch.float64)
            # all reaches zero (cannot happen on-policy): fall back to t-weights
            lw = torch.where(torch.isfinite(lw).any(0, keepdim=True), lw, logw)
            logw = lw
        w = torch.softmax(logw, 0).float()  # [T, n]
        avg = (w[..., None] * P).sum(0)
        return avg, P

    # ------------------------------------------------------------ one hand
    def new_hand(self) -> None:
        self.log_reach = torch.zeros(len(self.nets), 1, dtype=torch.float64)
        self._last: torch.Tensor | None = None

    def act_probs(self, feats: dict[str, torch.Tensor]) -> torch.Tensor:
        """Average policy ``[1, A]`` at this hand's current own decision."""
        avg, P = self.average(feats, self.log_reach)
        self._last = P
        return avg

    def observe(self, action: int) -> None:
        """Record the own action taken at the last :meth:`act_probs` decision."""
        if self._last is None:
            return
        p = self._last[:, 0, int(action)].double().cpu()
        self.log_reach = self.log_reach + torch.log(p.clamp(min=0))[:, None]
        self._last = None

    # ------------------------------------------------------------ one net per hand
    def sample_index(self, rng: np.random.Generator) -> int:
        """A net drawn with probability proportional to its iteration weight
        (play a whole hand with it; see the module docstring)."""
        w = torch.softmax(self.log_w, 0).numpy()
        return int(rng.choice(len(w), p=w / w.sum()))

    @torch.no_grad()
    def net_probs(self, feats: dict[str, torch.Tensor], index: int) -> torch.Tensor:
        """``[n, A]`` policy of net ``index`` alone."""
        feats = {k: v.to(self.device) for k, v in feats.items()}
        out = self.nets[index](feats).float()
        if self.policy_head == "softmax":
            P = torch.softmax(out.masked_fill(~feats["legal"].bool(), float("-inf")), -1)
        else:
            P = regret_matching(out, feats["legal"], self.fallback)
        return self._with_tables(P, feats, index)
