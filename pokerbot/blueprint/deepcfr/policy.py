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

Indexing: the net saved after iteration ``t`` has weight ``t``. The uniform
strategy of iteration 1 (no net yet) is not included.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch

from .checkpoint import list_checkpoints, load_net
from .networks import AdvantageNet, regret_matching


class SDCFRPolicy:
    def __init__(
        self,
        nets: Sequence[tuple[int, AdvantageNet]],
        last_n: int | None = None,
        reach_weighted: bool = True,
        fallback: str = "uniform",
        device: torch.device | str = "cpu",
    ) -> None:
        nets = sorted(nets, key=lambda x: x[0])
        if last_n:
            nets = nets[-int(last_n) :]
        if not nets:
            raise ValueError("SDCFRPolicy needs at least one network")
        self.device = torch.device(device)
        self.iterations = [t for t, _ in nets]
        self.nets = [net.to(self.device).eval() for _, net in nets]
        self.log_w = torch.log(torch.tensor(self.iterations, dtype=torch.float64))
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
        **kwargs,
    ) -> SDCFRPolicy:
        cks = list_checkpoints(path, player, max_iter)
        if last_n:
            cks = cks[-int(last_n) :]
        if not cks:
            raise FileNotFoundError(f"no checkpoints for player {player} under {path}")
        return cls([load_net(f, device) for _, f in cks], device=device, **kwargs)

    def __len__(self) -> int:
        return len(self.nets)

    # ------------------------------------------------------------ stateless
    @torch.no_grad()
    def net_policies(self, feats: dict[str, torch.Tensor]) -> torch.Tensor:
        """``[T, n, A]`` regret-matching policy of every net."""
        feats = {k: v.to(self.device) for k, v in feats.items()}
        legal = feats["legal"]
        return torch.stack(
            [regret_matching(net(feats).float(), legal, self.fallback) for net in self.nets]
        )

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
