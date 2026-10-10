"""Network input features shared by traversal, training, memories and play.

The canonical feature dict (every tensor has a leading batch dim ``n``):

* ``cards [n, 7]`` long: own hole cards, then the board; 52 = undealt.
* ``card_mask [n, 7]`` bool: visible cards.
* ``hist [n, T]`` long: env history tokens (0 = padding, left-aligned).
* ``hist_amt [n, T]`` float: chips each action put in / starting stack.
* ``scalars [n, S]`` float: ``obs["scalars"]`` (14 values) followed, when
  enabled, by ``equity`` (1) and ``equity_hist`` (``hist_bins``), then by the
  11 table-lookup strength columns of :mod:`.strength`
  (``strength_tables``).
* ``legal [n, A]`` bool.

``hist_mask`` is always ``hist != 0`` and is recomputed by the network, so
it is neither stored nor passed around.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch

from ...env.obs import NUM_SCALARS
from ...env.vec_env import HISTORY_LEN
from .strength import NUM_STRENGTH

FEATURE_KEYS = ("cards", "card_mask", "hist", "hist_amt", "scalars", "legal")


@dataclass(frozen=True)
class FeatureConfig:
    """Which optional equity features the networks see (off by default).

    ``equity_samples > 0`` appends the Monte Carlo equity vs a random hand;
    ``hist_runouts > 0`` appends a ``hist_bins``-bin river-equity histogram
    over that many sampled runouts (``hist_opp_samples`` opponents per
    runout, 0 = exact against all 990). ``strength_tables`` (a bucket-build
    directory) appends 11 hand-strength columns looked up per canonical hand
    class (:mod:`.strength`).
    """

    equity_samples: int = 0
    hist_runouts: int = 0
    hist_bins: int = 10
    hist_opp_samples: int = 0
    history_len: int = HISTORY_LEN
    strength_tables: str | None = None

    @property
    def num_scalars(self) -> int:
        n = NUM_SCALARS
        if self.equity_samples > 0:
            n += 1
        if self.hist_runouts > 0:
            n += self.hist_bins
        if self.strength_tables:
            n += NUM_STRENGTH
        return n

    def obs_kwargs(self) -> dict[str, int]:
        return {
            "equity_samples": self.equity_samples,
            "hist_runouts": self.hist_runouts,
            "hist_bins": self.hist_bins,
            "hist_opp_samples": self.hist_opp_samples,
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any] | None) -> FeatureConfig:
        return FeatureConfig(**(d or {}))


def features_from_obs(obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Canonical feature dict from ``env.obs(...)`` (equity columns appended)."""
    scal = [obs["scalars"].float()]
    if "equity" in obs:
        scal.append(obs["equity"].float().reshape(-1, 1))
    if "equity_hist" in obs:
        scal.append(obs["equity_hist"].float())
    return {
        "cards": obs["cards"].long(),
        "card_mask": obs["card_mask"].bool(),
        "hist": obs["hist"].long(),
        "hist_amt": obs["hist_amt"].float(),
        "scalars": torch.cat(scal, 1) if len(scal) > 1 else scal[0],
        "legal": obs["legal"].bool(),
    }


def index_features(feats: dict[str, torch.Tensor], idx: torch.Tensor) -> dict[str, torch.Tensor]:
    return {k: v[idx] for k, v in feats.items()}


def cat_features(parts: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {k: torch.cat([p[k] for p in parts], 0) for k in parts[0]}


def features_to(
    feats: dict[str, torch.Tensor], device: torch.device | str
) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in feats.items()}
