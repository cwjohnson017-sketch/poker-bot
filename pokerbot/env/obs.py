"""Observation encoder for the acting player (network inputs, DESIGN.md 5.5).

``encode_obs(env)`` returns a dict of tensors, all with leading dim ``n``:

* ``cards [n, 7]`` long: own hole cards (2) then board (5); undealt board
  slots hold ``NO_CARD`` (52), a padding index for embeddings.
* ``card_mask [n, 7]`` bool: True for dealt, visible cards.
* ``hist [n, T]`` long: betting tokens, 0 = padding. A token is
  ``1 + (street * 2 + actor_is_button) * A + abstract_index``
  (``env.vocab_size`` tokens in total). ``hist_mask [n, T]`` bool.
* ``hist_amt [n, T]`` float: chips each action put in, / starting stack.
* ``scalars [n, NUM_SCALARS]`` float, amounts as fractions of the acting
  player's starting stack: pot, own stack, opponent stack, own street bet,
  opponent street bet, to call, min raise-to, max raise-to; then the
  street one-hot (4), is-button, raises this street / max raises.
* ``legal [n, A]`` bool and ``actor [n]`` long (-1 for finished slots,
  whose other fields describe seat 0).

Equity hooks (off by default, they cost far more than a step):
``equity_samples > 0`` adds ``equity [n]`` (Monte Carlo vs a random hand);
``hist_runouts > 0`` adds ``equity_hist [n, bins]``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .cards import NO_CARD
from .equity import equity_histogram, equity_vs_random

if TYPE_CHECKING:
    from .vec_env import VecNLHE

SCALAR_NAMES = (
    "pot", "stack", "opp_stack", "street_bet", "opp_street_bet", "to_call", "min_raise_to", "max_raise_to",
    "preflop", "flop", "turn", "river", "is_button", "raises_frac",
)
NUM_SCALARS = len(SCALAR_NAMES)


@torch.no_grad()
def encode_obs(
    env: "VecNLHE",
    equity_samples: int = 0,
    hist_runouts: int = 0,
    hist_bins: int = 10,
    hist_opp_samples: int = 0,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    info = env.legal_info()
    targets = env._targets(info)
    legal = env._mask(info, targets)
    p = env.actor.clamp(min=0)
    pi, oi = p[:, None], 1 - p[:, None]
    cards9 = env.cards
    hole = torch.where(p[:, None] == 0, cards9[:, 0:2], cards9[:, 2:4])
    board = cards9[:, 4:9]
    slot = torch.arange(5, device=env.device)
    board_vis = slot[None, :] < env.board_len[:, None]
    board = torch.where(board_vis, board, NO_CARD)
    cards = torch.cat([hole, board], 1)
    card_mask = torch.cat([torch.ones_like(board_vis[:, :2]), board_vis], 1)

    start = env.start_stacks[p].float()
    stacks = env.stacks.float()
    bets = env.street_bets.float()
    street = env.street.clamp(0, 3)
    feats = [
        info.pot.float(),
        stacks.gather(1, pi).squeeze(1),
        stacks.gather(1, oi).squeeze(1),
        bets.gather(1, pi).squeeze(1),
        bets.gather(1, oi).squeeze(1),
        info.to_call.float(),
        info.min_raise_to.float(),
        info.max_raise_to.float(),
    ]
    scal = torch.stack(feats, 1) / start[:, None]
    street_oh = torch.nn.functional.one_hot(street, 4).float()
    is_button = (p == env.button).float()[:, None]
    raises = (env.n_raises.float() / max(1, env.spec.max_raises))[:, None]
    scalars = torch.cat([scal, street_oh, is_button, raises], 1)

    hist = env.hist_tok
    out = {
        "cards": cards,
        "card_mask": card_mask,
        "hist": hist,
        "hist_mask": hist != 0,
        "hist_amt": env.hist_amt.float() / start[:, None],
        "scalars": scalars,
        "legal": legal,
        "actor": info.actor,
    }
    if equity_samples > 0:
        out["equity"] = equity_vs_random(hole, board, equity_samples, generator)
    if hist_runouts > 0:
        out["equity_hist"] = equity_histogram(hole, board, hist_runouts, hist_bins, hist_opp_samples, generator)
    return out
