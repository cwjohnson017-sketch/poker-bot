"""Batched hand-equity kernels (pure torch, device-agnostic).

All functions take ``hole [N, 2]`` and a board ``[N, k]`` with ``k <= 5``;
board entries equal to ``NO_CARD`` (52) are unknown and get sampled, so rows
of one batch may sit on different streets. Equity counts a tie as half a win.

* :func:`equity_vs_random` - Monte Carlo: random runout + random opponent hand.
* :func:`equity_river` - exact on a complete board: all 990 opponent hands.
* :func:`equity_histogram` - distribution of river equity over sampled
  runouts, as ``bins`` equal-width bins (the "equity histogram" feature).

Work is split into chunks of at most ``max_rows`` evaluated hands to bound
memory; the chunk loop is over blocks, never over individual hands.
"""

from __future__ import annotations

import torch

from .cards import NO_CARD
from .evaluator import evaluate_batch

_PAIRS: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}


def _pairs45(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    key = str(device)
    if key not in _PAIRS:
        ij = torch.combinations(torch.arange(45), 2).to(device)
        _PAIRS[key] = (ij[:, 0].contiguous(), ij[:, 1].contiguous())
    return _PAIRS[key]


def pad_board(board: torch.Tensor | None, n: int, device: torch.device) -> torch.Tensor:
    """``[N, k]`` -> ``[N, 5]`` padded with ``NO_CARD``."""
    if board is None or board.shape[-1] == 0:
        return torch.full((n, 5), NO_CARD, dtype=torch.long, device=device)
    board = board.long()
    if board.shape[1] < 5:
        pad = torch.full((n, 5 - board.shape[1]), NO_CARD, dtype=torch.long, device=device)
        board = torch.cat([board, pad], 1)
    return board


def sample_unknown(
    known: torch.Tensor, k: int, generator: torch.Generator | None = None
) -> torch.Tensor:
    """``k`` distinct cards per row, uniformly from cards not in ``known``.

    ``known`` is ``[M, K]`` and may contain ``NO_CARD`` padding.
    """
    m = known.shape[0]
    keys = torch.rand(m, NO_CARD + 1, generator=generator, device=known.device)
    keys.scatter_(1, known.long(), 2.0)
    keys[:, NO_CARD] = 3.0
    return keys.topk(k, dim=1, largest=False).indices


def _chunks(n: int, per_row: int, max_rows: int):
    step = max(1, max_rows // max(1, per_row))
    for s in range(0, n, step):
        yield s, min(n, s + step)


@torch.no_grad()
def equity_vs_random(
    hole: torch.Tensor,
    board: torch.Tensor | None,
    n_samples: int = 256,
    generator: torch.Generator | None = None,
    max_rows: int = 1 << 20,
) -> torch.Tensor:
    """Monte Carlo equity against one uniformly random opponent hand, ``[N]`` float."""
    hole = hole.long()
    n, dev = hole.shape[0], hole.device
    board5 = pad_board(board, n, dev)
    known = torch.cat([hole, board5], 1)
    out = torch.empty(n, dtype=torch.float32, device=dev)
    S = int(n_samples)
    for s, e in _chunks(n, S, max_rows):
        m = e - s
        kn = known[s:e].repeat_interleave(S, 0)
        samp = sample_unknown(kn, 7, generator)
        b = board5[s:e].repeat_interleave(S, 0)
        full = torch.where(b == NO_CARD, samp[:, :5], b)
        hero = torch.cat([hole[s:e].repeat_interleave(S, 0), full], 1)
        vill = torch.cat([samp[:, 5:7], full], 1)
        r = evaluate_batch(torch.stack([hero, vill], 1))
        score = (r[:, 0] > r[:, 1]).float() + 0.5 * (r[:, 0] == r[:, 1]).float()
        out[s:e] = score.view(m, S).mean(1)
    return out


@torch.no_grad()
def equity_river(hole: torch.Tensor, board5: torch.Tensor, max_rows: int = 1 << 21) -> torch.Tensor:
    """Exact equity on a complete board vs all 990 opponent hands, ``[N]`` float."""
    hole, board5 = hole.long(), board5.long()
    n, dev = hole.shape[0], hole.device
    ii, jj = _pairs45(dev)
    out = torch.empty(n, dtype=torch.float32, device=dev)
    for s, e in _chunks(n, 990, max_rows):
        m = e - s
        h, b = hole[s:e], board5[s:e]
        known = torch.cat([h, b], 1)
        used = torch.zeros(m, 52, dtype=torch.long, device=dev).scatter_(1, known, 1)
        remaining = torch.argsort(used, dim=1, stable=True)[:, :45]  # unused cards, ascending
        opp = torch.stack([remaining[:, ii], remaining[:, jj]], 2)  # [m, 990, 2]
        vill = torch.cat([opp, b[:, None, :].expand(m, 990, 5)], 2)
        rv = evaluate_batch(vill)  # [m, 990]
        rh = evaluate_batch(known)  # [m]
        score = (rh[:, None] > rv).float() + 0.5 * (rh[:, None] == rv).float()
        out[s:e] = score.mean(1)
    return out


@torch.no_grad()
def equity_histogram(
    hole: torch.Tensor,
    board: torch.Tensor | None,
    n_runouts: int = 32,
    bins: int = 10,
    n_opp: int = 0,
    generator: torch.Generator | None = None,
    return_equity: bool = False,
):
    """Histogram of river equity over ``n_runouts`` sampled board completions.

    Returns ``[N, bins]`` float rows summing to 1 (and the mean equity ``[N]``
    when ``return_equity``). Each runout's equity is exact (990 opponent hands)
    when ``n_opp == 0``, otherwise estimated from ``n_opp`` sampled opponents.
    """
    hole = hole.long()
    n, dev = hole.shape[0], hole.device
    R = int(n_runouts)
    board5 = pad_board(board, n, dev)
    known = torch.cat([hole, board5], 1).repeat_interleave(R, 0)
    samp = sample_unknown(known, 5, generator)
    b = board5.repeat_interleave(R, 0)
    full = torch.where(b == NO_CARD, samp, b)
    h = hole.repeat_interleave(R, 0)
    eq = equity_river(h, full) if n_opp == 0 else equity_vs_random(h, full, n_opp, generator)
    idx = (eq * bins).long().clamp(0, bins - 1)
    hist = (
        torch.zeros(n * R, bins, device=dev).scatter_(1, idx[:, None], 1.0).view(n, R, bins).mean(1)
    )
    if return_equity:
        return hist, eq.view(n, R).mean(1)
    return hist
