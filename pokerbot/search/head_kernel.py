"""Fused inference of the value nets' per-combo residual head (Triton).

The residual head of :class:`~.value_net.RiverValueNet` is an MLP ``8 -> H -> H
-> 1`` applied to every (row, player, combo). In PyTorch its features are stacked
into an ``[n, 2, 1326, 8]`` tensor and every layer and activation writes an
``[n, 2, 1326, H]`` one. That memory traffic, not the arithmetic, made
``river_v3r`` 8x dearer per row than ``river_v3`` (11.0 against 1.4 ms for the
1,440 rows of a depth-0 turn search's leaves, RTX 4070 Ti).

:func:`fused_head` instead computes the head of a block of combos in registers
from the per-combo inputs and writes ``ev + head`` once. The three context
features are constant over a row and are folded into the first layer's bias.

* **When it applies** (:func:`applies`): inference only (no autograd), CUDA
  tensors, Triton importable, the standard head (``Linear, act, Linear, act,
  Linear``, GELU or ReLU, a power-of-two width of at least 16), and
  :data:`ENABLED` (environment ``POKERBOT_FUSED_HEAD=0`` turns it off).
  Otherwise the caller uses the PyTorch path.
* **Precision.** The math is fp32, with the layer-2 product in TF32: a little
  closer to the trained function than the bf16 autocast path, so outputs
  differ from it by bf16 rounding noise.

Triton ships with PyTorch on Linux; on Windows it is the ``triton-windows``
package (3.6 for PyTorch 2.11).
"""

from __future__ import annotations

import os
from typing import Any

import torch
from torch import nn

from .combos import NUM_COMBOS, blocked_sum

try:  # Triton is optional: without it (CPU-only installs) the PyTorch path is used
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - depends on the install
    triton = None

C = NUM_COMBOS
NUM_FEATURES = 8  # ev, pct, blocked, strength-weighted blocked, own reach, 3 context
BLOCK = 128  # combos per program
# False forces the PyTorch path (comparisons, debugging); POKERBOT_FUSED_HEAD=0 sets it
ENABLED = os.environ.get("POKERBOT_FUSED_HEAD", "1") != "0"


if triton is not None:

    @triton.jit
    def _act(x, GELU: tl.constexpr):
        if GELU:
            return 0.5 * x * (1.0 + tl.erf(x * 0.7071067811865476))
        return tl.maximum(x, 0.0)

    @triton.jit
    def _head_kernel(
        ev_ptr,
        m_ptr,
        sw_ptr,
        r_ptr,
        pct_ptr,
        valid_ptr,
        nvalid_ptr,
        ctx_ptr,
        w1_ptr,
        b1_ptr,
        w2_ptr,
        b2_ptr,
        w3_ptr,
        b3_ptr,
        out_ptr,
        NC: tl.constexpr,
        NF: tl.constexpr,
        H: tl.constexpr,
        BLOCK_C: tl.constexpr,
        GELU: tl.constexpr,
    ):
        rp = tl.program_id(0)  # row * 2 + player
        row = rp // 2
        offs = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
        mask = offs < NC
        base = rp.to(tl.int64) * NC
        rbase = row.to(tl.int64) * NC
        ev = tl.load(ev_ptr + base + offs, mask=mask, other=0.0)
        m = tl.load(m_ptr + base + offs, mask=mask, other=0.0)
        sw = tl.load(sw_ptr + base + offs, mask=mask, other=0.0)
        r = tl.load(r_ptr + base + offs, mask=mask, other=0.0)
        pct = tl.load(pct_ptr + rbase + offs, mask=mask, other=0.0)
        vf = tl.load(valid_ptr + rbase + offs, mask=mask, other=0).to(tl.float32)
        nv = tl.load(nvalid_ptr + row)
        j = tl.arange(0, H)
        # layer 1: W1 is [H, NF]; the context features (columns 5-7) go into the bias
        b1 = tl.load(b1_ptr + j)
        for k in tl.static_range(3):
            b1 += tl.load(ctx_ptr + row * 3 + k) * tl.load(w1_ptr + j * NF + 5 + k)
        h = ev[:, None] * tl.load(w1_ptr + j * NF)[None, :]
        h += pct[:, None] * tl.load(w1_ptr + j * NF + 1)[None, :]
        h += (10.0 * (1.0 - m) * vf)[:, None] * tl.load(w1_ptr + j * NF + 2)[None, :]
        h += (10.0 * sw)[:, None] * tl.load(w1_ptr + j * NF + 3)[None, :]
        h += (r * nv)[:, None] * tl.load(w1_ptr + j * NF + 4)[None, :]
        h = _act(h + b1[None, :], GELU)
        # layer 2: W2 is [H_out, H_in]; load it transposed
        w2t = tl.load(w2_ptr + j[None, :] * H + j[:, None])
        h = _act(tl.dot(h, w2t) + tl.load(b2_ptr + j)[None, :], GELU)
        # layer 3: one output
        head = tl.sum(h * tl.load(w3_ptr + j)[None, :], axis=1) + tl.load(b3_ptr)
        tl.store(out_ptr + base + offs, ev + head, mask=mask)


def _layers(head: nn.Module) -> tuple[nn.Linear, nn.Linear, nn.Linear, bool] | None:
    """The three Linear layers and whether the activation is GELU, or ``None``
    if ``head`` is not the standard residual head."""
    if not isinstance(head, nn.Sequential) or len(head) != 5:
        return None
    l1, a1, l2, a2, l3 = head
    if not all(isinstance(x, nn.Linear) for x in (l1, l2, l3)):
        return None
    if type(a1) is not type(a2) or not isinstance(a1, (nn.GELU, nn.ReLU)):
        return None
    if isinstance(a1, nn.GELU) and a1.approximate != "none":
        return None
    H = l1.out_features
    if l1.in_features != NUM_FEATURES or l2.in_features != H or l2.out_features != H:
        return None
    if l3.in_features != H or l3.out_features != 1 or H < 16 or H & (H - 1):
        return None
    return l1, l2, l3, isinstance(a1, nn.GELU)


def applies(head: nn.Module | None, device: torch.device) -> bool:
    """Whether :func:`fused_head` can replace the PyTorch head here."""
    return (
        ENABLED
        and triton is not None
        and head is not None
        and device.type == "cuda"
        and not torch.is_grad_enabled()
        and _layers(head) is not None
    )


def _f32(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to(torch.float32).contiguous()


@torch.no_grad()
def fused_head(
    head: nn.Module,
    ev: torch.Tensor,
    r: torch.Tensor,
    pct: torch.Tensor,
    valid: torch.Tensor,
    m: torch.Tensor,
    ctx: torch.Tensor,
) -> torch.Tensor:
    """``ev + head(features)`` as ``[n, 2, 1326]`` fp32, the features as in
    :meth:`~.value_net.RiverValueNet._head`: ``ev`` the decoded values, ``r``
    the normalised ranges and ``m`` :func:`~.value_net.opponent_mass` (all
    ``[n, 2, 1326]``), ``pct`` and ``valid`` ``[n, 1326]``, ``ctx [n, 3]``."""
    layers: Any = _layers(head)
    if layers is None:
        raise ValueError("not the standard residual head")
    l1, l2, l3, gelu = layers
    n = r.shape[0]
    out = torch.empty(n, 2, C, dtype=torch.float32, device=r.device)
    if n == 0:
        return out
    r = _f32(r)
    pct = _f32(pct)
    vf = valid.to(torch.float32)
    # strength-weighted opponent mass sharing a card with the combo (masked)
    q = r.flip(1) * pct[:, None, :]
    sw = (q.sum(-1, keepdim=True) - blocked_sum(q)) * vf[:, None, :]
    grid = (2 * n, triton.cdiv(C, BLOCK))
    _head_kernel[grid](
        _f32(ev),
        _f32(m),
        sw.contiguous(),
        r,
        pct,
        valid.contiguous().view(torch.uint8),
        vf.sum(-1).contiguous(),
        _f32(ctx),
        _f32(l1.weight),
        _f32(l1.bias),
        _f32(l2.weight),
        _f32(l2.bias),
        _f32(l3.weight).view(-1),
        _f32(l3.bias),
        out,
        NC=C,
        NF=NUM_FEATURES,
        H=l1.out_features,
        BLOCK_C=BLOCK,
        GELU=gelu,
    )
    return out


__all__ = ["ENABLED", "applies", "fused_head"]
