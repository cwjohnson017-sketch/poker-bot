"""Checkpoint layout of a Deep CFR run.

::

    <run>/checkpoints/meta.json          spec, features, game, net config, value scale
    <run>/checkpoints/p0/iter1.pt ...    advantage net of seat 0 after iteration t
    <run>/checkpoints/p1/iter1.pt ...
    <run>/memory/adv_p0/*.npy            reservoir snapshots (resume points)
    <run>/trainer_state.pt               last completed iteration + RNG states

Each ``iter{t}.pt`` is self-contained: ``state_dict``, ``net_config``,
``iteration``, ``player`` and ``meta`` (the same dict as ``meta.json``), plus
``preflop`` (the seat's preflop strategy table ``[nodes * 169, A]``) when the
run keeps tabular preflop regrets.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import torch

from .networks import AdvantageNet, NetConfig

_ITER_RE = re.compile(r"iter(\d+)\.pt$")
_DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def checkpoint_root(path: str | Path) -> Path:
    """Accept a run directory or its ``checkpoints`` directory."""
    path = Path(path)
    if (path / "checkpoints" / "meta.json").exists():
        return path / "checkpoints"
    return path


def write_meta(root: str | Path, meta: dict[str, Any]) -> None:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "meta.json").write_text(json.dumps(meta, indent=2))


def read_meta(root: str | Path) -> dict[str, Any]:
    return json.loads((checkpoint_root(root) / "meta.json").read_text())


def save_net(
    root: str | Path,
    player: int,
    iteration: int,
    net: AdvantageNet,
    meta: dict[str, Any],
    dtype: str = "float32",
    preflop: torch.Tensor | None = None,
) -> Path:
    d = Path(root) / f"p{player}"
    d.mkdir(parents=True, exist_ok=True)
    cast = _DTYPES[dtype]
    sd = {
        k: (v.detach().to("cpu", cast) if v.is_floating_point() else v.detach().cpu())
        for k, v in net.state_dict().items()
    }
    path = d / f"iter{iteration}.pt"
    tmp = path.with_suffix(".tmp")
    ck = {
        "state_dict": sd,
        "net_config": net.cfg.to_dict(),
        "iteration": int(iteration),
        "player": int(player),
        "meta": meta,
    }
    if preflop is not None:
        ck["preflop"] = preflop.detach().to("cpu", torch.float32)
    torch.save(ck, tmp)
    tmp.replace(path)
    return path


def load_checkpoint(
    path: str | Path, device: torch.device | str = "cpu"
) -> tuple[int, AdvantageNet, torch.Tensor | None]:
    """``(iteration, net, preflop table or None)`` of one checkpoint file."""
    ck = torch.load(path, map_location="cpu", weights_only=True)
    net = AdvantageNet(NetConfig.from_dict(ck["net_config"]))
    sd = {k: (v.float() if v.is_floating_point() else v) for k, v in ck["state_dict"].items()}
    net.load_state_dict(sd)
    pre = ck.get("preflop")
    return int(ck["iteration"]), net.to(device).eval(), pre


def load_net(path: str | Path, device: torch.device | str = "cpu") -> tuple[int, AdvantageNet]:
    t, net, _ = load_checkpoint(path, device)
    return t, net


def list_checkpoints(
    root: str | Path, player: int, max_iter: int | None = None
) -> list[tuple[int, Path]]:
    d = checkpoint_root(root) / f"p{player}"
    out = []
    if d.exists():
        for f in d.iterdir():
            m = _ITER_RE.search(f.name)
            if m:
                t = int(m.group(1))
                if max_iter is None or t <= max_iter:
                    out.append((t, f))
    return sorted(out)
