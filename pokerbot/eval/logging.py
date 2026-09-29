"""``RunLogger``: scalars to CSV (always) and TensorBoard (when installed).

::

    with RunLogger("runs/abr_tiny", config=cfg) as log:
        log.scalar("train/loss", loss, step)
        log.scalars({"eval/mbb": m, "eval/ci": hw}, step)
        log.write_json("result.json", result)

Files in ``log_dir``: ``scalars.csv`` (long format: ``wall_time, step, tag,
value``; appended across restarts), ``run.json`` (provenance: git hash,
start time, config) and TensorBoard event files when the ``tensorboard``
package is importable and ``tensorboard`` is not False.
"""

from __future__ import annotations

import csv
import json
import os
import time
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_jsonable(v) for v in obj]
    if hasattr(obj, "item") and callable(obj.item):
        try:
            return obj.item()
        except (ValueError, RuntimeError):
            pass
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, float | int | str | bool) or obj is None:
        return obj
    return str(obj)


def write_json_atomic(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    with open(tmp, "w") as fh:
        json.dump(_jsonable(obj), fh, indent=2, sort_keys=False)
        fh.write("\n")
    os.replace(tmp, path)


def _summary_writer(log_dir: Path) -> Any:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception:  # ImportError, or tensorboard missing inside torch's shim
        return None
    try:
        return SummaryWriter(log_dir=str(log_dir))
    except Exception as e:  # pragma: no cover - broken installs
        warnings.warn(f"TensorBoard writer disabled: {e}", stacklevel=3)
        return None


class RunLogger:
    def __init__(
        self,
        log_dir: str | Path | None,
        tensorboard: bool | None = None,
        config: Mapping[str, Any] | None = None,
        echo: bool = False,
    ) -> None:
        """``log_dir=None`` gives a no-op logger. ``tensorboard``: None = use it
        when installed, False = never, True = warn when unavailable."""
        self.log_dir = Path(log_dir) if log_dir is not None else None
        self.echo = echo
        self._csv = None
        self._writer = None
        self._tb = None
        if self.log_dir is None:
            return
        self.log_dir.mkdir(parents=True, exist_ok=True)
        path = self.log_dir / "scalars.csv"
        new = not path.exists() or path.stat().st_size == 0
        self._csv = open(path, "a", newline="")  # noqa: SIM115 - closed in close()
        self._writer = csv.writer(self._csv)
        if new:
            self._writer.writerow(["wall_time", "step", "tag", "value"])
            self._csv.flush()
        if tensorboard is not False:
            self._tb = _summary_writer(self.log_dir)
            if self._tb is None and tensorboard:
                warnings.warn("tensorboard is not installed; logging to CSV only", stacklevel=2)
        info = {"start_time": time.strftime("%Y-%m-%dT%H:%M:%S"), "config": config}
        try:
            from ..config import git_hash

            info["git"] = git_hash()
        except Exception:  # pragma: no cover
            info["git"] = "unknown"
        write_json_atomic(self.log_dir / "run.json", info)

    @property
    def tensorboard_enabled(self) -> bool:
        return self._tb is not None

    def scalar(self, tag: str, value: float, step: int) -> None:
        value = float(value)
        if self._writer is not None:
            self._writer.writerow([f"{time.time():.3f}", int(step), tag, repr(value)])
        if self._tb is not None:
            self._tb.add_scalar(tag, value, int(step))
        if self.echo:
            print(f"[{step}] {tag} = {value:.6g}")

    def scalars(self, values: Mapping[str, float], step: int) -> None:
        for tag, v in values.items():
            self.scalar(tag, v, step)
        self.flush()

    def text(self, tag: str, text: str, step: int = 0) -> None:
        if self._tb is not None:
            self._tb.add_text(tag, text, int(step))
        if self.log_dir is not None:
            with open(self.log_dir / "notes.txt", "a") as fh:
                fh.write(f"[{step}] {tag}: {text}\n")

    def write_json(self, name: str, obj: Any) -> Path | None:
        if self.log_dir is None:
            return None
        path = self.log_dir / name
        write_json_atomic(path, obj)
        return path

    def flush(self) -> None:
        if self._csv is not None:
            self._csv.flush()
        if self._tb is not None:
            self._tb.flush()

    def close(self) -> None:
        self.flush()
        if self._csv is not None:
            self._csv.close()
            self._csv = None
            self._writer = None
        if self._tb is not None:
            self._tb.close()
            self._tb = None

    def __enter__(self) -> RunLogger:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def read_scalars(path: str | Path) -> dict[str, list[tuple[int, float]]]:
    """Read ``scalars.csv`` (file or its directory) into ``{tag: [(step, value)]}``."""
    path = Path(path)
    if path.is_dir():
        path = path / "scalars.csv"
    out: dict[str, list[tuple[int, float]]] = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            out.setdefault(row["tag"], []).append((int(row["step"]), float(row["value"])))
    return out
