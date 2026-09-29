"""Fixed-capacity reservoir memories on the host (DESIGN.md 5.5 "Memories").

Every sample is an infoset's features plus a target vector (instantaneous
regrets for an advantage memory, a policy for a strategy memory) and the CFR
iteration it came from. Storage is a set of preallocated host arrays in a
compact layout (bytes per sample at the defaults, A = 6, T = 24, S = 14):

====================  =========================  =====
field                 dtype / shape              bytes
====================  =========================  =====
``cards``             uint8 ``[7]``              7
``hist``              uint8 ``[T]`` (int16 if    24
                      vocab > 256)
``hist_amt``          float16 ``[T]``            48
``scalars``           float16 ``[S]``            28
``legal``             bool ``[A]``               6
``target``            float16 ``[A]``            12
``iteration``         uint16                     2
====================  =========================  =====

About 127 bytes per sample, 5.1 GB for 40M samples. Arrays are numpy
(allocated with ``np.zeros``, so untouched capacity costs no RAM) and are
exposed to torch with ``torch.from_numpy`` only on the gathered minibatch.

Reservoir sampling (Vitter's algorithm R): the ``k``-th sample ever offered
(0-based) goes to slot ``k`` while the buffer fills, afterwards it replaces
a uniformly random slot with probability ``capacity / (k + 1)``. Every
sample ever offered is therefore held with the same probability. A batch is
processed exactly as the sequential algorithm would (when two samples of a
batch pick the same slot, the later one wins).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ...env.vec_env import HISTORY_LEN


class ReservoirMemory:
    def __init__(
        self,
        capacity: int,
        num_actions: int,
        num_scalars: int,
        history_len: int = HISTORY_LEN,
        vocab_size: int = 256,
        seed: int = 0,
        name: str = "memory",
    ) -> None:
        self.capacity = int(capacity)
        self.num_actions = int(num_actions)
        self.num_scalars = int(num_scalars)
        self.history_len = int(history_len)
        self.vocab_size = int(vocab_size)
        self.name = name
        self.seen = 0  # samples ever offered
        self.size = 0
        self.rng = np.random.default_rng(seed)
        C, T, A, S = self.capacity, self.history_len, self.num_actions, self.num_scalars
        hist_dtype = np.uint8 if vocab_size <= 256 else np.int16
        self.arrays: dict[str, np.ndarray] = {
            "cards": np.zeros((C, 7), np.uint8),
            "hist": np.zeros((C, T), hist_dtype),
            "hist_amt": np.zeros((C, T), np.float16),
            "scalars": np.zeros((C, S), np.float16),
            "legal": np.zeros((C, A), np.bool_),
            "target": np.zeros((C, A), np.float16),
            "iteration": np.zeros((C,), np.uint16),
        }
        self._pinned: dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------ info
    def __len__(self) -> int:
        return self.size

    @property
    def bytes_per_sample(self) -> int:
        return int(sum(a.dtype.itemsize * int(np.prod(a.shape[1:])) for a in self.arrays.values()))

    def stats(self) -> dict[str, Any]:
        n = self.size
        out: dict[str, Any] = {
            "size": n,
            "seen": self.seen,
            "capacity": self.capacity,
            "fill": n / max(1, self.capacity),
            "bytes_per_sample": self.bytes_per_sample,
            "resident_mb": n * self.bytes_per_sample / 2**20,
        }
        if n:
            # a strided subsample keeps stats cheap on 40M rows
            step = max(1, n // 100_000)
            it = self.arrays["iteration"][:n:step].astype(np.float64)
            tg = self.arrays["target"][:n:step].astype(np.float32)
            lg = self.arrays["legal"][:n:step]
            out["mean_iteration"] = float(it.mean())
            out["max_iteration"] = int(it.max())
            out["target_abs_mean"] = float(np.abs(tg[lg]).mean()) if lg.any() else 0.0
        return out

    # ------------------------------------------------------------------ writing
    def add_batch(self, batch: dict[str, Any]) -> int:
        """Offer ``m`` samples (dict of arrays/tensors with leading dim m).

        Keys: ``cards, hist, hist_amt, scalars, legal, target`` and
        ``iteration`` (a scalar or ``[m]``). Returns the number written.
        """
        data = {k: _to_numpy(v) for k, v in batch.items()}
        m = int(data["cards"].shape[0])
        if m == 0:
            return 0
        it = data["iteration"]
        if np.ndim(it) == 0:
            data["iteration"] = np.full((m,), int(it), np.int64)
        k = self.seen + np.arange(m, dtype=np.int64)
        pos = np.where(
            k < self.capacity, k, self.rng.integers(0, k + 1, dtype=np.int64)
        )  # high is exclusive: j uniform in [0, k]
        keep = pos < self.capacity
        src = np.nonzero(keep)[0]
        dst = pos[keep]
        # later samples overwrite earlier ones on collisions: keep the last
        # occurrence of each destination slot
        rev_dst = dst[::-1]
        _, first_rev = np.unique(rev_dst, return_index=True)
        sel = len(dst) - 1 - first_rev
        src, dst = src[sel], dst[sel]
        for name, arr in self.arrays.items():
            v = data[name]
            if name == "legal":
                v = v.astype(np.bool_)
            arr[dst] = v[src].astype(arr.dtype, copy=False)
        self.seen += m
        self.size = min(self.capacity, self.seen)
        return len(dst)

    # ------------------------------------------------------------------ reading
    def sample(
        self,
        batch_size: int,
        device: torch.device | str = "cpu",
        rng: np.random.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        """Uniform minibatch (with replacement) as training tensors on ``device``.

        Returns ``cards/hist`` long, ``card_mask`` bool, ``hist_amt/scalars/
        target`` float32, ``legal`` bool and ``iteration`` float32. On CUDA the
        compact host batch goes through reused pinned buffers and a
        non-blocking copy; decoding to wide dtypes happens on the device.
        """
        if self.size == 0:
            raise ValueError(f"{self.name} is empty")
        rng = rng or self.rng
        idx = np.sort(rng.integers(0, self.size, size=int(batch_size)))
        return self.gather(idx, device)

    def gather(
        self, idx: np.ndarray, device: torch.device | str = "cpu"
    ) -> dict[str, torch.Tensor]:
        device = torch.device(device)
        host = {k: torch.from_numpy(np.ascontiguousarray(a[idx])) for k, a in self.arrays.items()}
        if device.type == "cuda":
            moved = {}
            for k, t in host.items():
                buf = self._pinned.get(k)
                if buf is None or buf.shape != t.shape or buf.dtype != t.dtype:
                    buf = torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
                    self._pinned[k] = buf
                buf.copy_(t)
                moved[k] = buf.to(device, non_blocking=True)
            host = moved
        cards = host["cards"].long()
        return {
            "cards": cards,
            "card_mask": cards < 52,
            "hist": host["hist"].long(),
            "hist_amt": host["hist_amt"].float(),
            "scalars": host["scalars"].float(),
            "legal": host["legal"].bool(),
            "target": host["target"].float(),
            "iteration": host["iteration"].float(),
        }

    # ------------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> None:
        """Write the filled rows as one ``.npy`` per field plus ``meta.json``."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        n = self.size
        for name, arr in self.arrays.items():
            tmp = path / f"{name}.tmp.npy"
            np.save(tmp, arr[:n])
            tmp.replace(path / f"{name}.npy")
        meta = {
            "capacity": self.capacity,
            "num_actions": self.num_actions,
            "num_scalars": self.num_scalars,
            "history_len": self.history_len,
            "vocab_size": self.vocab_size,
            "seen": self.seen,
            "size": n,
            "name": self.name,
            "rng": self.rng.bit_generator.state,
        }
        (path / "meta.json").write_text(json.dumps(meta))

    @classmethod
    def load(
        cls, path: str | Path, capacity: int | None = None, mmap: bool = False
    ) -> ReservoirMemory:
        """Load a saved memory. Files are opened memory-mapped and streamed into
        freshly allocated arrays of ``capacity`` rows (default: the saved one).

        With ``mmap=True`` and a full buffer of the same capacity the arrays
        stay memory-mapped copy-on-write (pages are read on demand, writes stay
        in RAM, the files are never modified).
        """
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())
        cap = int(capacity or meta["capacity"])
        mem = cls(
            cap,
            meta["num_actions"],
            meta["num_scalars"],
            meta["history_len"],
            meta["vocab_size"],
            name=meta.get("name", "memory"),
        )
        n = int(meta["size"])
        if mmap and n == cap:
            mem.arrays = {k: np.load(path / f"{k}.npy", mmap_mode="c") for k in mem.arrays}
        else:
            keep = min(n, cap)
            for k, arr in mem.arrays.items():
                src = np.load(path / f"{k}.npy", mmap_mode="r")
                arr[:keep] = src[:keep]
            n = keep
        mem.size = n
        # A full buffer keeps its stream count (every offered sample is still held
        # with probability capacity / seen). A partly filled one restarts it.
        mem.seen = int(meta["seen"]) if n == cap else n
        if "rng" in meta:
            mem.rng.bit_generator.state = meta["rng"]
        return mem


def _to_numpy(v: Any) -> np.ndarray:
    if isinstance(v, torch.Tensor):
        v = v.detach()
        if v.dtype == torch.bfloat16:
            v = v.float()
        return v.cpu().numpy()
    return np.asarray(v)
