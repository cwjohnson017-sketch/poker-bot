"""Averaged-strategy files written by ``poker_engine.Trainer.export_strategy``.

Pure numpy reader for inspection, tests and tools; the play-time agent uses
the Rust loader ``poker_engine.BlueprintStrategy`` instead. File layout
(little-endian, see ``engine/src/mccfr.rs``)::

    magic "PBSTRAT1" | u32 version (1)
    u32 len + solver config (binary, for the Rust loader)
    u32 len + solver config as JSON
    u32 len + training metadata as JSON (may be empty)
    zero padding to a multiple of 8 bytes
    u64 N (infosets) | u64 P (probabilities)
    N x (u64 key_hi, u64 key_lo)   keys sorted ascending as 128-bit integers
    (N + 1) x u32 offsets          row i = probs[offsets[i]:offsets[i + 1]]
    P x u16 probs                  probability * 65535, rounded

Each row has one probability per abstract action available at the infoset,
in the order of ``ActionAbstraction.legal(state)``.

Infoset keys are 128-bit integers: bits 126-127 street, bits 97-125 the
acting player's card bucket, bits 0-96 the abstract betting sequence (a
leading 1 bit, then one 4-bit abstract action index per action of the hand,
oldest first).
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

MAGIC = b"PBSTRAT1"
SEQ_BITS = 97
BUCKET_BITS = 29
STREET_SHIFT = 126


def make_key(street: int, bucket: int, sequence: list[int] | tuple[int, ...]) -> int:
    """Infoset key from street, bucket and abstract action indices."""
    if len(sequence) > 24:
        raise ValueError("betting sequence longer than 24 actions")
    seq = 1
    for i in sequence:
        if not 0 <= i < 16:
            raise ValueError(f"abstract index {i} out of range")
        seq = (seq << 4) | int(i)
    if not 0 <= bucket < (1 << BUCKET_BITS):
        raise ValueError("bucket out of range")
    return seq | (int(bucket) << SEQ_BITS) | (int(street) << STREET_SHIFT)


def decode_key(key: int) -> tuple[int, int, list[int]]:
    """``(street, bucket, sequence)`` of an infoset key."""
    street = key >> STREET_SHIFT
    bucket = (key >> SEQ_BITS) & ((1 << BUCKET_BITS) - 1)
    seq = key & ((1 << SEQ_BITS) - 1)
    out: list[int] = []
    while seq > 1:
        out.append(seq & 0xF)
        seq >>= 4
    return int(street), int(bucket), out[::-1]


@dataclass
class StrategyFile:
    config: dict[str, Any]
    meta: dict[str, Any]
    keys_hi: np.ndarray
    keys_lo: np.ndarray
    offsets: np.ndarray
    probs_q: np.ndarray
    _index: dict[int, int] | None = field(default=None, repr=False)

    def __len__(self) -> int:
        return len(self.keys_hi)

    def key(self, i: int) -> int:
        return (int(self.keys_hi[i]) << 64) | int(self.keys_lo[i])

    def row(self, i: int) -> np.ndarray:
        q = self.probs_q[self.offsets[i] : self.offsets[i + 1]].astype(np.float64)
        s = q.sum()
        return q / s if s > 0 else np.full(len(q), 1.0 / len(q))

    def find(self, key: int) -> int | None:
        """Row index of ``key`` (binary search), or None."""
        hi, lo = np.uint64(key >> 64), np.uint64(key & ((1 << 64) - 1))
        a = int(np.searchsorted(self.keys_hi, hi, side="left"))
        b = int(np.searchsorted(self.keys_hi, hi, side="right"))
        if a == b:
            return None
        j = a + int(np.searchsorted(self.keys_lo[a:b], lo, side="left"))
        if j < b and self.keys_lo[j] == lo:
            return j
        return None

    def probs(self, key: int) -> np.ndarray | None:
        """Normalized probabilities for ``key``, or None when absent."""
        i = self.find(key)
        return None if i is None else self.row(i)

    def items(self):
        for i in range(len(self)):
            yield self.key(i), self.row(i)

    @property
    def game(self) -> dict[str, Any]:
        return self.config["game"]


def read_strategy(path: str | Path) -> StrategyFile:
    """Read an exported strategy file with numpy (no Rust needed)."""
    data = Path(path).read_bytes()
    if data[:8] != MAGIC:
        raise ValueError(f"{path}: not a strategy file")
    (version,) = struct.unpack_from("<I", data, 8)
    if version != 1:
        raise ValueError(f"{path}: unsupported version {version}")
    pos = 12
    blobs = []
    for _ in range(3):
        (n,) = struct.unpack_from("<I", data, pos)
        blobs.append(data[pos + 4 : pos + 4 + n])
        pos += 4 + n
    pos += (-pos) % 8
    n, p = struct.unpack_from("<QQ", data, pos)
    pos += 16
    keys = np.frombuffer(data, dtype="<u8", count=2 * n, offset=pos).reshape(n, 2)
    pos += 16 * n
    offsets = np.frombuffer(data, dtype="<u4", count=n + 1, offset=pos)
    pos += 4 * (n + 1)
    probs_q = np.frombuffer(data, dtype="<u2", count=p, offset=pos)
    config = json.loads(blobs[1].decode())
    meta = json.loads(blobs[2].decode()) if blobs[2] else {}
    return StrategyFile(
        config=config,
        meta=meta,
        keys_hi=np.ascontiguousarray(keys[:, 0]),
        keys_lo=np.ascontiguousarray(keys[:, 1]),
        offsets=offsets,
        probs_q=probs_q,
    )


def export_strategy(trainer: Any, path: str | Path) -> int:
    """Write ``trainer``'s averaged strategy to ``path``; returns the infoset count."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return int(trainer.export_strategy(str(path)))
