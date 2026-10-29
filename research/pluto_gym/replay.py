"""Bounded replay storage in host DRAM.

The arrays contain training features and binary labels. Reported payload sizes
are exact for these arrays; they exclude the Python interpreter, sampled batch
copies, optimizer arrays and any other process allocations. This is a resident
CPU buffer, with no GPU paging or zero-copy claim.
"""

from __future__ import annotations

from array import array
import math
import numbers
import operator
import random
import sys
from typing import Iterable, Mapping, Sequence


def _index(value, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or type(value).__name__ == "bool_":
        raise ValueError(f"{name} must be an integer, not bool")
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    return value


class DRAMReplay:
    """Fixed-capacity circular storage with deterministic sampling.

    ``max_bytes`` limits the preallocated array payload. Appending validates the
    entire example before overwriting a slot. Sampling uses replacement, so a
    small corpus can serve a declared batch size without an implicit resize.
    """

    def __init__(self, capacity: int = 4096, feature_dim: int = 2,
                 max_bytes: int = 64 * 1024 * 1024):
        self.capacity = _index(capacity, "capacity", 1, 1_000_000)
        self.feature_dim = _index(feature_dim, "feature_dim", 1, 4096)
        self.max_bytes = _index(max_bytes, "max_bytes", 1, 2**40)
        self._item_bytes = array("d").itemsize
        self.row_bytes = self.feature_dim * self._item_bytes + 1
        if self.capacity * self.row_bytes > self.max_bytes:
            raise ValueError("replay array payload exceeds max_bytes")
        self._features = array("d", [0.0]) * (self.capacity * self.feature_dim)
        self._targets = bytearray(self.capacity)
        self._size = 0
        self._next = 0
        self.total_appended = 0

    def __len__(self) -> int:
        return self._size

    def append(self, features: Sequence[float], target: int) -> None:
        if isinstance(features, (str, bytes)):
            raise ValueError("features must be a numeric sequence")
        try:
            raw = tuple(features)
        except TypeError as exc:
            raise ValueError("features must be a numeric sequence") from exc
        if len(raw) != self.feature_dim:
            raise ValueError("feature dimension mismatch")
        values = []
        for value in raw:
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                raise ValueError("features must be real finite numbers")
            value = float(value)
            if not math.isfinite(value):
                raise ValueError("features must be real finite numbers")
            values.append(value)
        label = _index(target, "target", 0, 1)
        start = self._next * self.feature_dim
        self._features[start:start + self.feature_dim] = array("d", values)
        self._targets[self._next] = label
        self._next = (self._next + 1) % self.capacity
        self._size = min(self.capacity, self._size + 1)
        self.total_appended += 1

    def extend(self, examples: Iterable[Mapping]) -> int:
        count = 0
        for example in examples:
            if not isinstance(example, Mapping) or "features" not in example or "target" not in example:
                raise ValueError("example requires features and target")
            self.append(example["features"], example["target"])
            count += 1
        return count

    def rows(self) -> list[dict]:
        """Return copies in oldest-to-newest order for inspection/checkpointing."""
        return list(self.iter_rows())

    def iter_rows(self):
        """Yield one copied row at a time without expanding the whole buffer."""
        first = self._next if self._size == self.capacity else 0
        for j in range(self._size):
            yield self._row((first + j) % self.capacity)

    def _row(self, index: int) -> dict:
        start = index * self.feature_dim
        return {"features": list(self._features[start:start + self.feature_dim]),
                "target": self._targets[index]}

    def sample(self, batch_size: int, *, seed: int | None = None,
               rng: random.Random | None = None) -> list[dict]:
        batch_size = _index(batch_size, "batch_size", 1, 65_536)
        if not self._size:
            raise ValueError("cannot sample empty replay")
        if seed is not None and rng is not None:
            raise ValueError("provide seed or rng, not both")
        if rng is None:
            seed = 0 if seed is None else _index(seed, "seed", 0, 2**63 - 1)
            rng = random.Random(seed)
        elif not isinstance(rng, random.Random):
            raise ValueError("rng must be random.Random")
        return [self._row(rng.randrange(self._size)) for _ in range(batch_size)]

    def ledger(self) -> dict:
        allocated = len(self._features) * self._item_bytes + len(self._targets)
        python_buffers = sys.getsizeof(self._features) + sys.getsizeof(self._targets)
        return {
            "storage": "host_dram_packed_float64_binary_labels",
            "capacity_rows": self.capacity,
            "retained_rows": self._size,
            "total_appended": self.total_appended,
            "overwritten_rows": max(0, self.total_appended - self.capacity),
            "feature_dim": self.feature_dim,
            "feature_item_bytes": self._item_bytes,
            "row_payload_bytes": self.row_bytes,
            "allocated_payload_bytes": allocated,
            "retained_payload_bytes": self._size * self.row_bytes,
            "declared_payload_limit_bytes": self.max_bytes,
            "python_buffer_overhead_bytes": python_buffers - allocated,
            "python_container_bytes": sys.getsizeof(self) + sys.getsizeof(self.__dict__),
            "batch_copies_included": False,
            "process_rss_included": False,
            "gpu_transfers_bytes": 0,
        }
