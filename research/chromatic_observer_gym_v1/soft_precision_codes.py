"""COE-03: exact-rational analog precision vs binary and temporal observer width.

Mathematical reference for finite *equilateral* test-state metrics. Synthetic only.
No claim of an LLM checkpoint, SAE, or causal mechanism measurement.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
from itertools import combinations
from math import ceil, exp, log
from random import Random


@dataclass(frozen=True)
class SoftCode:
    """Each word's 1-bits carry the exact positive rational `amplitude`."""
    n: int
    k: int
    words: tuple[int, ...]
    amplitude: Fraction

    def __post_init__(self) -> None:
        if self.n < 2 or self.k < 1 or len(self.words) != self.n:
            raise ValueError('invalid nonempty finite code')
        if not 0 < self.amplitude <= 1:
            raise ValueError('amplitude must lie in (0,1]')
        if any(word < 0 or word >= 1 << self.k for word in self.words):
            raise ValueError('word outside declared coordinate width')

    def distance(self, u: int, v: int) -> Fraction:
        return (self.words[u] ^ self.words[v]).bit_count() * self.amplitude

    def extreme_hamming(self) -> tuple[int, int]:
        ds = [(u ^ v).bit_count() for u, v in combinations(self.words, 2)]
        return min(ds), max(ds)

    def check(self, lower: object = 1, upper: object = 2) -> bool:
        low, high = self.extreme_hamming()
        return Fraction(lower) <= low * self.amplitude and high * self.amplitude <= Fraction(upper)

    def digest(self) -> str:
        """Stable domain-separated digest: n/k/scale plus little-endian words."""
        h = sha256(b'coe03-soft-code-v1\x00')
        h.update(self.n.to_bytes(4, 'little'))
        h.update(self.k.to_bytes(4, 'little'))
        h.update(str(self.amplitude).encode('ascii') + b'\x00')
        for word in self.words:
            h.update(word.to_bytes((self.k + 7)//8, 'little'))
        return h.hexdigest()


def seeded_scaled_code(n: int, k: int, seed: int, amplitude: object) -> SoftCode:
    if n < 2 or k < 1:
        raise ValueError('n>=2 and k>=1 required')
    rng = Random(seed)
    return SoftCode(n, k, tuple(rng.getrandbits(k) for _ in range(n)), Fraction(amplitude))


def hoeffding_sufficient_width(n: int, delta: Fraction = Fraction(1, 6)) -> int:
    """Convenient float calculation of the rigorous union-bound sufficient width.

    If k > ln(n(n-1))/(2 delta^2), the probability that any pair's
    Hamming fraction exits [1/2-delta,1/2+delta] is below one.
    The analytic theorem is independent of floating-point rounding here.
    """
    if n < 2 or not 0 < delta < Fraction(1, 2):
        raise ValueError('n>=2 and delta in (0,1/2) required')
    k = max(3, int(log(n * (n - 1)) / (2 * float(delta)**2)) + 1)
    while n * (n-1) * exp(-2 * k * float(delta)**2) >= 1:
        k += 1
    return k


def k5_soft_exact() -> tuple[tuple[Fraction, ...], ...]:
    """An explicit 3-channel soft code for all ten edges of equilateral K5."""
    raw = ((0, 0, 0), (3, 0, 0), (0, 3, 0),
           (0, 0, 3), (1, 1, 1))
    return tuple(tuple(Fraction(x, 3) for x in row) for row in raw)


def soft_l1(a: tuple[Fraction, ...], b: tuple[Fraction, ...]) -> Fraction:
    if len(a) != len(b):
        raise ValueError('coordinate dimensions disagree')
    return sum((abs(u-v) for u, v in zip(a, b)), Fraction(0))


def soft_distance_matrix(states: tuple[tuple[Fraction, ...], ...]):
    return tuple(tuple(soft_l1(a,b) for b in states) for a in states)


def quantize_threshold(states: tuple[tuple[Fraction, ...], ...],
                       thresholds: tuple[Fraction, ...]) -> tuple[int, ...]:
    if any(len(row) != len(thresholds) for row in states):
        raise ValueError('threshold dimension mismatch')
    return tuple(sum((int(x >= t) << j) for j, (x, t)
                     in enumerate(zip(row, thresholds))) for row in states)


def binary_diameter_two(words: tuple[int, ...]) -> bool:
    return (len(set(words)) == len(words) and
            all(1 <= (a ^ b).bit_count() <= 2 for a,b in combinations(words,2)))


def multiplex_onehot(n: int, channels: int) -> tuple[tuple[int, ...], ...]:
    """The transcript is T blocks, each of `channels` bits, with one-hot codes.

    The *entire transcript* has Hamming diameter at most two and margin one.
    This assumes the state stays accessible each step and the evaluator reads
    the full transcript; it makes no recurrent hidden-memory claim.
    """
    if n < 2 or channels < 1:
        raise ValueError('n>=2 and channels>=1 required')
    rounds = (n - 1 + channels - 1) // channels
    rows = []
    for v in range(n):
        bits = 0 if v == 0 else (1 << (v-1))
        rows.append(tuple((bits >> (t * channels)) & ((1 << channels)-1)
                          for t in range(rounds)))
    return tuple(rows)


def transcript_distance(a: tuple[int, ...], b: tuple[int, ...]) -> int:
    if len(a) != len(b):
        raise ValueError('transcript length mismatch')
    return sum((x ^ y).bit_count() for x, y in zip(a,b))


def validate_transcript(transcripts: tuple[tuple[int, ...], ...], channels: int) -> bool:
    return bool(transcripts) and all(len(row) == len(transcripts[0]) for row in transcripts) and all(
        0 <= word < (1 << channels) for row in transcripts for word in row
    ) and all(1 <= transcript_distance(a,b) <= 2 for a,b in combinations(transcripts,2))


# Pinned finite demonstration, independent of the existence theorem constants:
DEMO_N = 1024
DEMO_K = 256
DEMO_SEED = 2305
DEMO_GRID_BITS = 10
DEMO_GRID_AMPLITUDE = Fraction(12, (1 << DEMO_GRID_BITS) - 1)


def reference_report() -> dict:
    code = seeded_scaled_code(DEMO_N, DEMO_K, DEMO_SEED, DEMO_GRID_AMPLITUDE)
    low, high = code.extreme_hamming()
    assert code.check(1, 2)
    soft5 = k5_soft_exact()
    d5 = soft_distance_matrix(soft5)
    assert min(d5[i][j] for i in range(5) for j in range(i)) == 1
    assert max(d5[i][j] for i in range(5) for j in range(i)) == 2
    transcript = multiplex_onehot(17, 4)
    assert validate_transcript(transcript, 4)
    return {
        'format': 'coe03-reference-v1', 'provenance': 'deterministic synthetic, not checkpoint activations',
        'soft_k5_width': 3, 'soft_k5_pairwise_distances': sorted({str(d5[i][j]) for i in range(5) for j in range(i)}),
        'demo_states': code.n, 'demo_soft_channels': code.k, 'demo_seed': DEMO_SEED,
        'demo_grid_bits': DEMO_GRID_BITS, 'demo_amplitude': str(code.amplitude),
        'demo_hamming_min': low, 'demo_hamming_max': high,
        'demo_l1_min': str(low * code.amplitude), 'demo_l1_max': str(high * code.amplitude),
        'demo_digest': code.digest(),
        'binary_one_shot_min_channels': DEMO_N - 1,
        'mux_n17_k4_rounds': len(transcript[0]),
    }


if __name__ == '__main__':
    import json
    print(json.dumps(reference_report(), indent=2, sort_keys=True))
