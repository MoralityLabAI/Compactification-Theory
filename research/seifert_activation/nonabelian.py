"""Exact nonabelian braid witnesses for cyclic activation trajectories.

A closed pure braid has an abelianized pairwise-winding signature. For 3+
strands, commutator braids can have every winding zero while remaining
nonidentity. We use Artin's faithful action on the rank-n free group to
compare BASED braids at fixed strand basepoints.

This automorphism is not a closed-link invariant: conjugate braids can
close to the same link, and Markov stabilizations change strand count.
Images are reduced free words in generators x_1,...,x_n, with signed ints
representing generators/inverses. A size cap prevents unbounded growth.
"""
from __future__ import annotations

from hashlib import sha256
import json
from typing import Iterable, Sequence

FreeWord = tuple[int, ...]
Automorphism = tuple[FreeWord, ...]


class ArtinWordBudgetExceeded(ValueError):
    """Automorphism image exceeded its configured word-length budget."""


def free_reduce(letters: Iterable[int]) -> FreeWord:
    result: list[int] = []
    for x in letters:
        if x == 0:
            raise ValueError("Free-group generator zero is invalid")
        if result and result[-1] == -x:
            result.pop()
        else:
            result.append(int(x))
    return tuple(result)


def inverse_free_word(word: Sequence[int]) -> FreeWord:
    return tuple(-x for x in reversed(word))


def generator_action(n: int, index: int) -> Automorphism:
    """Artin sigma_i: xi -> xi x(i+1) xi^-1; x(i+1) -> xi."""
    if n < 2 or index == 0 or abs(index) >= n:
        raise ValueError(f"Generator {index} invalid for braid group B_{n}")
    i = abs(index)
    images = [(j,) for j in range(1, n + 1)]
    if index > 0:
        images[i - 1] = (i, i + 1, -i)
        images[i] = (i,)
    else:
        images[i - 1] = (i + 1,)
        images[i] = (-(i + 1), i, i + 1)
    return tuple(images)


def substitute(word: Sequence[int], images: Automorphism, *, max_letters: int) -> FreeWord:
    """Apply an automorphism to a word, canceling adjacent inverses."""
    result: list[int] = []
    n = len(images)
    for letter in word:
        if not 1 <= abs(letter) <= n:
            raise ValueError("Free group word outside basis")
        image = images[abs(letter) - 1]
        for x in image if letter > 0 else inverse_free_word(image):
            if result and result[-1] == -x:
                result.pop()
            else:
                result.append(x)
            if len(result) > max_letters:
                raise ArtinWordBudgetExceeded(f"Artin image exceeds {max_letters} letters")
    return tuple(result)


def compose(left: Automorphism, right: Automorphism, *, max_letters: int = 200_000) -> Automorphism:
    """Compose actions left after right: (left ∘ right)(x_i)."""
    if len(left) != len(right):
        raise ValueError("Automorphisms must share a free-group rank")
    return tuple(substitute(word, left, max_letters=max_letters) for word in right)


def artin_action(n: int, braid: Sequence[int], *, max_letters: int = 200_000) -> Automorphism:
    """Faithful Artin action of the based braid, with exact free reduction."""
    if n < 2 or max_letters < 1:
        raise ValueError("Require n>=2 and a positive word budget")
    images: Automorphism = tuple((i,) for i in range(1, n + 1))
    for letter in braid:
        images = compose(images, generator_action(n, int(letter)), max_letters=max_letters)
    return images


def artin_witness(n: int, braid: Sequence[int], *, max_letters: int = 200_000) -> dict:
    """Report exact automorphism images or an explicit size error."""
    images = artin_action(n, braid, max_letters=max_letters)
    identity = tuple((i,) for i in range(1, n + 1))
    canonical = json.dumps(images, separators=(",", ":"))
    return {
        "representation": "faithful_artin_action_on_Fn",
        "is_identity": images == identity,
        "images": [list(word) for word in images],
        "image_lengths": [len(word) for word in images],
        "sha256": sha256(canonical.encode("ascii")).hexdigest(),
        "scope": "based_braid_at_fixed_labeled_basepoints",
    }
