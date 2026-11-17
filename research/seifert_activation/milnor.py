"""Exact Milnor triple linking for oriented, labeled three-component braid closures.

This module computes the length-three Milnor invariant mu_123 of a *pure*
three-braid closure via normalized Artin longitudes and degree-two Magnus
coefficients. When all pairwise linking numbers vanish, mu_123 is a genuine
integer invariant of the oriented, ordered link, independent of braid basepoint
and pure conjugation. In general it is only defined modulo the gcd of pairwise
linking numbers. Based Artin actions and their hashes are separate observables.

All input words, free-group actions, linking numbers and Magnus coefficients are
computed over integers. The result pertains to a link in the measured planar
chart; no 4D surface-isotopy conclusion follows from it.
"""
from __future__ import annotations

import math
from typing import Sequence

from nonabelian import ArtinWordBudgetExceeded, artin_action, free_reduce


class PureThreeBraidRequired(ValueError):
    """The triple Milnor invariant requires a pure three-strand braid."""


def pure_linking_numbers(n: int, word: Sequence[int]) -> dict[tuple[int, int], int]:
    """Oriented linking numbers of the ordered closure, from braid crossings.

    The first original strand has label 1; after a crossing, the two strands
    exchange positions. An oriented linking number is half their signed
    crossing count. A pure braid returns each original label to its slot.
    """
    if n < 2:
        raise ValueError("Need at least two strands")
    order = list(range(1, n + 1))
    crossings: dict[tuple[int, int], int] = {
        (i, j): 0 for i in range(1, n + 1) for j in range(i + 1, n + 1)
    }
    for generator in word:
        k = abs(int(generator))
        if generator == 0 or k >= n:
            raise ValueError(f"Invalid braid generator {generator} for B_{n}")
        a, b = order[k - 1], order[k]
        crossings[tuple(sorted((a, b)))] += 1 if generator > 0 else -1
        order[k - 1], order[k] = b, a
    if order != list(range(1, n + 1)):
        raise PureThreeBraidRequired("Braid permutation is not identity")
    if any(count % 2 for count in crossings.values()):
        raise ArithmeticError("Pure braid has an odd pairwise signed crossing sum")
    return {pair: count // 2 for pair, count in crossings.items()}


def normalized_conjugator(image: Sequence[int], strand: int, *, max_letters: int = 20000) -> tuple[int, ...]:
    """Unique xi-exponent-zero u with image of xi = u xi u^-1.

    Every pure braid Artin action sends x_i to a conjugate of x_i. Strip
    inverse outer letters until only x_i remains, then remove all x_i
    exponent from the conjugator by right-multiplying by x_i^-k.
    """
    if strand <= 0:
        raise ValueError("Strand index is positive")
    letters = tuple(image)
    if not letters or len(letters) > max_letters:
        raise ArtinWordBudgetExceeded("Image empty or exceeds longitude budget")
    left, right = 0, len(letters) - 1
    while left < right and letters[left] == -letters[right]:
        left += 1
        right -= 1
    if left != right or letters[left] != strand:
        raise ValueError("Image is not a conjugate of the correct strand meridian")
    prefix = letters[:left]
    power = sum((1 if x > 0 else -1) for x in prefix if abs(x) == strand)
    extension = ((-strand,) * power if power > 0 else (strand,) * (-power))
    out = free_reduce(prefix + extension)
    if len(out) > max_letters:
        raise ArtinWordBudgetExceeded("Normalized longitude exceeds budget")
    return out


def mixed_magnus_coefficient(word: Sequence[int], first: int, second: int) -> int:
    """Exact coefficient [X_first X_second] of Magnus expansion, first != second.

    For letters x_k^eps, the degree-one term is eps*X_k. The quadratic
    term of x_k^-1 is X_k^2, which never contributes to a *mixed* pair.
    Hence sum_{p<q} eps_p eps_q 1[k_p=first, k_q=second].
    """
    if first <= 0 or second <= 0 or first == second:
        raise ValueError("Two distinct positive free-group indices required")
    prefix_exponent = 0
    coefficient = 0
    for x in word:
        if not x:
            raise ValueError("Invalid zero free-group letter")
        sign = 1 if x > 0 else -1
        if abs(x) == second:
            coefficient += sign * prefix_exponent
        if abs(x) == first:
            prefix_exponent += sign
    return coefficient


def triple_milnor(n: int, word: Sequence[int], *, max_letters: int = 20000) -> dict:
    """Milnor mu_123 for a pure three-braid, with exact closure indeterminacy.

    Our Artin convention sends xi -> u_i xi u_i^-1. The raw triple is
    [X1 X2] Magnus(u_3); cyclic checks [X2 X3](u_1) and [X3 X1](u_2).
    For zero pairwise linking, all three must equal as integers. For other
    links, they agree modulo gcd(lk12,lk13,lk23).
    """
    if n != 3:
        raise PureThreeBraidRequired("SCM-04 implements ordered three-component closures only")
    if max_letters < 1:
        raise ValueError("Positive longitude budget required")
    linking = pure_linking_numbers(n, word)
    images = artin_action(n, word, max_letters=max_letters)
    longitudes = [normalized_conjugator(img, i + 1, max_letters=max_letters)
                  for i, img in enumerate(images)]
    cyclic = (
        mixed_magnus_coefficient(longitudes[2], 1, 2),
        mixed_magnus_coefficient(longitudes[0], 2, 3),
        mixed_magnus_coefficient(longitudes[1], 3, 1),
    )
    divisor = math.gcd(*linking.values())
    residues = [v if divisor == 0 else v % divisor for v in cyclic]
    if len(set(residues)) != 1:
        raise ArithmeticError("Cyclic Milnor coefficients disagree modulo indeterminacy")
    return {
        "status": "exact_integer" if divisor == 0 else "modulo_indeterminacy",
        "invariant": "ordered_oriented_closed_link_milnor_mu123",
        "pairwise_linking": {f"{i},{j}": value for (i, j), value in linking.items()},
        "indeterminacy_gcd": divisor,
        "mu123": residues[0],
        "raw_cyclic_coefficients": list(cyclic),
        "longitude_lengths": [len(w) for w in longitudes],
        "convention": "Artin xi->ui xi ui^-1, Magnus [X1 X2] of u3",
    }


def token_order_milnor(frames: Sequence[Sequence[Sequence[float]]],
                       braid_word: Sequence[int], *, max_letters: int = 20000) -> dict:
    """Turn x-rank-labeled mu_123 into fixed *token-identity* order.

    `extract_closed_braid` enumerates strands by increasing x at the first
    frame. Moving the angular cut can permute those x ranks. The triple
    invariant is alternating under component permutation (exactly with zero
    pairwise linking, modulo the usual gcd otherwise). We restore original
    strand identities, corresponding to the recorded token positions.

    This correction is essential when changing the intervention-cycle start
    angle or when the registered readout chart reorders its x positions.
    """
    if len(frames) < 2 or len(frames[0]) != 3:
        raise PureThreeBraidRequired("Exactly three tracked strands required")
    order = sorted(range(3), key=lambda i: float(frames[0][i][0]))
    if len({float(frames[0][i][0]) for i in range(3)}) != 3:
        raise ValueError("Ambiguous start-frame x ranking")
    swaps = sum(order[i] > order[j] for i in range(3) for j in range(i + 1, 3))
    sign = -1 if swaps % 2 else 1
    linking = pure_linking_numbers(3, braid_word)
    if all(v == 0 for v in linking.values()):
        result = triple_milnor_nilpotent(3, braid_word)
    else:
        result = triple_milnor(3, braid_word, max_letters=max_letters)
    mu = result['mu123'] * sign
    divisor = result['indeterminacy_gcd']
    if divisor:
        mu %= divisor
    return {**result,
            'mu123_position_order': result['mu123'],
            'mu123_token_order': mu,
            'position_to_token_strands': order,
            'token_order_permutation_sign': sign,
            'token_order_convention': 'ascending original tracked token indices'}


def _cubic_mul(a: dict[tuple[int, ...], int],
               b: dict[tuple[int, ...], int]) -> dict[tuple[int, ...], int]:
    """Noncommutative multiplication in Z<X1,X2,X3> / (degree >= 4)."""
    result: dict[tuple[int, ...], int] = {}
    for left, x in a.items():
        for right, y in b.items():
            if len(left) + len(right) <= 3:
                word = left + right
                result[word] = result.get(word, 0) + x * y
    return {word: c for word, c in result.items() if c}


def _cubic_inverse(a: dict[tuple[int, ...], int]) -> dict[tuple[int, ...], int]:
    """(1+q)^-1 = 1-q+q²-q³, exactly up to degree three."""
    if a.get(()) != 1:
        raise ValueError("Magnus group series must have constant coefficient one")
    q = {w: c for w, c in a.items() if w}
    sq = _cubic_mul(q, q)
    cube = _cubic_mul(sq, q)
    inv: dict[tuple[int, ...], int] = {(): 1}
    for polynomial, sign in ((q, -1), (sq, 1), (cube, -1)):
        for word, value in polynomial.items():
            inv[word] = inv.get(word, 0) + sign * value
    return {word: c for word, c in inv.items() if c}


def cubic_artin_images(n: int, braid: Sequence[int]) -> tuple[dict[tuple[int, ...], int], ...]:
    """Exact Artin images in the degree <= 3 Magnus quotient; O(word length).

    For fixed n=3 and degree 3, each image has at most 40 monomial
    coordinates (1 + 3 + 9 + 27). Group words can grow exponentially,
    while this quotient remains bounded in dimension; integer coefficients
    themselves can grow and are stored without truncating their bit lengths.
    """
    if n != 3:
        raise ValueError("The fixed cubic implementation uses B_3")
    images = [{(): 1, (i,): 1} for i in range(1, 4)]
    for generator in braid:
        k = abs(int(generator))
        if generator == 0 or not (1 <= k <= 2):
            raise ValueError(f"Invalid generator {generator} for B_3")
        i = k - 1
        a, b = images[i], images[i + 1]
        if generator > 0:
            images[i] = _cubic_mul(_cubic_mul(a, b), _cubic_inverse(a))
            images[i + 1] = a
        else:
            images[i] = b
            images[i + 1] = _cubic_mul(_cubic_mul(_cubic_inverse(b), a), b)
    return tuple(images)


def triple_milnor_nilpotent(n: int, braid: Sequence[int]) -> dict:
    """Exact closure mu123 at ZERO pairwise linking, without long free words.

    If all pairwise linking numbers vanish, the normalized Artin
    conjugators u_i belong to the free commutator subgroup. Write their
    Magnus expansion 1 + U_i^(2) + O(3). Then
      theta(u_i x_i u_i^-1) = 1+Xi + U_i^(2) Xi - Xi U_i^(2) + O(4).
    For i,j,k distinct, coefficient [Xi Xj Xk] of theta(phi(x_k))
    therefore equals the triple-linking coefficient [Xi Xj] of u_k.
    """
    if n != 3:
        raise PureThreeBraidRequired("SCM-04 cubic Milnor works on B_3 only")
    linking = pure_linking_numbers(n, braid)
    if any(linking.values()):
        raise ValueError("Cubic shortcut requires zero pairwise linking")
    images = cubic_artin_images(3, braid)
    cyclic = (
        images[2].get((1, 2, 3), 0),
        images[0].get((2, 3, 1), 0),
        images[1].get((3, 1, 2), 0),
    )
    if len(set(cyclic)) != 1:
        raise ArithmeticError("Cyclic cubic Magnus coefficients disagree")
    return {
        "status": "exact_integer",
        "method": "degree_three_nilpotent_magnus_artin",
        "invariant": "ordered_oriented_closed_link_milnor_mu123",
        "pairwise_linking": {f"{i},{j}": v for (i, j), v in linking.items()},
        "indeterminacy_gcd": 0,
        "mu123": cyclic[0],
        "raw_cyclic_coefficients": list(cyclic),
        "magnus_nonzero_terms": [len(image) for image in images],
        "convention": "Artin xi->ui xi ui^-1, Magnus [X1 X2] of u3",
    }
