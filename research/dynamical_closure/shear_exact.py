"""Exact exhaustive quotient and conditional Bayes risks for cyclic shear."""
from fractions import Fraction
from cyclic_addition import onehot_risk, states


def present(a, b, p=31):
    return (a + b) % p


def shear(a, b, p=31):
    return (a, (a + b) % p)


def after_shear(a, b, p=31):
    return (2 * a + b) % p


def exact_bits(p=31):
    """Minimum fixed-length bits for present task and for both observations."""
    if p < 2:
        raise ValueError("p must be >=2")
    return (p - 1).bit_length(), (p * p - 1).bit_length()


def oracle_code(a, b, budget, p=31):
    """Nested present-first partitions, at most budget bits, 0<=B<=2L."""
    l = (p - 1).bit_length()
    if not 0 <= budget <= 2 * l:
        raise ValueError("budget outside oracle path")
    s = present(a, b, p)
    if budget <= l:
        return (2**budget * s // p,)
    return (s, 2**(budget - l) * a // p)


def balanced_code(a, b, budget, p=31):
    """Nested, roughly balanced bins of present and post-shear labels."""
    l = (p - 1).bit_length()
    if not 0 <= budget <= 2*l:
        raise ValueError("budget outside balanced path")
    r = (budget + 1) // 2
    return (2**r * present(a, b, p) // p,
            2**(budget-r) * after_shear(a, b, p) // p)


def risks(code, p=31):
    """Exact uniform-state one-hot squared Bayes risks on realized code fibers."""
    xs = states(p)
    if len(code) != len(xs):
        raise ValueError("one code per microscopic state is required")
    r0 = onehot_risk(code, [present(*x, p) for x in xs])
    r1 = onehot_risk(code, [after_shear(*x, p) for x in xs])
    return r0, r1


def balanced_curve(p=31):
    l = (p - 1).bit_length()
    result = []
    for budget in range(2*l + 1):
        r = (budget+1)//2
        code = tuple(balanced_code(a,b,budget,p) for a,b in states(p))
        observed = risks(code,p)
        expected = (Fraction(p-min(2**r,p),p),
                    Fraction(p-min(2**(budget-r),p),p))
        if observed != expected:
            raise AssertionError((budget, observed, expected))
        result.append({"bits":budget, "codes":len(set(code)),
                       "present_risk":str(observed[0]), "after_shear_risk":str(observed[1])})
    return result


def oracle_curve(p=31):
    l = (p - 1).bit_length()
    result = []
    for budget in range(2*l + 1):
        code = tuple(oracle_code(a, b, budget, p) for a, b in states(p))
        observed = risks(code, p)
        if budget <= l:
            expected = (Fraction(p - min(2**budget, p), p), Fraction(p-1,p))
        else:
            expected = (Fraction(0), Fraction(p - min(2**(budget-l), p), p))
        if observed != expected:
            raise AssertionError((budget, observed, expected))
        result.append({"bits": budget, "codes": len(set(code)),
                       "present_risk": str(observed[0]), "after_shear_risk": str(observed[1])})
    return result
