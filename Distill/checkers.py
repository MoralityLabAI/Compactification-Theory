"""Independent toy-case oracles. These do not grade narrative explanations."""
def pluto_count(case, fields):
    if case is None or case["kind"] != "equal_weight_fronts":
        raise ValueError("wrong case kind")
    d = case["data"]
    if set(d) != {"fronts", "deadline", "command_ticks", "switch_ticks"}:
        raise ValueError("wrong case keys")
    n, D, c, s = (d[k] for k in ("fronts", "deadline", "command_ticks", "switch_ticks"))
    if any(type(x) is not int for x in (n, D, c, s)) or not (1 <= n <= 12 and 0 <= D <= 1000 and 1 <= c <= 1000 and 0 <= s <= 1000):
        raise ValueError("outside toy bounds")
    # Independent enumeration of command counts rather than use the closed form.
    best = max(k for k in range(n + 1) if k == 0 or k*c+(k-1)*s <= D)
    return fields["answer"].strip() == str(best)

def cycle_bits(case, fields):
    if case is None or case["kind"] != "cycle_observer":
        raise ValueError("wrong case kind")
    d = case["data"]
    if set(d) != {"vertices"} or type(d["vertices"]) is not int or not 3 <= d["vertices"] <= 12:
        raise ValueError("outside toy bounds")
    n = d["vertices"]
    # An exhaustive 1-bit assignment check; 2 bits always suffice for a cycle.
    two_colors = any(all(((mask >> i) & 1) != ((mask >> ((i+1)%n)) & 1)
                         for i in range(n)) for mask in range(1 << n))
    return fields["answer"].strip() == ("1" if two_colors else "2")

def example_checkers(packet):
    """Use only explicitly supported public fixtures, never dynamically import from packets."""
    case = packet["public_case"]
    if case is None:
        return {}
    key = (packet["gym_id"], case["kind"])
    lookup = {("IRC-PG-01", "equal_weight_fronts"): pluto_count,
              ("COE-01", "cycle_observer"): cycle_bits}
    return {"case.solve": lookup[key]} if key in lookup else {}
