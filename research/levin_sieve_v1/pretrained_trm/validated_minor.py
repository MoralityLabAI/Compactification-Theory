"""LSV-13: rigorous forward-mode differentiation in FLINT Arb ball arithmetic.

The caller supplies *exact* binary FP32 checkpoint weights and an exact
binary FP32 stage-1 hidden state, both embedded in the real numbers. All
nonlinear operations and numerical rounding are enclosed by Arb balls.
The resulting 4x4 Jacobian minor is for the exact-real recurrence from z1,
not a formal verification of the FP32 or FP64 machine-code inference path.
"""
from __future__ import annotations

from dataclasses import dataclass

# Import here so a missing Arb runtime fails clearly instead of silently
# substituting floating-point approximations.
from flint import arb, arb_mat, ctx


@dataclass
class State:
    value: arb_mat  # [81,48]
    tangents: tuple[arb_mat, ...]  # 4 x [81,48]


def exact_matrix(tensor):
    """Convert a tensor's *binary* floating-point entries to exact Arb inputs."""
    return arb_mat([[arb(float(v)) for v in row] for row in tensor.detach().cpu().tolist()])


def transposed_weight(linear):
    return exact_matrix(linear.weight).transpose()


def repeated_bias(linear, nrows):
    b = [arb(float(v)) for v in linear.bias.detach().cpu().tolist()]
    return arb_mat([list(b) for _ in range(nrows)])


def _jet_linear(st: State, w, bias=None) -> State:
    v = st.value * w
    if bias is not None:
        v += bias
    return State(v, tuple(t * w for t in st.tangents))


def _jet_transpose(st):
    return State(st.value.transpose(), tuple(t.transpose() for t in st.tangents))


def _jet_add(a, b):
    return State(a.value+b.value, tuple(da+db for da,db in zip(a.tangents,b.tangents)))


def _jet_gelu(st: State) -> State:
    """Exact erf-based GELU and derivative, with Arb-enclosed rounding."""
    inv_sqrt_2 = arb(1) / arb(2).sqrt()
    inv_sqrt_2pi = arb(1) / (2*arb.pi()).sqrt()
    values = st.value.tolist()
    tangents = [t.tolist() for t in st.tangents]
    nr, nc = st.value.nrows(), st.value.ncols()
    out = [[] for _ in range(nr)]
    derivative_out = [[[] for _ in range(nr)] for _ in st.tangents]
    for i in range(nr):
        for j in range(nc):
            x = values[i][j]
            phi = (arb(1) + (x*inv_sqrt_2).erf()) / 2
            slope = phi + x * (-(x*x)/2).exp() * inv_sqrt_2pi
            out[i].append(x*phi)
            for k in range(len(tangents)):
                derivative_out[k][i].append(slope*tangents[k][i][j])
    return State(arb_mat(out),tuple(arb_mat(v) for v in derivative_out))


def _jet_layernorm(st: State, norm) -> State:
    """LayerNorm with population variance and analytic directional derivative.

    The exact-real interpretation of torch LayerNorm's default epsilon is the
    exact binary64 value of the Python epsilon attribute. The two passes use
    a shared mean/variance computed on the exact-real operation graph.
    """
    vals = st.value.tolist()
    tangents = [t.tolist() for t in st.tangents]
    nrows, d = st.value.nrows(), st.value.ncols()
    gamma = [arb(float(v)) for v in norm.weight.detach().cpu().tolist()]
    beta = [arb(float(v)) for v in norm.bias.detach().cpu().tolist()]
    eps = arb(float(norm.eps))
    normed = [[] for _ in range(nrows)]
    derivs = [[[] for _ in range(nrows)] for _ in tangents]
    for i in range(nrows):
        mean = sum(vals[i], arb(0)) / d
        centered = [v - mean for v in vals[i]]
        variance = sum((v*v for v in centered), arb(0)) / d
        invstd = arb(1) / (variance + eps).sqrt()
        for k, tangent in enumerate(tangents):
            row = tangent[i]
            avg = sum(row,arb(0))/d
            centered_d = [v-avg for v in row]
            dvar = 2 * sum((c*u for c,u in zip(centered,centered_d)),arb(0)) / d
            dinvstd = -(invstd**3)*dvar/2
            derivs[k][i] = [gamma[j]*(centered_d[j]*invstd+centered[j]*dinvstd)
                            for j in range(d)]
        normed[i] = [gamma[j]*centered[j]*invstd+beta[j] for j in range(d)]
    return State(arb_mat(normed),tuple(arb_mat(v) for v in derivs))


class CertifiedBlock:
    def __init__(self, block):
        self.t0 = transposed_weight(block.tok_mix[0])
        self.t0b = repeated_bias(block.tok_mix[0],48)
        self.t1 = transposed_weight(block.tok_mix[2])
        self.t1b = repeated_bias(block.tok_mix[2],48)
        self.c0 = transposed_weight(block.chan_mix[0])
        self.c0b = repeated_bias(block.chan_mix[0],81)
        self.c1 = transposed_weight(block.chan_mix[2])
        self.c1b = repeated_bias(block.chan_mix[2],81)
        self.ln1, self.ln2 = block.ln1, block.ln2

    def __call__(self, st: State) -> State:
        tokens = _jet_linear(_jet_transpose(st),self.t0,self.t0b)
        tokens = _jet_gelu(tokens)
        tokens = _jet_linear(tokens,self.t1,self.t1b)
        st = _jet_layernorm(_jet_add(st,_jet_transpose(tokens)),self.ln1)
        channels = _jet_linear(st,self.c0,self.c0b)
        channels = _jet_gelu(channels)
        channels = _jet_linear(channels,self.c1,self.c1b)
        return _jet_layernorm(_jet_add(st,channels),self.ln2)


def initial_state(stage1, columns) -> State:
    """Four unit tangent directions at pinned state coordinate indices."""
    if tuple(stage1.shape)!=(81,48) or len(columns)!=4 or len(set(columns))!=4:
        raise ValueError('require 81x48 stage1 and four distinct coordinates')
    if any(type(j) is not int or not 0<=j<81*48 for j in columns):
        raise ValueError('invalid hidden coordinate')
    base=exact_matrix(stage1)
    tangent=[]
    for j in columns:
        entries=[arb(0)]*(81*48)
        entries[j]=arb(1)
        tangent.append(arb_mat(81,48,entries))
    return State(base,tuple(tangent))


def head_pair_margin(st:State, head, cell:int, winner:int, rival:int):
    """A four-entry Jacobian row and an interval around a fixed digit gap."""
    if not (0<=cell<81 and 0<=winner<10 and 0<=rival<10 and winner!=rival):
        raise ValueError('invalid fixed output margin')
    w=head.weight.detach().cpu().double()
    b=head.bias.detach().cpu().double()
    diff=[arb(float(w[winner,j]))-arb(float(w[rival,j])) for j in range(48)]
    bias=arb(float(b[winner]))-arb(float(b[rival]))
    z=st.value.tolist()[cell]
    val=sum((x*d for x,d in zip(z,diff)),bias)
    out=[sum((x*d for x,d in zip(t.tolist()[cell],diff)),arb(0))
         for t in st.tangents]
    return val,out


def all_head_logits(st:State,head,cell:int):
    w=head.weight.detach().cpu().double()
    b=head.bias.detach().cpu().double()
    z=st.value.tolist()[cell]
    return [sum((arb(float(w[k,j]))*z[j] for j in range(48)),arb(float(b[k])))
            for k in range(10)]


def validated_minor(model,stage1,columns,first_missing,winner,rival,precision=128):
    """Ball-enclose all sixteen true derivatives and their determinant.

    Caller must lock checkpoint, exact stage1, columns, input cell and fixed
    margin class pair. A ball containing zero does not establish nonzero rank.
    """
    if not isinstance(precision,int) or precision<64:
        raise ValueError('at least 64 bits of ball precision required')
    ctx.prec=precision
    st=initial_state(stage1,columns)
    net=CertifiedBlock(model.block)
    rows=[];values=[]
    for k in range(4):
        v,gradient=head_pair_margin(st,model.head,first_missing,winner,rival)
        values.append(v)
        rows.append(gradient)
        if k<3:
            st=net(st)
    M=arb_mat(rows)
    determinant=M.det()
    logits=all_head_logits(st,model.head,first_missing)
    top_two_locked=all(not (logits[winner]-logits[j]).contains(0) and
                       logits[winner].lower()>logits[j].upper()
                       for j in range(10) if j!=winner)
    rival_is_second=all(logits[rival].lower()>logits[j].upper()
                        for j in range(10) if j!=winner and j!=rival)
    return {'M':M,'det':determinant,'rows':rows,'values':values,
            'rank_four_proved':not determinant.contains(0),
            'winning_class_verified':top_two_locked,
            'runner_up_verified':rival_is_second}


def serialized_ball_contains_original(ball, rendered):
    """Independently verify that a printable decimal ball encloses its Arb ball.

    This checks the *serialization* contract with exact Python rationals, not
    the numerical correctness of the Arb operation graph itself.
    """
    from fractions import Fraction
    from verify_serialized_minors import parse_ball
    lo,hi=parse_ball(rendered)
    def exact_binary_endpoint(endpoint):
        mantissa, exponent=endpoint.man_exp()
        m=int(mantissa)
        exponent=int(exponent)
        return Fraction(m*(2**exponent),1) if exponent>=0 else Fraction(m,2**(-exponent))
    return lo<=exact_binary_endpoint(ball.lower()) and exact_binary_endpoint(ball.upper())<=hi
