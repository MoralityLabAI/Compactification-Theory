"""LSV-05: exact rational storage-rounding bound for a pretrained recurrence.

Derivations concern real-arithmetic evaluation of *stored* FP32/BF16 weights.
They do not certify arithmetic in torch kernels (roundoff, fused ops, GELU erf).
All matrix norms and weight perturbations are calculated as exact rationals
from the stored binary floating point tensors.
"""
from fractions import Fraction as Q
import math
import torch

GELU_LIP = Q(5,4)  # |d/dx(x Phi(x))| <= 1 + 1/sqrt(2*pi*e) < 5/4
NORM_DIM = 48
ROOT_DIM_UP = 7  # ceil(sqrt(48))
EPS_RECIP_ROOT_UP = 317  # >= 1/sqrt(eps), eps = 1e-5 from upstream LN
LAYER_NORM_CONST = Q((2+ROOT_DIM_UP)*EPS_RECIP_ROOT_UP)


def scalar(v):
    return Q.from_float(float(v))


def infinity_tensor(v):
    return max((abs(scalar(x)) for x in v.reshape(-1).tolist()), default=Q(0))


def row_sum_norm(v):
    assert v.ndim == 2
    return max((sum((abs(scalar(x)) for x in row),Q(0)) for row in v.tolist()),default=Q(0))


def row_delta_norm(a,b):
    assert a.shape == b.shape and a.ndim==2
    return max((sum((abs(scalar(x)-scalar(y)) for x,y in zip(r,s)),Q(0))
                for r,s in zip(a.tolist(),b.tolist())), default=Q(0))


def delta_infinity(a,b):
    assert a.shape == b.shape
    return max((abs(scalar(x)-scalar(y)) for x,y in zip(a.reshape(-1).tolist(),b.reshape(-1).tolist())),default=Q(0))


def affine_envelope(a, b, prefix, m_input, e_input):
    w1,w2=a[prefix+'.weight'],b[prefix+'.weight']
    bias1,bias2=a[prefix+'.bias'],b[prefix+'.bias']
    m_out = row_sum_norm(w1)*m_input + infinity_tensor(bias1)
    e_out = row_sum_norm(w2)*e_input + row_delta_norm(w1,w2)*m_input + delta_infinity(bias1,bias2)
    return m_out,e_out


def gelu_envelope(m,e):
    # |GELU(t)|<=|t|, Lipschitz<=5/4.
    return m, GELU_LIP*e


def layernorm_envelope(a,b,prefix,m_input,e_input):
    gamma,gamma_b=a[prefix+'.weight'],b[prefix+'.weight']
    bias,bias_b=a[prefix+'.bias'],b[prefix+'.bias']
    e = LAYER_NORM_CONST*infinity_tensor(gamma)*e_input + ROOT_DIM_UP*delta_infinity(gamma,gamma_b)+delta_infinity(bias,bias_b)
    m = ROOT_DIM_UP*infinity_tensor(gamma)+infinity_tensor(bias)
    return m,e


def block_norm_and_storage_delta(a,b,m_input):
    """Sound (conservative) bound for full block and BF16 weight-only perturbation."""
    m_tok,e_tok=affine_envelope(a,b,'block.tok_mix.0',m_input,Q(0))
    m_tok,e_tok=gelu_envelope(m_tok,e_tok)
    m_tok,e_tok=affine_envelope(a,b,'block.tok_mix.2',m_tok,e_tok)
    m_a,e_a=layernorm_envelope(a,b,'block.ln1',m_input+m_tok,e_tok)
    m_ch,e_ch=affine_envelope(a,b,'block.chan_mix.0',m_a,e_a)
    m_ch,e_ch=gelu_envelope(m_ch,e_ch)
    m_ch,e_ch=affine_envelope(a,b,'block.chan_mix.2',m_ch,e_ch)
    m_b,e_b=layernorm_envelope(a,b,'block.ln2',m_a+m_ch,e_a+e_ch)
    return m_b,e_b


def block_lipschitz(a):
    def lin(prefix):return row_sum_norm(a[prefix+'.weight'])
    ltok=lin('block.tok_mix.2')*GELU_LIP*lin('block.tok_mix.0')
    lchan=lin('block.chan_mix.2')*GELU_LIP*lin('block.chan_mix.0')
    ln1=LAYER_NORM_CONST*infinity_tensor(a['block.ln1.weight'])
    ln2=LAYER_NORM_CONST*infinity_tensor(a['block.ln2.weight'])
    return ln2*(1+lchan)*ln1*(1+ltok)


def fraction_description(x):
    if x==0:return {'upward_decimal':'0','log10_ceil':0}
    # Compute a conservative decimal *order* with integer arithmetic only.
    digits=len(str(x.numerator))-len(str(x.denominator))
    while x > Q(10)**digits: digits+=1
    # digits is a certified ceiling on log10(x); x may be <<1.
    return {'log10_upper_bound_integer':digits, 'value_float_approx':float(x) if digits<300 else None}


def certified_bounds(model):
    a={name:p.detach().cpu().float().contiguous() for name,p in model.named_parameters()}
    b={name:p.to(torch.bfloat16).float().cpu().contiguous() for name,p in a.items()}
    emb_init=infinity_tensor(a['token_emb.weight'])+infinity_tensor(a['pos_emb.weight'])
    emb_bf=infinity_tensor(b['token_emb.weight'])+infinity_tensor(b['pos_emb.weight'])
    e0=delta_infinity(a['token_emb.weight'],b['token_emb.weight'])+delta_infinity(a['pos_emb.weight'],b['pos_emb.weight'])
    bound_fp=ROOT_DIM_UP*infinity_tensor(a['block.ln2.weight'])+infinity_tensor(a['block.ln2.bias'])
    bound_bf=ROOT_DIM_UP*infinity_tensor(b['block.ln2.weight'])+infinity_tensor(b['block.ln2.bias'])
    uniform_m=max(emb_init,emb_bf,bound_fp,bound_bf)
    m1,storage_delta=block_norm_and_storage_delta(a,b,uniform_m)
    lip=block_lipschitz(a)
    e=e0
    for k in range(model.n_steps):
        e=lip*e+storage_delta
    # Head storage + latent bound error.
    head=a['head.weight'];head_b=b['head.weight']
    head_error=row_sum_norm(head)*e+row_delta_norm(head,head_b)*bound_bf+delta_infinity(a['head.bias'],b['head.bias'])
    return {
      'formal_scope':'ideal real arithmetic with quantized stored weights, not runtime roundoff',
      'max_block_gain_linf':fraction_description(lip),
      'block_gain_strict_contraction': lip < 1,
      'absolute_post_LN2_state_bound_original_linf':fraction_description(bound_fp),
      'absolute_post_LN2_state_bound_bfloat16_linf':fraction_description(bound_bf),
      'full_model_BF16_storage_logit_error_linf_upper':fraction_description(head_error),
      'final_hidden_error_linf_upper':fraction_description(e),
      'embeddings_storage_error_linf_upper':fraction_description(e0),
      'global_block_storage_error_linf_upper':fraction_description(storage_delta),
      'strict_numerical_certificate':True,
      'constants':{'GELU_derivative_bound':'5/4','LN_max_normalized_coordinate':'7','LN_input_linf_L_bound':'2853 * max_abs_ln_gamma','LN_eps':'1e-5'},
    }
