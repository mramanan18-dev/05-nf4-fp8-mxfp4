"""
EdgeLayer, Blog 5: Smarter rulers (NF4, FP8, MXFP4)

Synthetic demo. Every format is simulated in NumPy ("fake quant"):
values are rounded onto each format's grid and turned back into floats,
so we can measure the rounding error without special hardware.

Tests
  A. Gaussian weights, 4-bit:     INT4 · FP4 · NF4 · MXFP4 · NVFP4
  B. Heavy-tailed weights, 4-bit: same set
  C. Activations with hot channels, 8-bit: INT8 vs FP8 (E4M3), per-tensor
  D. Real bits per weight once the scales are counted

Run: python edgelayer-05-formats-demo.py
"""
import numpy as np
from scipy.stats import norm

rng = np.random.default_rng(0)


# ----------------------------------------------------------------------------
# 1. The rulers (each format's set of allowed values)
# ----------------------------------------------------------------------------

def int4_grid():
    # symmetric INT4 with absmax scaling: -7 .. 7, evenly spaced
    return np.arange(-7, 8) / 7.0


def nf4_grid():
    # NF4 as built in bitsandbytes (create_normal_map): quantiles of N(0,1),
    # 8 positive levels, 7 negative levels, plus an exact zero, scaled to [-1, 1]
    offset = 0.9677083
    pos = norm.ppf(np.linspace(offset, 0.5, 9)[:-1])
    neg = -norm.ppf(np.linspace(offset, 0.5, 8)[:-1])
    v = np.sort(np.concatenate([pos, [0.0], neg]))
    return v / np.abs(v).max()


def fp4_e2m1_grid():
    # FP4 E2M1: 1 sign, 2 exponent, 1 mantissa bit
    mag = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    return np.unique(np.concatenate([-mag, mag]))


def round_to_grid(x, grid):
    # snap every value to the nearest level in a sorted grid
    idx = np.clip(np.searchsorted(grid, x), 1, len(grid) - 1)
    lo, hi = grid[idx - 1], grid[idx]
    return np.where(np.abs(x - lo) <= np.abs(hi - x), lo, hi)


def round_fp(x, e_bits, m_bits, bias, max_val):
    # round to a small float format (handles subnormals, saturates at max)
    x = np.asarray(x, dtype=np.float64)
    a = np.abs(x)
    min_exp = 1 - bias
    exp = np.floor(np.log2(np.maximum(a, 1e-300)))
    exp = np.maximum(exp, min_exp)
    step = 2.0 ** (exp - m_bits)          # gap between ticks at this size
    q = np.round(a / step) * step
    return np.sign(x) * np.minimum(q, max_val)


def fp8_e4m3(x):
    return round_fp(x, 4, 3, 7, 448.0)


def fp8_e5m2(x):
    return round_fp(x, 5, 2, 15, 57344.0)


# ----------------------------------------------------------------------------
# 2. Quantizers (block of weights -> scale -> snap -> unscale)
# ----------------------------------------------------------------------------

def blocks(w, size):
    return w.reshape(-1, size)


def q_absmax(w, grid, block, scale_fmt="fp16"):
    """Scale each block so its largest value hits the grid's edge, snap, unscale."""
    b = blocks(w, block)
    scale = np.abs(b).max(axis=1, keepdims=True) / np.abs(grid).max()
    scale = np.where(scale == 0, 1.0, scale)
    return (round_to_grid(b / scale, grid) * scale).reshape(w.shape)


def q_mxfp4(w, block=32):
    """OCP MXFP4: 32 E2M1 values share one power-of-two (E8M0) scale."""
    b = blocks(w, block)
    amax = np.abs(b).max(axis=1, keepdims=True)
    shared_exp = np.floor(np.log2(np.maximum(amax, 1e-30))) - 2   # 2 = E2M1's top exponent
    scale = 2.0 ** shared_exp
    q = round_to_grid(np.clip(b / scale, -6, 6), fp4_e2m1_grid())
    return (q * scale).reshape(w.shape)


def q_nvfp4(w, block=16):
    """NVFP4: 16 E2M1 values share one FP8 (E4M3) scale, plus one FP32 scale per tensor."""
    b = blocks(w, block)
    t_scale = np.abs(w).max() / (448.0 * 6.0)          # keeps block scales inside E4M3's range
    blk = np.abs(b).max(axis=1, keepdims=True) / 6.0 / t_scale
    blk = fp8_e4m3(blk)
    blk = np.where(blk == 0, 1.0, blk)
    scale = blk * t_scale
    q = round_to_grid(np.clip(b / scale, -6, 6), fp4_e2m1_grid())
    return (q * scale).reshape(w.shape)


FORMATS = {
    "INT4 (block 64)":   lambda w: q_absmax(w, int4_grid(), 64),
    "FP4 E2M1 (block 64)": lambda w: q_absmax(w, fp4_e2m1_grid(), 64),
    "NF4 (block 64)":    lambda w: q_absmax(w, nf4_grid(), 64),
    "MXFP4 (block 32)":  q_mxfp4,
    "NVFP4 (block 16)":  q_nvfp4,
}


def rel_err(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(a)


# ----------------------------------------------------------------------------
# 3. A small synthetic layer, same shape as Blogs 3 and 4
# ----------------------------------------------------------------------------

d_out, d_in, n_tok = 256, 512, 2048

def correlated_inputs(n):
    mix = rng.normal(size=(d_in, d_in)) / np.sqrt(d_in)
    base = rng.normal(size=(n, d_in))
    return (0.6 * base + 0.8 * base @ mix).T          # (d_in, n)

X = correlated_inputs(n_tok)

def run_weights(W, label):
    print(f"\n{label}")
    print(f"  {'format':<22}{'weight err':>12}{'output err':>12}")
    res = {}
    for name, fn in FORMATS.items():
        Wq = fn(W)
        we, oe = rel_err(W, Wq), rel_err(W @ X, Wq @ X)
        res[name] = (we, oe)
        print(f"  {name:<22}{we:>11.1%}{oe:>12.1%}")
    return res


# ----------------------------------------------------------------------------
# Test A: bell-shaped weights (what trained weights mostly look like)
# ----------------------------------------------------------------------------
W_gauss = rng.normal(0, 0.02, size=(d_out, d_in))
res_A = run_weights(W_gauss, "Test A: Gaussian weights, 4-bit")

# ----------------------------------------------------------------------------
# Test B: heavier tails (a few weights far from the rest)
# ----------------------------------------------------------------------------
W_heavy = 0.02 * rng.standard_t(df=4, size=(d_out, d_in)) / np.sqrt(2.0)
res_B = run_weights(W_heavy, "Test B: heavy-tailed weights (Student-t, df=4), 4-bit")


# ----------------------------------------------------------------------------
# Test C: the Blog 2 problem. Activations with a few very hot channels.
# One scale for the whole tensor, INT8 vs FP8 E4M3.
# ----------------------------------------------------------------------------
n_ch, n_t = 512, 2048
A = rng.normal(0, 1.0, size=(n_t, n_ch))
hot = rng.choice(n_ch, size=5, replace=False)          # 1% of channels
A[:, hot] *= 100.0                                     # ~100x hotter, as in Blog 2
normal_ch = np.setdiff1d(np.arange(n_ch), hot)

# INT8, symmetric per-tensor: the hot channels set the scale
s_int8 = np.abs(A).max() / 127.0
A_int8 = np.clip(np.round(A / s_int8), -127, 127) * s_int8

# FP8 E4M3, per-tensor: largest value mapped to 448
s_fp8 = np.abs(A).max() / 448.0
A_fp8 = fp8_e4m3(A / s_fp8) * s_fp8

int8_zero = np.mean(A_int8[:, normal_ch] == 0)
fp8_zero = np.mean(A_fp8[:, normal_ch] == 0)

print("\nTest C: activations with 5 hot channels (100x), per-tensor scale")
print(f"  {'':<10}{'all values':>12}{'ordinary channels':>20}{'ordinary values -> 0':>22}")
for name, Aq, z in [("INT8", A_int8, int8_zero), ("FP8 E4M3", A_fp8, fp8_zero)]:
    print(f"  {name:<10}{rel_err(A, Aq):>11.1%}"
          f"{rel_err(A[:, normal_ch], Aq[:, normal_ch]):>20.1%}{z:>21.1%}")


# ----------------------------------------------------------------------------
# Test D: what "4-bit" really costs once the scales are stored
# ----------------------------------------------------------------------------
bpw = {
    "INT4, group 128, FP16 scale + zero": 4 + 32 / 128,
    "NF4, block 64, FP32 scale":           4 + 32 / 64,
    "NF4 + double quant":                  4 + 8 / 64 + 32 / (64 * 256),
    "MXFP4, block 32, 8-bit scale":        4 + 8 / 32,
    "NVFP4, block 16, FP8 scale":          4 + 8 / 16,     # + one FP32 per tensor
    "FP8 (per-channel scale)":             8.0,
}
print("\nTest D: bits per weight, scales included")
for k, v in bpw.items():
    print(f"  {k:<38}{v:>7.3f}")


# ----------------------------------------------------------------------------
# Handy printouts used in the post
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    print("\nNF4 levels:     ", nf4_grid())
    print("FP4 E2M1 levels:", fp4_e2m1_grid())
    # share of N(0,1) mass that INT4's outer ticks serve (scaled so 3.0 sigma = edge)
    edge = 3.0
    g = int4_grid() * edge
    cells = (g[1:] + g[:-1]) / 2
    print("INT4 ticks beyond +/-2 sigma:", int(np.sum(np.abs(g) > 2)),
          "of 15, holding", f"{2 * (1 - norm.cdf(2)) :.1%}", "of the weights")
