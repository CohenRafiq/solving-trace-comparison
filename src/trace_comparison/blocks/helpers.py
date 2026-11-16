import numpy as np
from scipy.stats import rankdata

def minmax(x, eps, axis=-1):
    lo = x.min(axis=axis, keepdims=True)
    rng = x.max(axis=axis, keepdims=True) - lo
    return np.where(rng > eps, (x - lo) / np.maximum(rng, eps), 0.0)

def rank(x, eps, axis=-1):
    out = (rankdata(x, axis=axis) - 1) / max(x.shape[axis] - 1, 1)
    constant = np.ptp(x, axis=axis, keepdims=True) <= eps
    return np.where(constant, 0.5, out)