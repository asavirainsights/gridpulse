import numpy as np


def psi(reference, current, bins=10):
    """Population Stability Index between a reference and a current sample."""
    edges = np.quantile(reference, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    edges = np.unique(edges)
    ref_p = np.histogram(reference, edges)[0] / len(reference)
    cur_p = np.histogram(current, edges)[0] / len(current)
    ref_p = np.clip(ref_p, 1e-4, None)
    cur_p = np.clip(cur_p, 1e-4, None)
    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))