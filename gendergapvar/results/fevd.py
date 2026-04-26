import numpy as np


def compute_fevd(irf_array, n_periods=40):
    K        = irf_array.shape[1]
    n_shocks = irf_array.shape[2]
    fevd     = np.zeros((n_periods + 1, K, n_shocks))
    for h in range(n_periods + 1):
        mse = np.zeros((K, n_shocks))
        for t in range(h + 1):
            for j in range(n_shocks):
                mse[:, j] += irf_array[t, :, j] ** 2
        total = mse.sum(axis=1, keepdims=True)
        total = np.where(total == 0, 1, total)
        fevd[h] = mse / total
    return fevd


def compute_fevd_distribution(irf_draws, n_periods=40):
    all_fevds = []
    for irf in irf_draws:
        fevd = compute_fevd(irf, n_periods)
        all_fevds.append(fevd)
    all_fevds = np.array(all_fevds)
    median = np.median(all_fevds, axis=0)
    p16    = np.percentile(all_fevds, 16, axis=0)
    p84    = np.percentile(all_fevds, 84, axis=0)
    return median, p16, p84