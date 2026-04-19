import numpy as np


def compute_irf(B_var, impact, n_periods=40):
    """
    Compute Impulse Response Functions.
    
    Parameters:
    -----------
    B_var  : VAR coefficient matrix (K x Kp+1)
    impact : impact matrix B (K x n_shocks)
    n_periods : number of periods
    
    Returns:
    --------
    irf : array of shape (n_periods+1, K, n_shocks)
    """
    K        = impact.shape[0]
    n_shocks = impact.shape[1]
    p        = (B_var.shape[1] - 1) // K

    # Extract lag matrices A1, A2, ..., Ap
    A = []
    for i in range(p):
        A.append(B_var[:, 1 + i*K : 1 + (i+1)*K])

    # Initialize IRF array
    irf = np.zeros((n_periods + 1, K, n_shocks))

    # Period 0: impact
    irf[0] = impact

    # Periods 1 to n_periods
    for t in range(1, n_periods + 1):
        for lag in range(min(t, p)):
            irf[t] += A[lag] @ irf[t - lag - 1]

    return irf


def compute_irf_distribution(B_var, accepted_draws, n_periods=40):
    """
    Compute IRF distribution across accepted SVAR draws.
    
    Returns median, 16th and 84th percentiles.
    """
    n_draws  = len(accepted_draws)
    K        = accepted_draws[0].shape[0]
    n_shocks = accepted_draws[0].shape[1]

    all_irfs = np.zeros((n_draws, n_periods + 1, K, n_shocks))

    for d, impact in enumerate(accepted_draws):
        all_irfs[d] = compute_irf(B_var, impact, n_periods)

    median = np.median(all_irfs, axis=0)
    p16    = np.percentile(all_irfs, 16, axis=0)
    p84    = np.percentile(all_irfs, 84, axis=0)

    return median, p16, p84


def compute_gpg_irf(irf_array):
    """
    Compute GPG IRF as difference between male and female wage IRFs.
    GPG = (wage_m - wage_f) / wage_m
    In logs: GPG_irf ≈ irf_wage_m - irf_wage_f
    
    Assumes variables are ordered:
    [wage_m, wage_f, unemp_m, unemp_f, lfp_m, lfp_f, gdp, cpi]
    """
    return irf_array[:, 0, :] - irf_array[:, 1, :]
