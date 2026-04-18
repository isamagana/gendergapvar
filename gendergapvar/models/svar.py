import numpy as np
import pandas as pd


def sample_invwishart(S, v):
    K = S.shape[0]
    L = np.linalg.cholesky(S)
    A = np.zeros((K, K))
    for i in range(K):
        A[i, i] = np.sqrt(np.random.chisquare(v - i))
        for j in range(i):
            A[i, j] = np.random.normal()
    X = np.linalg.solve(A, L.T)
    return X.T @ X


def build_matrices(data, p):
    T, K = data.shape
    Y = data[p:].T
    X = np.ones((1, T - p))
    for lag in range(1, p + 1):
        X = np.vstack([X, data[p-lag:T-lag].T])
    return Y, X


def estimate_ols(data, p):
    Y, X = build_matrices(data, p)
    T_eff = Y.shape[1]
    B_ols = Y @ X.T @ np.linalg.inv(X @ X.T)
    U_ols = Y - B_ols @ X
    S_ols = (U_ols @ U_ols.T) / T_eff
    return B_ols, S_ols, U_ols, X, T_eff


def draw_posterior(data, p, n_draws=2500):
    K = data.shape[1]
    B_ols, S_ols, U_ols, X, T_eff = estimate_ols(data, p)

    V_prior = np.linalg.inv(X @ X.T)
    B_prior = B_ols.copy()
    S_prior = S_ols.copy()
    v_prior = T_eff

    V_post  = np.linalg.inv(np.linalg.inv(V_prior) + X @ X.T)
    B_post  = V_post @ (np.linalg.inv(V_prior) @ B_prior + X @ U_ols.T + X @ (B_ols @ X).T).T
    v_post  = v_prior + T_eff
    S_post  = S_prior + U_ols @ U_ols.T

    posterior_Sigma = []
    posterior_B     = []

    for _ in range(n_draws):
        Sigma_d = sample_invwishart(S_post, v_post)
        B_d     = np.zeros_like(B_ols)
        for k in range(K):
            cov_k  = Sigma_d[k, k] * V_post
            B_d[k] = np.random.multivariate_normal(B_ols[k], cov_k)
        posterior_Sigma.append(Sigma_d)
        posterior_B.append(B_d)

    return np.array(posterior_Sigma), np.array(posterior_B), B_ols, S_ols


def check_column(col, s_col, z_col):
    for i in range(len(col)):
        if z_col[i] == 1 and abs(col[i]) > 1e-8:
            return False
        if s_col[i] > 0 and col[i] <= 0:
            return False
        if s_col[i] < 0 and col[i] >= 0:
            return False
    return True


def try_flip(col, s_col):
    flipped = col * -1
    for i in range(len(flipped)):
        if s_col[i] > 0 and flipped[i] <= 0:
            return col
        if s_col[i] < 0 and flipped[i] >= 0:
            return col
    return flipped


def find_rotation(Sigma, S, Z, n_candidates=1000):
    K        = Sigma.shape[0]
    n_shocks = S.shape[1]
    P        = np.linalg.cholesky(Sigma)

    for _ in range(n_candidates):
        Q      = np.linalg.qr(np.random.randn(K, K))[0]
        impact = P @ Q
        ok     = True
        for j in range(n_shocks):
            col          = try_flip(impact[:, j].copy(), S[:, j])
            impact[:, j] = col
            if not check_column(col, S[:, j], Z[:, j]):
                ok = False
                break
        if ok:
            return impact[:, :n_shocks]
    return None


def run_svar(data, p, S, Z, n_posterior=2500, n_accepted=500, n_candidates=1000):
    print(f"Drawing {n_posterior} posterior samples...")
    post_Sigma, post_B, B_ols, S_ols = draw_posterior(data, p, n_draws=n_posterior)
    print(f"Posterior draws complete.")

    print(f"Searching for valid rotations (target={n_accepted})...")
    accepted = []
    for d in range(n_posterior):
        impact = find_rotation(post_Sigma[d], S, Z, n_candidates=n_candidates)
        if impact is not None:
            accepted.append(impact)
        if len(accepted) >= n_accepted:
            break

    print(f"Accepted: {len(accepted)} out of {n_posterior}")
    return np.array(accepted) if accepted else None
