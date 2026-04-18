import numpy as np


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
    Y, X   = build_matrices(data, p)
    T_eff  = Y.shape[1]
    XX_inv = np.linalg.inv(X @ X.T)
    B_ols  = Y @ X.T @ XX_inv
    U_ols  = Y - B_ols @ X
    S_ols  = (U_ols @ U_ols.T) / T_eff
    return B_ols, S_ols, U_ols, X, T_eff, XX_inv


def draw_posterior(data, p, n_draws=2500):
    K = data.shape[1]
    B_ols, S_ols, U_ols, X, T_eff, XX_inv = estimate_ols(data, p)
    v_post = T_eff
    S_post = U_ols @ U_ols.T
    posterior_Sigma = []
    posterior_B     = []
    for _ in range(n_draws):
        Sigma_d = sample_invwishart(S_post, v_post)
        B_d     = np.zeros_like(B_ols)
        for k in range(K):
            cov_k  = Sigma_d[k, k] * XX_inv
            B_d[k] = np.random.multivariate_normal(B_ols[k], cov_k)
        posterior_Sigma.append(Sigma_d)
        posterior_B.append(B_d)
    return np.array(posterior_Sigma), np.array(posterior_B), B_ols, S_ols


def project_to_null(q, P, zero_rows):
    """
    Project q so that P@q has zeros in the specified rows.
    This implements zero restrictions by projecting q onto
    the null space of the corresponding rows of P.
    """
    if len(zero_rows) == 0:
        return q
    # Rows of P that must be zero
    C = P[zero_rows, :]
    # Project q onto null space of C
    # q_proj = q - C^T (C C^T)^{-1} C q
    CCt     = C @ C.T
    if np.linalg.matrix_rank(CCt) < len(zero_rows):
        return None
    q_proj  = q - C.T @ np.linalg.solve(CCt, C @ q)
    norm    = np.linalg.norm(q_proj)
    if norm < 1e-10:
        return None
    return q_proj / norm


def check_sign_col(col, s_col):
    for i in range(len(col)):
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


def find_rotation_sequential(P, S, Z, n_candidates=500):
    K        = P.shape[0]
    n_shocks = S.shape[1]
    impact   = np.zeros((K, n_shocks))
    Q_cols   = []

    for j in range(n_shocks):
        s_col    = S[:, j]
        z_col    = Z[:, j]
        zero_rows = [i for i in range(K) if z_col[i] == 1]
        found    = False

        for _ in range(n_candidates):
            q = np.random.randn(K)
            q = q / np.linalg.norm(q)

            # Orthogonalize against previous columns
            for q_prev in Q_cols:
                q = q - np.dot(q, q_prev) * q_prev
            norm = np.linalg.norm(q)
            if norm < 1e-10:
                continue
            q = q / norm

            # Apply zero restrictions via projection
            q = project_to_null(q, P, zero_rows)
            if q is None:
                continue

            # Re-orthogonalize after projection
            for q_prev in Q_cols:
                q = q - np.dot(q, q_prev) * q_prev
            norm = np.linalg.norm(q)
            if norm < 1e-10:
                continue
            q = q / norm

            col     = P @ q
            col_try = try_flip(col.copy(), s_col)

            if check_sign_col(col_try, s_col):
                impact[:, j] = col_try
                Q_cols.append(q.copy())
                found = True
                break

        if not found:
            return None

    return impact


def run_svar(data, p, S, Z, n_posterior=2500, n_accepted=500, n_candidates=500):
    print(f"Drawing {n_posterior} posterior samples...")
    post_Sigma, post_B, B_ols, S_ols = draw_posterior(data, p, n_draws=n_posterior)
    print(f"Posterior draws complete.")
    print(f"Searching for valid rotations (target={n_accepted})...")
    accepted = []
    for d in range(n_posterior):
        P      = np.linalg.cholesky(post_Sigma[d])
        impact = find_rotation_sequential(P, S, Z, n_candidates=n_candidates)
        if impact is not None:
            accepted.append(impact)
        if len(accepted) >= n_accepted:
            break
        if d % 500 == 0 and d > 0:
            print(f"  draws: {d}, accepted: {len(accepted)}")
    print(f"Accepted: {len(accepted)} out of {n_posterior}")
    return np.array(accepted) if accepted else None