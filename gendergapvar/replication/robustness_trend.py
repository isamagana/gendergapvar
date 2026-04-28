import sys
sys.path.insert(0, '/Users/administrador/Documents/GitHub/gendergapvar')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from table2_output import SVARRestrictions

np.random.seed(42)

df = pd.read_excel("data_merged.xlsx", index_col=0)
df.index = pd.PeriodIndex(df.index, freq="Q")
cols = ["wage_m", "wage_f", "unemp_m", "unemp_f", "lfp_m", "lfp_f", "gdp", "cpi"]
data = np.log(df[cols]).values
T, K = data.shape
p = 4

restr = SVARRestrictions()
S = restr.get_sign_restrictions()
Z = restr.get_zero_restrictions()

def build_matrices_trend(data, p):
    T, K = data.shape
    Y = data[p:].T
    t = np.arange(1, T - p + 1).reshape(1, -1)
    X = np.vstack([np.ones((1, T - p)), t])
    for lag in range(1, p + 1):
        X = np.vstack([X, data[p-lag:T-lag].T])
    return Y, X

def estimate_ols_trend(data, p):
    Y, X   = build_matrices_trend(data, p)
    T_eff  = Y.shape[1]
    XX_inv = np.linalg.inv(X @ X.T)
    B_ols  = Y @ X.T @ XX_inv
    U_ols  = Y - B_ols @ X
    S_ols  = (U_ols @ U_ols.T) / T_eff
    return B_ols, S_ols, U_ols, X, T_eff, XX_inv

from gendergapvar.models.svar import sample_invwishart, find_rotation_sequential, draw_posterior
from gendergapvar.results.irf import compute_gpg_irf

def run_svar_trend(data, p, S, Z, n_posterior=2500, n_accepted=300):
    B_ols, S_ols, U_ols, X, T_eff, XX_inv = estimate_ols_trend(data, p)
    v_post = T_eff
    S_post = U_ols @ U_ols.T
    K = data.shape[1]

    print(f"Drawing {n_posterior} posterior samples (with trend)...")
    post_Sigma = []
    for _ in range(n_posterior):
        Sigma_d = sample_invwishart(S_post, v_post)
        post_Sigma.append(Sigma_d)

    print(f"Searching for valid rotations (target={n_accepted})...")
    accepted = []
    for d in range(n_posterior):
        P      = np.linalg.cholesky(post_Sigma[d])
        impact = find_rotation_sequential(P, S, Z, n_candidates=500)
        if impact is not None:
            accepted.append(impact)
        if len(accepted) >= n_accepted:
            break

    print(f"Accepted: {len(accepted)}")
    return np.array(accepted) if accepted else None, B_ols

def compute_irf_trend(B_var, impact, n_periods=40):
    K        = impact.shape[0]
    n_shocks = impact.shape[1]
    p        = (B_var.shape[1] - 2) // K
    A = []
    for i in range(p):
        A.append(B_var[:, 2 + i*K : 2 + (i+1)*K])
    irf = np.zeros((n_periods + 1, K, n_shocks))
    irf[0] = impact
    for t in range(1, n_periods + 1):
        for lag in range(min(t, p)):
            irf[t] += A[lag] @ irf[t - lag - 1]
    return irf

print("Running SVAR with linear trend...")
accepted_trend, B_trend = run_svar_trend(data, p, S, Z)

print("Running SVAR baseline...")
from gendergapvar.models.svar import run_svar, estimate_ols
from gendergapvar.results.irf import compute_irf_distribution
accepted_base = run_svar(data, p=4, S=S, Z=Z, n_posterior=2500, n_accepted=300)
B_base, _, _, _, _, _ = estimate_ols(data, p=4)
med_base, p16_base, p84_base = compute_irf_distribution(B_base, accepted_base, n_periods=40)
gpg_base = compute_gpg_irf(med_base)

if accepted_trend is not None:
    all_irfs = [compute_irf_trend(B_trend, imp) for imp in accepted_trend]
    all_irfs = np.array(all_irfs)
    med_trend = np.median(all_irfs, axis=0)
    p16_trend = np.percentile(all_irfs, 16, axis=0)
    p84_trend = np.percentile(all_irfs, 84, axis=0)
    gpg_trend     = compute_gpg_irf(med_trend)
    gpg_trend_p16 = compute_gpg_irf(p16_trend)
    gpg_trend_p84 = compute_gpg_irf(p84_trend)

    VINO     = "#6B2D5E"
    DARK     = "#2C2C2C"
    quarters = np.arange(41)

    shock_labels = [
        "Demand shock", "Technology shock", "Wage bargaining shock",
        "Labor supply shock", "Male labor supply shock", "Female labor supply shock",
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), facecolor="white")
    axes = axes.flatten()

    for j in range(len(restr.shock_names)):
        ax = axes[j]
        ax.fill_between(quarters, gpg_trend_p16[:, j]*100, gpg_trend_p84[:, j]*100,
                        alpha=0.2, color=VINO)
        ax.plot(quarters, gpg_trend[:, j]*100, color=VINO, linewidth=1.5,
                linestyle="--", label="Linear trend")
        ax.plot(quarters, gpg_base[:, j]*100, color=DARK, linewidth=1.5,
                linestyle="-", label="Baseline")
        ax.axhline(0, color=DARK, linewidth=0.7, linestyle="--", alpha=0.5)
        ax.set_title(f"GPG — {shock_labels[j]}", fontsize=9, color=DARK, pad=4)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.15, linestyle="--")
        ax.tick_params(labelsize=7)
        ax.set_xlim(0, 40)
        ax.set_xlabel("Quarter", fontsize=8)
        ax.set_ylabel("GPG response (%)", fontsize=8)
        if j == 0:
            ax.legend(fontsize=7)

    plt.suptitle("Robustness Check — Linear Trend\nReplication of Kovalenko & Töpfer (2021)",
                 fontsize=11, color=DARK, y=1.01)
    plt.tight_layout()
    plt.savefig("/Users/administrador/Documents/tesis/robustness_trend.png",
                dpi=150, bbox_inches="tight")
    print("Saved -> robustness_trend.png")
    plt.show()