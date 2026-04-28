import sys
sys.path.insert(0, '/Users/administrador/Documents/GitHub/gendergapvar')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gendergapvar.models.svar import run_svar, estimate_ols
from gendergapvar.results.irf import compute_irf_distribution, compute_gpg_irf
from table2_output import SVARRestrictions

np.random.seed(42)

df = pd.read_excel("data_merged.xlsx", index_col=0)
df.index = pd.PeriodIndex(df.index, freq="Q")
cols = ["wage_m", "wage_f", "unemp_m", "unemp_f", "lfp_m", "lfp_f", "gdp", "cpi"]

restr = SVARRestrictions()
S = restr.get_sign_restrictions()
Z = restr.get_zero_restrictions()

VINO  = "#6B2D5E"
DARK  = "#2C2C2C"
quarters = np.arange(41)

shock_labels = {
    "demand"             : "Demand shock",
    "technology"         : "Technology shock",
    "wage_bargaining"    : "Wage bargaining shock",
    "labor_supply"       : "Labor supply shock",
    "male_labor_supply"  : "Male labor supply shock",
    "female_labor_supply": "Female labor supply shock",
}

def run_period(label, start, end):
    print(f"\n{'='*60}")
    print(f"Sample: {start} to {end}")
    print(f"{'='*60}")
    data = np.log(df.loc[start:end, cols]).values
    print(f"Observations: {len(data)}")
    accepted = run_svar(data, p=4, S=S, Z=Z,
                        n_posterior=2500, n_accepted=300)
    if accepted is None:
        print("No accepted draws")
        return None, None, None
    B_var, _, _, _, _, _ = estimate_ols(data, p=4)
    median, p16, p84 = compute_irf_distribution(B_var, accepted, n_periods=40)
    gpg_med = compute_gpg_irf(median)
    gpg_p16 = compute_gpg_irf(p16)
    gpg_p84 = compute_gpg_irf(p84)
    return gpg_med, gpg_p16, gpg_p84

print("Running full sample...")
data_full = np.log(df[cols]).values
accepted_full = run_svar(data_full, p=4, S=S, Z=Z, n_posterior=2500, n_accepted=300)
B_full, _, _, _, _, _ = estimate_ols(data_full, p=4)
med_full, p16_full, p84_full = compute_irf_distribution(B_full, accepted_full, n_periods=40)
gpg_full = compute_gpg_irf(med_full)

print("Running 1979-1999...")
gpg_79, gpg_79_p16, gpg_79_p84 = run_period("1979-1999", "1979Q1", "1999Q4")

print("Running 2000-2019...")
gpg_00, gpg_00_p16, gpg_00_p84 = run_period("2000-2019", "2000Q1", "2019Q2")

print("\nPlotting sample split results...")
fig, axes = plt.subplots(2, 3, figsize=(14, 8), facecolor="white")
axes = axes.flatten()

for j, shock in enumerate(restr.shock_names):
    ax = axes[j]

    ax.plot(quarters, gpg_full[:, j]*100, color=DARK, linewidth=1.5,
            linestyle="-", label="Full (1979-2019)")

    if gpg_79 is not None:
        ax.fill_between(quarters, gpg_79_p16[:, j]*100, gpg_79_p84[:, j]*100,
                        alpha=0.15, color=VINO)
        ax.plot(quarters, gpg_79[:, j]*100, color=VINO, linewidth=1.5,
                linestyle="--", label="1979-1999")

    if gpg_00 is not None:
        ax.fill_between(quarters, gpg_00_p16[:, j]*100, gpg_00_p84[:, j]*100,
                        alpha=0.15, color="#888780")
        ax.plot(quarters, gpg_00[:, j]*100, color="#888780", linewidth=1.5,
                linestyle=":", label="2000-2019")

    ax.axhline(0, color=DARK, linewidth=0.7, linestyle="--", alpha=0.5)
    ax.set_title(f"GPG — {shock_labels[shock]}", fontsize=9, color=DARK, pad=4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.15, linestyle="--")
    ax.tick_params(labelsize=7)
    ax.set_xlim(0, 40)
    ax.set_xlabel("Quarter", fontsize=8)
    ax.set_ylabel("GPG response (%)", fontsize=8)
    if j == 0:
        ax.legend(fontsize=7)

plt.suptitle("Robustness Check — Sample Split\nReplication of Kovalenko & Töpfer (2021)",
             fontsize=11, color=DARK, y=1.01)
plt.tight_layout()
plt.savefig("/Users/administrador/Documents/tesis/robustness_sample_split.png",
            dpi=150, bbox_inches="tight")
print("Saved -> robustness_sample_split.png")
plt.show()