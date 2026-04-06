import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

def plot_gpg(df, trend, cycle):
    dates = [str(d) for d in df.index]
    x = np.arange(len(dates))
    tick_every = 20

    VINO     = "#6B2D5E"
    VINO_LIGHT = "#C9A0BC"
    DARK     = "#2C2C2C"

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), facecolor="white")

    ax1.fill_between(x, df["gpg"] * 100, alpha=0.15, color=VINO)
    ax1.plot(x, df["gpg"] * 100, color=VINO, linewidth=1.8, label="GPG original")
    ax1.plot(x, trend * 100, color=DARK, linewidth=2, linestyle="--", label="Tendencia HP (λ=1600)")
    ax1.set_title("Gender Pay Gap — Serie original y tendencia HP", fontsize=13, color=DARK, pad=12)
    ax1.set_ylabel("GPG (%)", color=DARK)
    ax1.set_xticks(x[::tick_every])
    ax1.set_xticklabels(dates[::tick_every], rotation=45, fontsize=9)
    ax1.legend(framealpha=0.9, fontsize=10)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.grid(axis="y", alpha=0.2, linestyle="--")

    ax2.fill_between(x, cycle * 100, where=(cycle >= 0), alpha=0.3, color=VINO, label="GPG sobre tendencia")
    ax2.fill_between(x, cycle * 100, where=(cycle < 0),  alpha=0.3, color=VINO_LIGHT, label="GPG bajo tendencia")
    ax2.plot(x, cycle * 100, color=VINO, linewidth=1.5)
    ax2.axhline(0, color=DARK, linewidth=0.8, linestyle="-")
    ax2.set_title("Ciclo del GPG — Componente cíclico (filtro HP)", fontsize=13, color=DARK, pad=12)
    ax2.set_ylabel("Ciclo (%)", color=DARK)
    ax2.set_xticks(x[::tick_every])
    ax2.set_xticklabels(dates[::tick_every], rotation=45, fontsize=9)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.grid(axis="y", alpha=0.2, linestyle="--")
    ax2.legend(framealpha=0.9, fontsize=10)

    plt.suptitle("Kovalenko & Töpfer (2021) — Replicación", fontsize=10, color="gray", y=1.01)
    plt.tight_layout()
    plt.savefig("gpg_plot.png", dpi=150, bbox_inches="tight")
    print("Saved -> gpg_plot.png")
    plt.show()