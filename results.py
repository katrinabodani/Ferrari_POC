"""
FERRARI POC — Results vs Paper Claims
NeurIPS 2024 | FAST NUCES Islamabad | Spring 2026
Katrina Bodani (22i0545) & Syed Zain Abbas (22i1905)

Saves 3 separate graph PNGs, one per scenario.
"""

import matplotlib.pyplot as plt
import numpy as np

PAPER_COLOR = "#C8102E"   # Ferrari red
OUR_COLOR   = "#1A1A2E"   # Deep navy
W = 0.35


def label_bars(ax, bars):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3,
                f"{h:.2f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")


# ── Scenario 1: Backdoor Unlearning (MNIST) ──────────────────────────────────
labels1 = ["Baseline Dr\n(retain acc)", "Baseline Du\n(backdoor ASR)",
           "Ferrari Dr\nafter unlearning", "Ferrari Du\nafter unlearning"]
paper1  = [95.65, 97.43, 95.93,  0.11]
errors1 = [ 1.39,  3.69,  0.45,  0.01]
ours1   = [96.48, 97.78, 95.00,  0.12]

x1 = np.arange(len(labels1))
fig1, ax1 = plt.subplots(figsize=(10, 6))
bp = ax1.bar(x1 - W/2, paper1, W, label="Paper Claim", color=PAPER_COLOR, alpha=0.88,
             yerr=errors1, capsize=5, error_kw=dict(ecolor="gray", capthick=1.5, elinewidth=1.5))
bo = ax1.bar(x1 + W/2, ours1,  W, label="Our Result",  color=OUR_COLOR,   alpha=0.88)
label_bars(ax1, bp); label_bars(ax1, bo)
ax1.set_title("Scenario 1 — Backdoor Unlearning (MNIST)", fontsize=13, fontweight="bold")
ax1.set_xticks(x1); ax1.set_xticklabels(labels1, fontsize=10)
ax1.set_ylabel("Accuracy / ASR (%)"); ax1.set_ylim(0, 115)
ax1.legend(fontsize=10); ax1.grid(axis="y", linestyle="--", alpha=0.5)
ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)
fig1.tight_layout()
fig1.savefig("scenario1_backdoor_mnist.png", dpi=150, bbox_inches="tight")
print("Saved → scenario1_backdoor_mnist.png")


# ── Scenario 2: Bias Unlearning (CMNIST) ─────────────────────────────────────
labels2 = ["Baseline Dr\n(unbiased acc)", "Baseline Du\n(biased acc)",
           "Ferrari Dr\nafter unlearning", "Ferrari Du\nafter unlearning"]
paper2  = [64.94, 98.88, 84.31, 84.62]
errors2 = [ 7.88,  4.90,  2.63,  3.59]
ours2   = [67.91, 98.31, 83.37, 83.35]

x2 = np.arange(len(labels2))
fig2, ax2 = plt.subplots(figsize=(10, 6))
bp2 = ax2.bar(x2 - W/2, paper2, W, label="Paper Claim", color=PAPER_COLOR, alpha=0.88,
              yerr=errors2, capsize=5, error_kw=dict(ecolor="gray", capthick=1.5, elinewidth=1.5))
bo2 = ax2.bar(x2 + W/2, ours2,  W, label="Our Result",  color=OUR_COLOR,   alpha=0.88)
label_bars(ax2, bp2); label_bars(ax2, bo2)
ax2.set_title("Scenario 2 — Bias Unlearning (CMNIST Background)", fontsize=13, fontweight="bold")
ax2.set_xticks(x2); ax2.set_xticklabels(labels2, fontsize=10)
ax2.set_ylabel("Accuracy (%)"); ax2.set_ylim(0, 120)
ax2.legend(fontsize=10); ax2.grid(axis="y", linestyle="--", alpha=0.5)
ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
ax2.annotate("Fairness gap ≈ 0%\n(Dr ≈ Du)", xy=(3.18, 83.35), xytext=(2.4, 108),
             arrowprops=dict(arrowstyle="->", color="#555"),
             fontsize=9, color="#27AE60", fontweight="bold")
fig2.tight_layout()
fig2.savefig("scenario2_bias_cmnist.png", dpi=150, bbox_inches="tight")
print("Saved → scenario2_bias_cmnist.png")


# ── Scenario 3: Sensitive Feature Unlearning (Diabetes) ─────────────────────
labels3 = ["Baseline\ntrain acc", "Baseline\ntest acc", "Unlearn client\ntrain acc",
           "After Ferrari\ntrain acc", "After Ferrari\ntest acc"]
paper3 = [75.00, 73.00, 85.78, 73.00, 72.00]
ours3  = [74.55, 72.19, 85.55, 72.59, 71.80]

x3 = np.arange(len(labels3))
fig3, ax3 = plt.subplots(figsize=(11, 6))
bp3 = ax3.bar(x3 - W/2, paper3, W, label="Paper Claim", color=PAPER_COLOR, alpha=0.88)
bo3 = ax3.bar(x3 + W/2, ours3,  W, label="Our Result",  color=OUR_COLOR,   alpha=0.88)
label_bars(ax3, bp3); label_bars(ax3, bo3)
ax3.set_title("Scenario 3 — Sensitive Feature Unlearning (Diabetes)", fontsize=13, fontweight="bold")
ax3.set_xticks(x3); ax3.set_xticklabels(labels3, fontsize=10)
ax3.set_ylabel("Accuracy (%)"); ax3.set_ylim(60, 100)
ax3.legend(fontsize=10); ax3.grid(axis="y", linestyle="--", alpha=0.5)
ax3.spines["top"].set_visible(False); ax3.spines["right"].set_visible(False)
fig3.tight_layout()
fig3.savefig("scenario3_diabetes.png", dpi=150, bbox_inches="tight")
print("Saved → scenario3_diabetes.png")

plt.show()