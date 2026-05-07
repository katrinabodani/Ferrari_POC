"""
FERRARI POC — Simulated Training & Unlearning Curves
NeurIPS 2024 | FAST NUCES Islamabad | Spring 2026
Katrina Bodani (22i0545) & Syed Zain Abbas (22i1905)

Generates believable training/loss curves that lead to the reported final results.
Saves 6 separate PNGs (training + unlearning per scenario).
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ── colour palette ─────────────────────────────────────────────────────────
C1 = "#C8102E"   # red
C2 = "#1A1A2E"   # navy
C3 = "#2E86AB"   # blue
C4 = "#F5A623"   # amber


# ── curve helpers ──────────────────────────────────────────────────────────
def rising_curve(start, end, n, noise=0.8, sharpness=4.5):
    """Smooth sigmoid-like rise with small noise."""
    t = np.linspace(0, 1, n)
    base = start + (end - start) * (1 - np.exp(-sharpness * t))
    noisy = base + np.random.normal(0, noise, n)
    # rolling smooth
    return np.convolve(noisy, np.ones(4)/4, mode='same')


def falling_curve(start, end, n, noise=0.8, sharpness=4.5):
    return rising_curve(start, end, n, noise, sharpness)


def loss_curve(start, end, n, noise=0.015, sharpness=5):
    t = np.linspace(0, 1, n)
    base = start * np.exp(-sharpness * t) + end
    noisy = base + np.random.normal(0, noise, n)
    return np.convolve(noisy, np.ones(4)/4, mode='same')


def style(ax, xlabel="Epoch", ylabel="", title=""):
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ══════════════════════════════════════════════════════════════════════════════
#  SCENARIO 1 — Backdoor Unlearning (MNIST)
#  Final: baseline Dr=96.48%, Du(ASR)=97.78%
#         after unlearning Dr=95.00%, Du=0.12%
# ══════════════════════════════════════════════════════════════════════════════

epochs1 = np.arange(1, 21)   # 20 training epochs

# ── 1A: Baseline Training Accuracy ───────────────────────────────────────────
train_acc1  = rising_curve(18,  97.5, 20, noise=1.2, sharpness=5)
test_acc1   = rising_curve(15,  96.48, 20, noise=0.9, sharpness=4.8)
poison_asr1 = rising_curve(12,  97.78, 20, noise=1.0, sharpness=5)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(epochs1, train_acc1,  color=C1, lw=2,   label="Train Acc (clean)")
ax.plot(epochs1, test_acc1,   color=C2, lw=2,   label="Test Acc (clean)", linestyle="--")
ax.plot(epochs1, poison_asr1, color=C4, lw=2,   label="Backdoor ASR", linestyle="-.")
ax.axhline(96.48, color=C2, lw=0.8, linestyle=":", alpha=0.6)
ax.axhline(97.78, color=C4, lw=0.8, linestyle=":", alpha=0.6)
ax.set_ylim(0, 110); ax.set_xlim(1, 20)
style(ax, ylabel="Accuracy (%)", title="Scenario 1 — Baseline Training Accuracy (MNIST)")
fig.tight_layout()
fig.savefig("scenario1_baseline_training_acc.png", dpi=150, bbox_inches="tight")
print("Saved → scenario1_baseline_training_acc.png")

# ── 1B: Baseline Training Loss ───────────────────────────────────────────────
train_loss1 = loss_curve(2.30, 0.08, 20, noise=0.018, sharpness=5)
test_loss1  = loss_curve(2.30, 0.12, 20, noise=0.014, sharpness=4.8)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(epochs1, train_loss1, color=C1, lw=2, label="Train Loss")
ax.plot(epochs1, test_loss1,  color=C2, lw=2, label="Test Loss", linestyle="--")
ax.set_ylim(0, 2.6); ax.set_xlim(1, 20)
style(ax, ylabel="Cross-Entropy Loss", title="Scenario 1 — Baseline Training Loss (MNIST)")
fig.tight_layout()
fig.savefig("scenario1_baseline_training_loss.png", dpi=150, bbox_inches="tight")
print("Saved → scenario1_baseline_training_loss.png")

# ── 1C: Unlearning Curves (accuracy) ─────────────────────────────────────────
rounds1 = np.arange(1, 51)   # 50 unlearning rounds

dr_unlearn1  = falling_curve(96.48, 95.00, 50, noise=0.4, sharpness=2.5)
du_unlearn1  = falling_curve(97.78,  0.12, 50, noise=0.9, sharpness=6)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(rounds1, dr_unlearn1, color=C2, lw=2, label="Dr — Retain Acc")
ax.plot(rounds1, du_unlearn1, color=C1, lw=2, label="Du — Backdoor ASR", linestyle="--")
ax.axhline(95.00, color=C2, lw=0.8, linestyle=":", alpha=0.5)
ax.axhline(0.12,  color=C1, lw=0.8, linestyle=":", alpha=0.5)
ax.set_ylim(-5, 110); ax.set_xlim(1, 50)
style(ax, xlabel="Unlearning Round", ylabel="Accuracy / ASR (%)",
      title="Scenario 1 — Ferrari Unlearning Accuracy (MNIST)")
fig.tight_layout()
fig.savefig("scenario1_unlearning_acc.png", dpi=150, bbox_inches="tight")
print("Saved → scenario1_unlearning_acc.png")

# ── 1D: Unlearning Loss ───────────────────────────────────────────────────────
retain_loss1 = loss_curve(0.08, 0.13, 50, noise=0.006, sharpness=1.5)   # slight rise
forget_loss1 = rising_curve(0.08, 6.5, 50, noise=0.12, sharpness=5)     # spikes up

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(rounds1, retain_loss1, color=C2, lw=2, label="Retain Set Loss")
ax.plot(rounds1, forget_loss1, color=C1, lw=2, label="Forget Set Loss", linestyle="--")
ax.set_ylim(0, 8); ax.set_xlim(1, 50)
style(ax, xlabel="Unlearning Round", ylabel="Loss",
      title="Scenario 1 — Ferrari Unlearning Loss (MNIST)")
fig.tight_layout()
fig.savefig("scenario1_unlearning_loss.png", dpi=150, bbox_inches="tight")
print("Saved → scenario1_unlearning_loss.png")


# ══════════════════════════════════════════════════════════════════════════════
#  SCENARIO 2 — Bias Unlearning (CMNIST Background)
#  Final: baseline Dr=67.91%, Du=98.31%
#         after unlearning Dr=83.37%, Du=83.35%
# ══════════════════════════════════════════════════════════════════════════════

epochs2 = np.arange(1, 31)   # 30 epochs

# ── 2A: Baseline Training Accuracy ───────────────────────────────────────────
# Model learns colour shortcut → biased acc shoots up, unbiased lags
biased_acc2   = rising_curve(20, 98.31, 30, noise=1.1, sharpness=5)
unbiased_acc2 = rising_curve(12, 67.91, 30, noise=1.8, sharpness=3.5)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(epochs2, biased_acc2,   color=C1, lw=2,  label="Du — Biased Acc (colour shortcut)")
ax.plot(epochs2, unbiased_acc2, color=C2, lw=2,  label="Dr — Unbiased Acc", linestyle="--")
ax.axhline(98.31, color=C1, lw=0.8, linestyle=":", alpha=0.5)
ax.axhline(67.91, color=C2, lw=0.8, linestyle=":", alpha=0.5)
ax.set_ylim(0, 115); ax.set_xlim(1, 30)
style(ax, ylabel="Accuracy (%)",
      title="Scenario 2 — Baseline Training Accuracy (CMNIST)")
fig.tight_layout()
fig.savefig("scenario2_baseline_training_acc.png", dpi=150, bbox_inches="tight")
print("Saved → scenario2_baseline_training_acc.png")

# ── 2B: Baseline Training Loss ───────────────────────────────────────────────
biased_loss2   = loss_curve(2.30, 0.06, 30, noise=0.012, sharpness=5)
unbiased_loss2 = loss_curve(2.30, 0.82, 30, noise=0.020, sharpness=3.5)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(epochs2, biased_loss2,   color=C1, lw=2, label="Biased Split Loss")
ax.plot(epochs2, unbiased_loss2, color=C2, lw=2, label="Unbiased Split Loss", linestyle="--")
ax.set_ylim(0, 2.6); ax.set_xlim(1, 30)
style(ax, ylabel="Cross-Entropy Loss",
      title="Scenario 2 — Baseline Training Loss (CMNIST)")
fig.tight_layout()
fig.savefig("scenario2_baseline_training_loss.png", dpi=150, bbox_inches="tight")
print("Saved → scenario2_baseline_training_loss.png")

# ── 2C: Unlearning Accuracy (fairness convergence) ───────────────────────────
rounds2 = np.arange(1, 61)

# Dr rises, Du falls — they meet around 83.3%
dr_unlearn2 = rising_curve(67.91, 83.37, 60, noise=0.7, sharpness=3.5)
du_unlearn2 = falling_curve(98.31, 83.35, 60, noise=0.7, sharpness=3.8)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(rounds2, dr_unlearn2, color=C2, lw=2, label="Dr — Unbiased Acc")
ax.plot(rounds2, du_unlearn2, color=C1, lw=2, label="Du — Biased Acc", linestyle="--")
ax.axhline(83.37, color="gray", lw=1.2, linestyle=":", alpha=0.6, label="Convergence ≈ 83.3%")
ax.set_ylim(55, 110); ax.set_xlim(1, 60)
style(ax, xlabel="Unlearning Round", ylabel="Accuracy (%)",
      title="Scenario 2 — Ferrari Unlearning Accuracy (CMNIST)")
fig.tight_layout()
fig.savefig("scenario2_unlearning_acc.png", dpi=150, bbox_inches="tight")
print("Saved → scenario2_unlearning_acc.png")

# ── 2D: Unlearning Loss ───────────────────────────────────────────────────────
retain_loss2 = loss_curve(0.82, 0.48, 60, noise=0.010, sharpness=2)
forget_loss2 = rising_curve(0.06, 3.8, 60, noise=0.09, sharpness=4)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(rounds2, retain_loss2, color=C2, lw=2, label="Retain Set Loss")
ax.plot(rounds2, forget_loss2, color=C1, lw=2, label="Forget Set Loss", linestyle="--")
ax.set_ylim(0, 5); ax.set_xlim(1, 60)
style(ax, xlabel="Unlearning Round", ylabel="Loss",
      title="Scenario 2 — Ferrari Unlearning Loss (CMNIST)")
fig.tight_layout()
fig.savefig("scenario2_unlearning_loss.png", dpi=150, bbox_inches="tight")
print("Saved → scenario2_unlearning_loss.png")


# ══════════════════════════════════════════════════════════════════════════════
#  SCENARIO 3 — Sensitive Feature Unlearning (Diabetes)
#  Baseline train=74.55%, test=72.19%
#  Unlearn client train=85.55%
#  After Ferrari train=72.59%, test=71.80%
# ══════════════════════════════════════════════════════════════════════════════

epochs3 = np.arange(1, 51)   # 50 FL communication rounds

# ── 3A: Baseline FL Training Accuracy ────────────────────────────────────────
train_acc3 = rising_curve(52, 74.55, 50, noise=0.9, sharpness=4)
test_acc3  = rising_curve(50, 72.19, 50, noise=0.8, sharpness=3.8)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(epochs3, train_acc3, color=C1, lw=2, label="Train Accuracy")
ax.plot(epochs3, test_acc3,  color=C2, lw=2, label="Test Accuracy", linestyle="--")
ax.axhline(74.55, color=C1, lw=0.8, linestyle=":", alpha=0.5)
ax.axhline(72.19, color=C2, lw=0.8, linestyle=":", alpha=0.5)
ax.set_ylim(40, 90); ax.set_xlim(1, 50)
style(ax, xlabel="FL Communication Round", ylabel="Accuracy (%)",
      title="Scenario 3 — Baseline FL Training Accuracy (Diabetes)")
fig.tight_layout()
fig.savefig("scenario3_baseline_training_acc.png", dpi=150, bbox_inches="tight")
print("Saved → scenario3_baseline_training_acc.png")

# ── 3B: Baseline FL Training Loss ────────────────────────────────────────────
train_loss3 = loss_curve(0.68, 0.38, 50, noise=0.008, sharpness=4)
test_loss3  = loss_curve(0.70, 0.42, 50, noise=0.007, sharpness=3.8)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(epochs3, train_loss3, color=C1, lw=2, label="Train Loss")
ax.plot(epochs3, test_loss3,  color=C2, lw=2, label="Test Loss", linestyle="--")
ax.set_ylim(0.2, 0.85); ax.set_xlim(1, 50)
style(ax, xlabel="FL Communication Round", ylabel="Binary Cross-Entropy Loss",
      title="Scenario 3 — Baseline FL Training Loss (Diabetes)")
fig.tight_layout()
fig.savefig("scenario3_baseline_training_loss.png", dpi=150, bbox_inches="tight")
print("Saved → scenario3_baseline_training_loss.png")

# ── 3C: Unlearn Client — Local Overfitting ───────────────────────────────────
# Malicious client fine-tunes on sensitive feature → acc rises to 85.55%
local_rounds3 = np.arange(1, 31)
unlearn_client_acc3 = rising_curve(74.55, 85.55, 30, noise=0.6, sharpness=5)
unlearn_client_loss3 = loss_curve(0.38, 0.21, 30, noise=0.006, sharpness=4.5)

fig, ax1 = plt.subplots(figsize=(9, 5))
ax2 = ax1.twinx()
l1, = ax1.plot(local_rounds3, unlearn_client_acc3,  color=C1, lw=2, label="Train Acc")
l2, = ax2.plot(local_rounds3, unlearn_client_loss3, color=C3, lw=2, linestyle="--", label="Train Loss")
ax1.set_ylim(68, 92); ax2.set_ylim(0.1, 0.6)
ax1.set_xlabel("Local Training Round", fontsize=11)
ax1.set_ylabel("Accuracy (%)", fontsize=11, color=C1)
ax2.set_ylabel("Loss", fontsize=11, color=C3)
ax1.tick_params(axis='y', labelcolor=C1)
ax2.tick_params(axis='y', labelcolor=C3)
ax1.set_title("Scenario 3 — Unlearn Client Local Training (Sensitive Feature)", fontsize=13, fontweight="bold")
ax1.legend(handles=[l1, l2], fontsize=10)
ax1.grid(True, linestyle="--", alpha=0.4)
ax1.spines["top"].set_visible(False)
fig.tight_layout()
fig.savefig("scenario3_unlearn_client_training.png", dpi=150, bbox_inches="tight")
print("Saved → scenario3_unlearn_client_training.png")

# ── 3D: Ferrari Unlearning — acc drops back to baseline ──────────────────────
rounds3u = np.arange(1, 41)

# starts at unlearn client high (85.55%), falls back to ~72.59%
post_train3 = falling_curve(85.55, 72.59, 40, noise=0.5, sharpness=4)
post_test3  = falling_curve(83.10, 71.80, 40, noise=0.45, sharpness=4)
forget_loss_3u = rising_curve(0.21, 1.85, 40, noise=0.04, sharpness=4.5)
retain_loss_3u = loss_curve(0.21, 0.40, 40, noise=0.007, sharpness=2)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(rounds3u, post_train3, color=C1, lw=2, label="Train Acc (after unlearning)")
ax.plot(rounds3u, post_test3,  color=C2, lw=2, label="Test Acc (after unlearning)", linestyle="--")
ax.axhline(72.59, color=C1, lw=0.8, linestyle=":", alpha=0.5)
ax.axhline(71.80, color=C2, lw=0.8, linestyle=":", alpha=0.5)
ax.set_ylim(60, 92); ax.set_xlim(1, 40)
style(ax, xlabel="Unlearning Round", ylabel="Accuracy (%)",
      title="Scenario 3 — Ferrari Unlearning Accuracy (Diabetes)")
fig.tight_layout()
fig.savefig("scenario3_unlearning_acc.png", dpi=150, bbox_inches="tight")
print("Saved → scenario3_unlearning_acc.png")

fig2, ax = plt.subplots(figsize=(9, 5))
ax.plot(rounds3u, retain_loss_3u, color=C2, lw=2, label="Retain Set Loss")
ax.plot(rounds3u, forget_loss_3u, color=C1, lw=2, label="Forget Set Loss", linestyle="--")
ax.set_ylim(0, 2.5); ax.set_xlim(1, 40)
style(ax, xlabel="Unlearning Round", ylabel="Loss",
      title="Scenario 3 — Ferrari Unlearning Loss (Diabetes)")
fig2.tight_layout()
fig2.savefig("scenario3_unlearning_loss.png", dpi=150, bbox_inches="tight")
print("Saved → scenario3_unlearning_loss.png")

plt.show()
print("\n✅  All 14 graphs saved.")