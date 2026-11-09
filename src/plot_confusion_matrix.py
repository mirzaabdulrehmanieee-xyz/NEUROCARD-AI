
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Project paths
ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

# Heartbeat classes
classes = ["N", "S", "V", "F", "Q"]

# Actual test confusion matrix
cm = np.array([
    [18048, 24, 23, 7, 16],
    [186, 359, 9, 0, 2],
    [92, 5, 1318, 29, 4],
    [42, 0, 6, 114, 0],
    [23, 7, 7, 0, 1571]
])

# Plot
fig, ax = plt.subplots(figsize=(9, 7))

fig.patch.set_facecolor("#0D1530")
ax.set_facecolor("#0D1530")

im = ax.imshow(cm, cmap="Blues")

# Axis labels
ax.set_xticks(np.arange(len(classes)))
ax.set_yticks(np.arange(len(classes)))

ax.set_xticklabels(classes, fontsize=12, color="white")
ax.set_yticklabels(classes, fontsize=12, color="white")

ax.set_xlabel(
    "Predicted Class",
    fontsize=13,
    color="white",
    labelpad=12
)

ax.set_ylabel(
    "Actual Class",
    fontsize=13,
    color="white",
    labelpad=12
)

ax.set_title(
    "ECG Heartbeat Classification\nConfusion Matrix",
    fontsize=16,
    fontweight="bold",
    color="white",
    pad=20
)

# Display values inside cells
threshold = cm.max() / 2

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):

        ax.text(
            j, i,
            f"{cm[i, j]:,}",
            ha="center",
            va="center",
            color="white" if cm[i, j] > threshold else "#173B56",
            fontsize=11,
            fontweight="bold"
        )

# Color bar
cbar = fig.colorbar(im, ax=ax)
cbar.ax.yaxis.set_tick_params(color="white")
plt.setp(
    cbar.ax.get_yticklabels(),
    color="white"
)

# Styling
ax.tick_params(colors="white")

for spine in ax.spines.values():
    spine.set_color("#52648F")

plt.tight_layout()

# Save image
output_path = RESULTS / "confusion_matrix.png"

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight",
    facecolor=fig.get_facecolor()
)

plt.close(fig)

print("Confusion matrix generated successfully!")
print("Saved at:", output_path)
