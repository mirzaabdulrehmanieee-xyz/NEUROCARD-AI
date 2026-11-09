import os
import matplotlib.pyplot as plt
from load_data import load_raw
CLASS_NAMES = [
    "N: normal",
    "S: supraventricular ectopic",
    "V: ventricular ectopic",
    "F: fusion",
    "Q: unknown / paced"
]
FS = 125  # Sampling frequency in Hz
def plot_one_per_class(df, out_path="results/example_beats.png"):
    """
    Plot the first beat of each class.
    """
    fig, axes = plt.subplots(
        1, 5,
        figsize=(16, 3),
        sharey=True
    )
    time_ms = [1000 * i / FS for i in range(187)]
    for k, ax in enumerate(axes):
        row = df[df[187] == k].iloc[0]
        ax.plot(
            time_ms,
            row.iloc[:187].values
        )
        ax.set_title(CLASS_NAMES[k], fontsize=9)
        ax.set_xlabel("Time (ms)")
    axes[0].set_ylabel("Normalised amplitude")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    raw = load_raw()
    plot_one_per_class(raw["train"])
