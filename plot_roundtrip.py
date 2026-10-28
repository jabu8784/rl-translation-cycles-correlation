import math
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.regression.mixed_linear_model import MixedLM

# (metric column, display label); each also needs a `roundtrip_<metric>` column
METRICS = [
    ("bleu", "BLEU"),
    ("chrf", "chrF"),
    ("comet22", "Comet-22"),
    ("bleurt", "BLEURT"),
    ("bertscore", "BERTScore"),
]
MAX_COLS = 3


def plot_roundtrip(csv_path):
    df = pd.read_csv(csv_path)
    if "sent_idx" not in df.columns:
        num_sents = len(df) // 10
        df["sent_idx"] = [s for idx in range(num_sents) for s in [idx] * 10]

    # Plot whichever metrics are present in the CSV
    present = [
        (metric, label)
        for metric, label in METRICS
        if metric in df.columns and f"roundtrip_{metric}" in df.columns
    ]
    if not present:
        raise ValueError(
            f"No '<metric>' / 'roundtrip_<metric>' column pairs found in {csv_path}"
        )

    ncols = min(len(present), MAX_COLS)
    nrows = math.ceil(len(present) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False
    )
    flat_axes = axes.flatten()
    summary = []

    for ax, (y_col, label) in zip(flat_axes, present):
        x_col = f"roundtrip_{y_col}"
        # Filter per metric (e.g. all-identical groups give NaN after normalizing)
        sub = df[np.isfinite(df[x_col]) & np.isfinite(df[y_col])]
        x = sub[x_col].to_numpy(dtype=float)
        y = sub[y_col].to_numpy(dtype=float)
        if len(x) < 3 or x.std() == 0 or y.std() == 0:
            ax.set_title(f"{label}: not enough variation to plot")
            continue

        slope, intercept = np.polyfit(x, y, deg=1)
        r_value = float(np.corrcoef(x, y)[0, 1])
        r2 = r_value * r_value
        summary.append((label, len(x), r_value, r2, slope))

        x_line = np.linspace(x.min(), x.max(), 200)
        y_line = slope * x_line + intercept
        ax.scatter(x, y, alpha=0.4, edgecolors="none", s=4)
        ax.plot(
            x_line,
            y_line,
            color="red",
            linewidth=1.5,
            label=f"y = {slope:.3f}x + {intercept:.3e}\n$R^2$ = {r2:.4f}",
        )
        ax.set_xlabel(f"Roundtrip {label} (normalized)")
        ax.set_ylabel(f"{label} (normalized)")
        ax.set_title(f"Roundtrip {label} vs {label}")
        ax.legend(fontsize=9)

        # Mixed model: random slope per sentence
        print(f"\n===== {label} =====")
        try:
            model = MixedLM.from_formula(
                f"{y_col} ~ {x_col}", sub, groups="sent_idx", re_formula=f"~0+{x_col}"
            )
            print(model.fit(method="nm").summary())
        except Exception as e:  # convergence / singular fits shouldn't kill the plot
            print(f"MixedLM failed for {label}: {e}")

    # Hide unused axes in the grid
    for ax in flat_axes[len(present) :]:
        ax.set_visible(False)

    if summary:
        print("\n===== Summary (within-sentence normalized scores) =====")
        print(f"{'metric':<12}{'n':>8}{'r':>10}{'R^2':>10}{'slope':>10}")
        for label, n, r, r2, slope in summary:
            print(f"{label:<12}{n:>8}{r:>10.4f}{r2:>10.4f}{slope:>10.4f}")

    fig.tight_layout()
    out_path = csv_path.rsplit(".", 1)[0] + "_roundtrip_plots.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python plot_roundtrip.py <path/to/data.csv>")
        sys.exit(1)
    plot_roundtrip(sys.argv[1])
