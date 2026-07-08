import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.regression.mixed_linear_model import MixedLM


def plot_roundtrip(csv_path):
    df = pd.read_csv(csv_path)
    if "sent_idx" not in df.columns:
        num_sents = len(df) // 10
        df["sent_idx"] = [s for idx in range(num_sents) for s in [idx] * 10]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    pairs = [
        ("roundtrip_bleu", "bleu", "BLEU"),
        ("roundtrip_chrf", "chrf", "chrF"),
        ("roundtrip_comet22", "comet22", "Comet-22"),
        ("roundtrip_bleurt", "bleurt", "BLEURT"),
        ("roundtrip_bertscore", "bertscore", "BERTScore"),
    ]

    for ax, (x_col, y_col, label) in zip(axes, pairs):
        mask = np.isfinite(df[x_col]) & np.isfinite(df[y_col])
        df = df[mask]

        # EDIT: ** operator wasn't working so I changed it to use the to_numpy method with dtype=float
        
        # x = df[x_col].values
        # y = df[y_col].values
        
        # slope, intercept, r, p, es = stats.linregress(x, y)
        # r2 = r ** 2

        x = df[x_col].to_numpy(dtype=float)
        y = df[y_col].to_numpy(dtype=float)

        slope, intercept = np.polyfit(x, y, deg=1)

        corr_matrix = np.corrcoef(x, y)
        r_value = float(corr_matrix[0, 1])
        r2 = r_value * r_value

        x_line = np.linspace(x.min(), x.max(), 200)
        y_line = slope * x_line + intercept
        # ----------------------------
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

        # MixedLM
        model = MixedLM.from_formula(
            f"{y_col} ~ {x_col}", df, groups="sent_idx", re_formula="~0+roundtrip_bleu"
        )
        results = model.fit(method="nm")
        print(results.summary())

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
