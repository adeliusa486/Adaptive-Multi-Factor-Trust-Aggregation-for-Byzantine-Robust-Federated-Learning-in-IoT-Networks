"""
Advanced publication-quality figures for AMFTA paper.
Generates:
  1. fig_heatmap_label.png  - Accuracy heatmap (method x byz_rate) for label flipping
  2. fig_heatmap_gauss.png  - Accuracy heatmap (method x byz_rate) for gaussian noise
  3. fig_radar.png          - Radar/spider chart comparing all methods across 5 criteria
"""

import os
import glob
import re
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
from statistics import mean, pstdev

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
FIGURES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
os.makedirs(FIGURES_DIR, exist_ok=True)

METHOD_ORDER = ["fedavg", "trimmed_mean", "krum", "fltrust", "feddbc", "amfta"]
METHOD_LABEL = {
    "fedavg": "FedAvg",
    "trimmed_mean": "Trimmed\nMean",
    "krum": "Krum",
    "fltrust": "FLTrust",
    "feddbc": "FedDBC",
    "amfta": "AMFTA",
}

PATTERN = re.compile(
    r"^(?P<method>[a-z_]+)_byz(?P<byz>[0-9.]+)_(?P<attack>[a-z_]+)_seed(?P<seed>\d+)_(?P<ts>\d{8}_\d{6})\.json$"
)

def collect_stats():
    latest = {}
    for path in glob.glob(os.path.join(RESULTS_DIR, "*.json")):
        name = os.path.basename(path)
        m = PATTERN.match(name)
        if not m:
            continue
        key = (m["method"], float(m["byz"]), m["attack"], int(m["seed"]))
        ts = m["ts"]
        if key not in latest or ts > latest[key][0]:
            latest[key] = (ts, path)

    grouped = defaultdict(list)
    for (method, byz, attack, seed), (ts, path) in latest.items():
        try:
            rounds = json.load(open(path))
            if not isinstance(rounds, list) or not rounds:
                continue
            tail = rounds[-5:]
            avg_acc = sum(r["accuracy"] for r in tail if "accuracy" in r) / len(tail) * 100
            grouped[(method, byz, attack)].append(avg_acc)
        except Exception as e:
            print(f"WARN: {path}: {e}")

    stats = {}
    for key, vals in grouped.items():
        stats[key] = (mean(vals), pstdev(vals) if len(vals) > 1 else 0.0)
    return stats


def plot_heatmap(stats, attack, filename, title):
    byz_rates = [0.10, 0.20, 0.30, 0.40]
    methods = [m for m in METHOD_ORDER]

    data = np.zeros((len(methods), len(byz_rates)))
    annot = np.empty_like(data, dtype=object)

    for i, method in enumerate(methods):
        for j, byz in enumerate(byz_rates):
            key = (method, byz, attack)
            if key in stats:
                val, std = stats[key]
                data[i, j] = val
                annot[i, j] = f"{val:.1f}"
            else:
                data[i, j] = np.nan
                annot[i, j] = "N/A"

    fig, ax = plt.subplots(figsize=(7.0, 4.4))

    # Use a diverging colormap - red=bad, green=good
    cmap = matplotlib.colormaps["RdYlGn"]
    cmap.set_bad(color="#DDDDDD")

    masked_data = np.ma.masked_invalid(data)
    im = ax.imshow(masked_data, cmap=cmap, aspect="auto", vmin=40, vmax=100)

    # Annotate cells
    for i in range(len(methods)):
        for j in range(len(byz_rates)):
            val = data[i, j]
            text_color = "black" if 55 < val < 90 else "white"
            ax.text(j, i, annot[i, j], ha="center", va="center",
                    fontsize=11, fontweight="bold", color=text_color)

    # Axes
    ax.set_xticks(range(len(byz_rates)))
    ax.set_xticklabels([f"{int(b*100)}%" for b in byz_rates])
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([METHOD_LABEL[m].replace("\n", " ") for m in methods])
    ax.set_xlabel("Byzantine Fraction ($\\rho$)")
    ax.set_ylabel("Aggregation Method")

    # Draw grid lines between cells
    ax.set_xticks(np.arange(-0.5, len(byz_rates), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(methods), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Test Accuracy (%)", rotation=270, labelpad=15)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, filename), format="pdf")
    fig.savefig(os.path.join(FIGURES_DIR, filename.replace(".png", ".png")), format="png")
    plt.close(fig)
    print(f"Generated {filename}")


def plot_radar(stats):
    """
    Radar chart comparing methods across 5 criteria:
      1. Clean Accuracy (byz=0.0, none)
      2. Mild Attack Resilience (byz=0.10, avg of label_flipping+gaussian)
      3. Moderate Attack Resilience (byz=0.20)
      4. Heavy Attack Resilience (byz=0.30)
      5. Worst-case Stability (inverse of std at byz=0.30)
    """
    categories = [
        "Clean\nAccuracy",
        "10%\nAttack",
        "20%\nAttack",
        "30%\nAttack",
        "Stability\n(Low Var)"
    ]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    METHOD_STYLE = {
        "fedavg":       {"color": "#95A5A6", "ls": "--", "lw": 1.6},
        "trimmed_mean": {"color": "#3498DB", "ls": "-.", "lw": 1.6},
        "krum":         {"color": "#16A085", "ls": ":",  "lw": 1.8},
        "fltrust":      {"color": "#E74C3C", "ls": ":",  "lw": 1.8},
        "feddbc":       {"color": "#F39C12", "ls": "--", "lw": 1.6},
        "amfta":        {"color": "#2C3E50", "ls": "-",  "lw": 2.4},
    }

    fig, ax = plt.subplots(figsize=(6.2, 5.8), subplot_kw=dict(polar=True))

    for method in METHOD_ORDER:
        values = []

        # 1. Clean accuracy - use byz=0.0, none (or lowest available)
        clean_key = (method, 0.0, "none")
        if clean_key in stats:
            values.append(stats[clean_key][0])
        else:
            # fallback to 0.10
            v1 = stats.get((method, 0.10, "label_flipping"), (0, 0))[0]
            v2 = stats.get((method, 0.10, "gaussian_noise"), (0, 0))[0]
            values.append((v1 + v2) / 2)

        # 2. 10% attack resilience
        v1 = stats.get((method, 0.10, "label_flipping"), (0, 0))[0]
        v2 = stats.get((method, 0.10, "gaussian_noise"), (0, 0))[0]
        values.append((v1 + v2) / 2 if (v1 or v2) else 0)

        # 3. 20% attack resilience
        v1 = stats.get((method, 0.20, "label_flipping"), (0, 0))[0]
        v2 = stats.get((method, 0.20, "gaussian_noise"), (0, 0))[0]
        values.append((v1 + v2) / 2 if (v1 or v2) else 0)

        # 4. 30% attack resilience
        v1 = stats.get((method, 0.30, "label_flipping"), (0, 0))[0]
        v2 = stats.get((method, 0.30, "gaussian_noise"), (0, 0))[0]
        values.append((v1 + v2) / 2 if (v1 or v2) else 0)

        # 5. Stability = 100 - avg std at 30%
        s1 = stats.get((method, 0.30, "label_flipping"), (0, 0))[1]
        s2 = stats.get((method, 0.30, "gaussian_noise"), (0, 0))[1]
        avg_std = (s1 + s2) / 2
        stability = max(0, 100 - avg_std * 3)  # scale so high std = low score
        values.append(stability)

        # close the polygon
        values += values[:1]

        style = METHOD_STYLE[method]
        label = METHOD_LABEL.get(method, method).replace("\n", " ")

        ax.plot(angles, values, linestyle=style["ls"], linewidth=style["lw"],
                color=style["color"], label=label)
        ax.fill(angles, values, alpha=0.07, color=style["color"])

    # Category labels
    ax.set_thetagrids(np.degrees(angles[:-1]), categories, fontsize=10.5)
    ax.set_ylim(0, 105)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(["25", "50", "75", "100%"], fontsize=8.5)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.spines["polar"].set_visible(True)

    ax.legend(loc="upper right", bbox_to_anchor=(1.38, 1.18),
              frameon=True, edgecolor="black", facecolor="white",
              framealpha=0.95, fontsize=10)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig_radar.png"), format="pdf")
    fig.savefig(os.path.join(FIGURES_DIR, "fig_radar.png"), format="png")
    plt.close(fig)
    print("Generated fig_radar.png")


if __name__ == "__main__":
    print("Parsing simulation results...")
    stats = collect_stats()

    plot_heatmap(stats, "label_flipping", "fig_heatmap_label.png",
                 "Accuracy Heatmap — Label Flipping")
    plot_heatmap(stats, "gaussian_noise", "fig_heatmap_gauss.png",
                 "Accuracy Heatmap — Gaussian Noise")
    plot_radar(stats)
    print("All advanced figures generated.")
