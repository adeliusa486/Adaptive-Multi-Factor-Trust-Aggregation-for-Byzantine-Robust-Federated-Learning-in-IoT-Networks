"""
Academic Figure Generation Script for Federated Learning Experiments
=====================================================================
Generates publication-quality figures using standard Matplotlib settings
typical of IEEE/ACM conference and journal publications.
"""

import os
import glob
import json
import re
from collections import defaultdict
from statistics import mean, pstdev
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Use classic academic serif font and clean grid settings
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.6,
})

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

METHOD_ORDER = ["fedavg", "trimmed_mean", "krum", "fltrust", "feddbc", "amfta", "amfta_noq"]
METHOD_LABEL = {
    "fedavg": "FedAvg",
    "trimmed_mean": "Trimmed Mean",
    "krum": "Krum",
    "fltrust": "FLTrust",
    "feddbc": "FedDBC",
    "amfta": "AMFTA",
    "amfta_noq": "AMFTA-ND",
}

# Standard academic tab10 color palette and classic markers
METHOD_STYLE = {
    "fedavg":       {"color": "tab:gray",   "marker": "o", "ls": "--"},
    "trimmed_mean": {"color": "tab:blue",   "marker": "s", "ls": "-."},
    "krum":         {"color": "tab:green",  "marker": "^", "ls": ":"},
    "fltrust":      {"color": "tab:red",    "marker": "D", "ls": ":"},
    "feddbc":       {"color": "tab:orange", "marker": "v", "ls": "-"},
    "amfta":        {"color": "tab:purple", "marker": "P", "ls": "-"},
    "amfta_noq":    {"color": "tab:brown",  "marker": "*", "ls": "-"},
}

PATTERN = re.compile(
    r"^(?P<method>[a-z_]+)_byz(?P<byz>[0-9.]+)_(?P<attack>[a-z_]+)_seed(?P<seed>\d+)_(?P<ts>\d{8}_\d{6})\.json$"
)

def collect_data():
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

    grouped = defaultdict(dict)
    for (method, byz, attack, seed), (ts, path) in latest.items():
        try:
            rounds = json.load(open(path))
            if not isinstance(rounds, list) or not rounds:
                continue
            tail = rounds[-5:]
            avg_acc = sum(r["accuracy"] for r in tail if "accuracy" in r) / len(tail)
            grouped[(method, byz, attack)][seed] = avg_acc * 100.0
        except Exception as e:
            print(f"WARN: could not read {path}: {e}")
    return grouped

def compute_stats(grouped):
    stats = {}
    for (method, byz, attack), seeds_dict in grouped.items():
        vals = list(seeds_dict.values())
        if not vals:
            continue
        m_val = mean(vals)
        s_val = pstdev(vals) if len(vals) > 1 else 0.0
        stats[(method, byz, attack)] = (m_val, s_val, len(vals))
    return stats

def plot_line_chart(stats, attack_type, filename):
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    byz_rates = [0.10, 0.20, 0.30, 0.40]
    
    for method in METHOD_ORDER:
        if method not in METHOD_LABEL:
            continue
        style = METHOD_STYLE.get(method, {"color": "black", "marker": "o", "ls": "-"})
        
        x_vals, y_vals, y_errs = [], [], []
        for byz in byz_rates:
            if (method, byz, attack_type) in stats:
                m_val, s_val, _ = stats[(method, byz, attack_type)]
                x_vals.append(byz * 100)
                y_vals.append(m_val)
                y_errs.append(s_val)
        
        if x_vals:
            ax.errorbar(
                x_vals, y_vals, yerr=y_errs,
                label=METHOD_LABEL[method],
                color=style["color"], marker=style["marker"],
                linestyle=style["ls"], linewidth=1.5,
                capsize=3, markersize=6
            )
            
    ax.set_xlabel("Byzantine Client Fraction (%)")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_xticks([10, 20, 30, 40])
    ax.set_xticklabels(["10%", "20%", "30%", "40%"])
    ax.set_ylim(35, 102)
    
    # Standard academic legend placed below the plot area
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.18),
        ncol=4, frameon=True, columnspacing=1.0
    )
    
    fig.savefig(os.path.join(FIGURES_DIR, filename), format="pdf")
    fig.savefig(os.path.join(FIGURES_DIR, filename.replace(".pdf", ".png")), format="png")
    plt.close(fig)
    print(f"Generated {filename}")

def plot_bar_chart(stats, filename):
    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    byz = 0.30
    
    methods = [m for m in METHOD_ORDER if any((m, byz, att) in stats for att in ["label_flipping", "gaussian_noise"])]
    n_methods = len(methods)
    
    x = np.arange(n_methods)
    width = 0.35
    
    flip_means, flip_errs = [], []
    gauss_means, gauss_errs = [], []
    
    for m in methods:
        if (m, byz, "label_flipping") in stats:
            fm, fe, _ = stats[(m, byz, "label_flipping")]
            flip_means.append(fm)
            flip_errs.append(fe)
        else:
            flip_means.append(0.0)
            flip_errs.append(0.0)
            
        if (m, byz, "gaussian_noise") in stats:
            gm, ge, _ = stats[(m, byz, "gaussian_noise")]
            gauss_means.append(gm)
            gauss_errs.append(ge)
        else:
            gauss_means.append(0.0)
            gauss_errs.append(0.0)
            
    # Standard academic bar styling without floating text clutter
    ax.bar(x - width/2, flip_means, width, yerr=flip_errs, label="Label Flipping (30%)",
           color="tab:red", edgecolor="black", capsize=3, alpha=0.85)
    ax.bar(x + width/2, gauss_means, width, yerr=gauss_errs, label="Gaussian Noise (30%)",
           color="tab:blue", edgecolor="black", capsize=3, alpha=0.85)
    
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABEL[m] for m in methods], rotation=20, ha="right")
    ax.set_ylim(0, 105)
    
    # Simple, clean legend placed above the plot area
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=2, frameon=True)
    
    fig.savefig(os.path.join(FIGURES_DIR, filename), format="pdf")
    fig.savefig(os.path.join(FIGURES_DIR, filename.replace(".pdf", ".png")), format="png")
    plt.close(fig)
    print(f"Generated {filename}")

def print_summary_table(stats):
    print("\n" + "="*80)
    print("SUMMARY TABLE OF ALL EXPERIMENT RESULTS FOUND IN Phase 1 (results/)")
    print("="*80)
    print(f"{'Method':<15} | {'Attack Type':<16} | {'Byz Fraction':<12} | {'Accuracy (Mean ± Std)':<22} | {'Seeds':<5}")
    print("-"*80)
    
    for method in METHOD_ORDER:
        for attack in ["label_flipping", "gaussian_noise"]:
            for byz in [0.10, 0.20, 0.30, 0.40]:
                key = (method, byz, attack)
                if key in stats:
                    m_val, s_val, n_seeds = stats[key]
                    label = METHOD_LABEL.get(method, method)
                    print(f"{label:<15} | {attack:<16} | {byz*100:5.1f}%       | {m_val:6.2f}% ± {s_val:5.2f}%        | {n_seeds:<5}")
    print("="*80 + "\n")

if __name__ == "__main__":
    print("Collecting real data from results/...")
    grouped = collect_data()
    stats = compute_stats(grouped)
    
    print_summary_table(stats)
    
    print("Generating figures...")
    plot_line_chart(stats, "label_flipping", "fig_lineflip.pdf")
    plot_line_chart(stats, "gaussian_noise", "fig_linegauss.pdf")
    plot_bar_chart(stats, "fig_bar30.pdf")
    print("All figures successfully generated in figures/ directory.")
