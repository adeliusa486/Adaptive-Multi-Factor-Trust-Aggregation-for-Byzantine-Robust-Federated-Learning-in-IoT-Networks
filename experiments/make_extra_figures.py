import os
import glob
import re
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Upgrade to true publication-quality formatting (IEEE/Elsevier style)
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Bitstream Vera Serif", "Computer Modern"],
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
FIGURES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
os.makedirs(FIGURES_DIR, exist_ok=True)

METHOD_ORDER = ["fedavg", "trimmed_mean", "krum", "fltrust", "feddbc", "amfta"]
METHOD_LEGEND_LABEL = {
    "fedavg": "FedAvg",
    "trimmed_mean": "Trimmed Mean",
    "krum": "Krum",
    "fltrust": "FLTrust",
    "feddbc": "FedDBC",
    "amfta": "AMFTA",
}

METHOD_STYLE = {
    "fedavg":       {"color": "#7F7F7F", "marker": "o", "ls": "--", "lw": 1.6},
    "trimmed_mean": {"color": "#1F77B4", "marker": "s", "ls": "-.", "lw": 1.6},
    "krum":         {"color": "#2CA02C", "marker": "^", "ls": ":",  "lw": 1.8},
    "fltrust":      {"color": "#D62728", "marker": "v", "ls": ":",  "lw": 1.8},
    "feddbc":       {"color": "#FF7F0E", "marker": "D", "ls": "--", "lw": 1.6},
    "amfta":        {"color": "#800080", "marker": "P", "ls": "-",  "lw": 2.2},
}

PATTERN = re.compile(
    r"^(?P<method>[a-z_]+)_byz(?P<byz>[0-9.]+)_(?P<attack>[a-z_]+)_seed(?P<seed>\d+)_(?P<ts>\d{8}_\d{6})\.json$"
)

def apply_journal_axes_style(ax):
    ax.set_axisbelow(True)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5, color="#A0A0A0")
    ax.tick_params(direction="in", length=5, width=1.0, top=True, right=True, labelsize=10.5)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_edgecolor("black")

def collect_full_data():
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
    for key, (ts, path) in latest.items():
        try:
            rounds = json.load(open(path))
            grouped[key] = rounds
        except Exception:
            pass
    return grouped

def plot_convergence(grouped, attack, byz_rate, filename):
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    
    # We will average over seeds
    for method in METHOD_ORDER:
        rounds_data = defaultdict(list)
        for seed in [42, 123, 456, 789, 1024]:
            key = (method, byz_rate, attack, seed)
            if key in grouped:
                data = grouped[key]
                for r in data:
                    rounds_data[r["round"]].append(r["accuracy"] * 100)
        
        if not rounds_data:
            continue
            
        x_vals = sorted(list(rounds_data.keys()))
        y_vals = [np.mean(rounds_data[x]) for x in x_vals]
        
        style = METHOD_STYLE.get(method, {"color": "black", "marker": "o", "ls": "-", "lw": 1.5})
        
        ax.plot(
            x_vals, y_vals,
            label=METHOD_LEGEND_LABEL[method],
            color=style["color"], marker=style["marker"],
            linestyle=style["ls"], linewidth=style["lw"],
            markersize=5, markeredgecolor="black", markeredgewidth=0.5, markevery=3
        )

    apply_journal_axes_style(ax)
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_ylim(20, 102)
    ax.legend(loc="lower right", ncol=2, frameon=True, edgecolor="black", facecolor="white", framealpha=0.95)
    
    fig.savefig(os.path.join(FIGURES_DIR, filename), format="png")
    plt.close(fig)
    print(f"Generated {filename}")

def plot_metric_bar(grouped, metric, attack, byz_rate, filename, y_label):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    
    means = []
    errs = []
    valid_methods = []
    
    for method in METHOD_ORDER:
        vals = []
        for seed in [42, 123, 456, 789, 1024]:
            key = (method, byz_rate, attack, seed)
            if key in grouped:
                data = grouped[key][-5:] # avg of last 5 rounds
                for d in data:
                    vals.append(d.get(metric, 0.0) * 100)
        
        if vals:
            means.append(np.mean(vals))
            errs.append(np.std(vals))
            valid_methods.append(method)

    x = np.arange(len(valid_methods))
    width = 0.5
    
    apply_journal_axes_style(ax)
    
    # We use a nice professional blue with hatching
    ax.bar(x, means, width, yerr=errs, capsize=5,
           color="#1F77B4", edgecolor="black", linewidth=1.5, alpha=0.95, hatch='//')
    
    ax.set_ylabel(y_label)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LEGEND_LABEL[m] for m in valid_methods], rotation=0, ha="center")
    ax.set_ylim(0, 105)
    
    fig.savefig(os.path.join(FIGURES_DIR, filename), format="png")
    plt.close(fig)
    print(f"Generated {filename}")

if __name__ == "__main__":
    print("Parsing full json data...")
    grouped = collect_full_data()
    
    # Generate convergence plots
    plot_convergence(grouped, "label_flipping", 0.3, "fig_conv_label30.png")
    plot_convergence(grouped, "gaussian_noise", 0.3, "fig_conv_gauss30.png")
    
    # Generate F1 score bar charts
    plot_metric_bar(grouped, "f1", "label_flipping", 0.3, "fig_f1_label30.png", "F1 Score (%)")
    plot_metric_bar(grouped, "f1", "gaussian_noise", 0.3, "fig_f1_gauss30.png", "F1 Score (%)")
