# evaluate_all.py
"""
Phase 4 — Evaluation & Visualisation Script.
Produces 4 charts saved to outputs/plots/.
Usage: python evaluate_all.py
"""

import os, json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METRICS_DIR = os.path.join("outputs", "metrics")
PLOTS_DIR   = os.path.join("outputs", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

INDIVIDUAL_MODELS = {"Random Forest", "Isolation Forest", "ANN", "LSTM"}
DISPLAY_ORDER = [
    "Random Forest", "Isolation Forest", "ANN", "LSTM",
    "RF + ANN", "IF + ANN", "RF + LSTM", "ANN + LSTM + RF (Master)",
]
METRIC_COLOURS = {"accuracy":"#2196F3","precision":"#4CAF50","recall":"#FF9800","f1":"#E91E63"}
INDIVIDUAL_COLOUR = "#4C72B0"
HYBRID_COLOUR     = "#DD8452"


def load_metrics():
    records = []
    for fname in os.listdir(METRICS_DIR):
        if fname.endswith(".json"):
            with open(os.path.join(METRICS_DIR, fname)) as f:
                records.append(json.load(f))
    order_map = {n: i for i, n in enumerate(DISPLAY_ORDER)}
    records.sort(key=lambda r: order_map.get(r["model_name"], 99))
    return records


def _dark_fig(figsize):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#0F1117")
    ax.set_facecolor("#1A1D2E")
    return fig, ax


def _style_ax(ax, title="", ylabel="Score"):
    ax.set_title(title, color="white", fontsize=13, fontweight="bold", pad=16)
    ax.set_ylabel(ylabel, color="white", fontsize=10)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#333355")
    ax.yaxis.grid(True, color="#333355", linewidth=0.5, zorder=0)


def _save(fig, path):
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[Chart] Saved → {path}")


# ── Chart 1: Grouped Bar ──────────────────────────────────────────────────────
def plot_grouped_bar(records):
    names   = [r["model_name"] for r in records]
    metrics = ["accuracy","precision","recall","f1"]
    labels  = ["Accuracy","Precision","Recall","F1-Score"]
    offsets = [-1.5,-0.5,0.5,1.5]
    width   = 0.18
    x       = np.arange(len(names))

    fig, ax = _dark_fig((16, 7))
    for metric, label, offset in zip(metrics, labels, offsets):
        values = [r[metric] for r in records]
        bars = ax.bar(x + offset*width, values, width*0.92,
                      label=label, color=METRIC_COLOURS[metric], alpha=0.85, zorder=3)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
                    f"{val:.3f}", ha="center", va="bottom",
                    fontsize=6.5, color="white", rotation=90)

    n_ind = sum(1 for r in records if r["model_name"] in INDIVIDUAL_MODELS)
    if 0 < n_ind < len(records):
        ax.axvline(x=n_ind-0.5, color="#FFFFFF50", linewidth=1.5, linestyle="--")
        ax.text((n_ind-1)/2, 1.06, "Individual Models",
                ha="center", color="#4C72B0", fontsize=9, fontweight="bold",
                transform=ax.get_xaxis_transform())
        ax.text((n_ind+len(records)-1)/2, 1.06, "Hybrid / Ensemble Models",
                ha="center", color="#DD8452", fontsize=9, fontweight="bold",
                transform=ax.get_xaxis_transform())

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9, color="white")
    ax.set_ylim(0, 1.15)
    _style_ax(ax, title="Model Performance Comparison — Accuracy / Precision / Recall / F1")
    ax.legend(labels, loc="upper right", fontsize=9,
              facecolor="#1A1D2E", labelcolor="white", framealpha=0.8)
    _save(fig, os.path.join(PLOTS_DIR, "01_grouped_bar_comparison.png"))


# ── Chart 2: Radar ────────────────────────────────────────────────────────────
def plot_radar(records):
    metrics = ["accuracy","precision","recall","f1"]
    labels  = ["Accuracy","Precision","Recall","F1"]
    N = len(labels)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist() + [0]

    n_cols = 4
    n_rows = (len(records) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows),
                             subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("#0F1117")
    fig.suptitle("Model Performance — Radar Charts",
                 color="white", fontsize=14, fontweight="bold", y=1.01)
    flat_axes = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for ax, rec in zip(flat_axes, records):
        vals   = [rec[m] for m in metrics] + [rec[metrics[0]]]
        colour = HYBRID_COLOUR if rec["model_name"] not in INDIVIDUAL_MODELS else INDIVIDUAL_COLOUR
        ax.set_facecolor("#1A1D2E")
        ax.plot(angles, vals, color=colour, linewidth=2)
        ax.fill(angles, vals, color=colour, alpha=0.25)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2,0.4,0.6,0.8,1.0])
        ax.set_yticklabels(["0.2","0.4","0.6","0.8","1.0"], fontsize=6, color="#AAAAAA")
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=8.5, color="white")
        ax.spines["polar"].set_color("#333355")
        ax.grid(color="#333355", linewidth=0.6)
        tag = "HYBRID" if rec["model_name"] not in INDIVIDUAL_MODELS else "INDIV"
        ax.set_title(f"{rec['model_name']}\n[{tag}]  F1={rec['f1']:.3f}",
                     color=colour, fontsize=7.5, fontweight="bold", pad=12)

    for idx in range(len(records), n_rows*n_cols):
        flat_axes[idx].set_visible(False)

    _save(fig, os.path.join(PLOTS_DIR, "02_radar_chart.png"))


# ── Chart 3: Heatmap ─────────────────────────────────────────────────────────
def plot_heatmap(records):
    metrics = ["accuracy","precision","recall","f1"]
    labels  = ["Accuracy","Precision","Recall","F1-Score"]
    names   = [r["model_name"] for r in records]
    data    = np.array([[r[m] for m in metrics] for r in records])

    fig, ax = plt.subplots(figsize=(10, max(5, len(names)*0.75+1)))
    fig.patch.set_facecolor("#0F1117")
    ax.set_facecolor("#0F1117")
    im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, color="white", fontsize=11)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, color="white", fontsize=9)
    ax.tick_params(axis="both", length=0)

    for i in range(len(names)):
        for j in range(len(labels)):
            val = data[i,j]
            tc  = "black" if 0.35 < val < 0.85 else "white"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=9, color=tc, fontweight="bold")

    n_ind = sum(1 for r in records if r["model_name"] in INDIVIDUAL_MODELS)
    if 0 < n_ind < len(names):
        ax.axhline(y=n_ind-0.5, color="white", linewidth=2, linestyle="--")

    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
    cbar.set_label("Score", color="white")
    ax.set_title("Metrics Heatmap — All Models",
                 color="white", fontsize=13, fontweight="bold", pad=15)
    _save(fig, os.path.join(PLOTS_DIR, "03_metrics_heatmap.png"))


# ── Chart 4: Delta Improvement ───────────────────────────────────────────────
def plot_improvement_delta(records):
    individual = [r for r in records if r["model_name"] in INDIVIDUAL_MODELS]
    hybrids    = [r for r in records if r["model_name"] not in INDIVIDUAL_MODELS]
    if not individual or not hybrids:
        print("[Chart 4] Skipped.")
        return

    base_f1  = np.mean([r["f1"]       for r in individual])
    base_acc = np.mean([r["accuracy"] for r in individual])
    hnames    = [r["model_name"] for r in hybrids]
    delta_f1  = [r["f1"]       - base_f1  for r in hybrids]
    delta_acc = [r["accuracy"] - base_acc for r in hybrids]

    x = np.arange(len(hnames)); width = 0.35
    fig, ax = _dark_fig((12, 6))
    bars1 = ax.bar(x-width/2, delta_f1,  width, label="ΔF1-Score",
                   color="#E91E63", alpha=0.85, zorder=3)
    bars2 = ax.bar(x+width/2, delta_acc, width, label="ΔAccuracy",
                   color="#2196F3", alpha=0.85, zorder=3)
    ax.axhline(0, color="white", linewidth=1, linestyle="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(hnames, rotation=20, ha="right", fontsize=9, color="white")
    _style_ax(ax, title="Hybrid vs Individual Baseline — Performance Gain (Δ)",
              ylabel="Score Delta vs Individual Mean")
    ax.legend(facecolor="#1A1D2E", labelcolor="white", fontsize=10)
    for bars in (bars1, bars2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x()+bar.get_width()/2,
                    h+(0.001 if h>=0 else -0.005), f"{h:+.3f}",
                    ha="center", va="bottom" if h>=0 else "top",
                    fontsize=8, color="white")
    _save(fig, os.path.join(PLOTS_DIR, "04_improvement_delta.png"))


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("="*55)
    print("  PHASE 4 — EVALUATION & VISUALISATION")
    print("="*55)

    records = load_metrics()
    if not records:
        print("[ERROR] No metrics found. Run train_individual.py and train_hybrid.py first.")
        return

    print(f"\n[Eval] {len(records)} model metrics loaded.\n")
    print(f"{'Model':<35} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7}  {'Type'}")
    print("-"*75)
    for r in records:
        tag = "◆ HYBRID" if r["model_name"] not in INDIVIDUAL_MODELS else "  indiv."
        print(f"{r['model_name']:<35} {r['accuracy']:>7.4f} "
              f"{r['precision']:>7.4f} {r['recall']:>7.4f} "
              f"{r['f1']:>7.4f}  {tag}")

    plot_grouped_bar(records)
    plot_radar(records)
    plot_heatmap(records)
    plot_improvement_delta(records)

    print(f"\n[✓] All 4 charts saved to {PLOTS_DIR}/")
    print("    Next: streamlit run run_dashboard.py")


if __name__ == "__main__":
    main()
