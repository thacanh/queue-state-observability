"""Build the manuscript's single-column figures from the verified tables."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1] / "outputs"
RESULTS = ROOT / "training_evidence" / "results"
FIGURES = ROOT / "submission_assets"
METHODS = {
    "latency": ("Latency-4M", "#167462", "^"),
    "fixed": ("Fixed DDQN", "#a4471f", "s"),
    "adaptive": ("Adaptive DDQN", "#155688", "o"),
}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8,
    "axes.labelsize": 8, "axes.linewidth": .65,
    "axes.labelcolor": "#202833", "text.color": "#202833",
    "xtick.color": "#394653", "ytick.color": "#394653",
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "xtick.major.size": 2.7, "ytick.major.size": 2.7,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "legend.fontsize": 7,
})


def save(fig, filename):
    fig.savefig(FIGURES / f"{filename}.pdf")
    fig.savefig(FIGURES / f"{filename}.png", dpi=240)
    plt.close(fig)


def plot_training_tradeoff():
    heuristics = pd.read_csv(ROOT / "heuristics_validation_summary.csv").set_index("policy")
    rl = pd.read_csv(RESULTS / "matched_training_summary_verified.csv").set_index("method")
    if not heuristics["n"].eq(10).all() or not rl.paired_rows.eq(50).all():
        raise ValueError("Unexpected matched_training comparison bank")
    quantiles = [50, 75, 90, 95, 99]
    sed = heuristics.loc[[f"SED_q0.{q}" for q in quantiles]]
    means = pd.concat([
        heuristics[["p95_ms_mean", "violation_mean"]].assign(
            violation_mean=lambda data: 100 * data.violation_mean),
        rl[["p95_mean_ms", "violation_mean_pct"]].rename(
            columns={"p95_mean_ms": "p95_ms_mean", "violation_mean_pct": "violation_mean"}),
    ]).drop_duplicates().sort_values(["p95_ms_mean", "violation_mean"])
    lowest = means.violation_mean.cummin().shift(fill_value=np.inf)
    frontier = means[means.violation_mean < lowest]

    fig, ax = plt.subplots(figsize=(3.35, 2.6))
    fig.subplots_adjust(left=.17, right=.98, bottom=.17, top=.97)
    ax.set(xlim=(1646, 1825), ylim=(.378, .92), xticks=[1650, 1700, 1750, 1800],
           yticks=[.4, .5, .6, .7, .8, .9],
           xlabel="Mean workload P95 (ms)", ylabel="Mean violation (%)")
    ax.grid(axis="y", color="#e5eaf0", linewidth=.55)
    ax.set_axisbelow(True)
    line, = ax.plot(sed.p95_ms_mean, 100 * sed.violation_mean, color="#58697b",
                    linewidth=1.35, marker="|", markersize=5, label="SED quantile sweep")
    edge, = ax.plot(frontier.p95_ms_mean, frontier.violation_mean,
                    linestyle=(0, (3, 2)), color="#253847", linewidth=1.05,
                    label="Mean Pareto frontier")
    for q, offset in {50: (0, 6), 75: (4, 6), 90: (0, 5), 95: (0, 6), 99: (1, -11)}.items():
        row = sed.loc[f"SED_q0.{q}"]
        ax.annotate(f"P{q}", (row.p95_ms_mean, 100 * row.violation_mean),
                    xytext=offset, textcoords="offset points", ha="center",
                    color="#46586a", fontsize=6.8)

    markers = {}
    for method, (label, color, symbol) in METHODS.items():
        row = rl.loc[method]
        markers[method] = ax.errorbar(
            row.p95_mean_ms, row.violation_mean_pct, xerr=row.p95_seed_SD_ms,
            yerr=row.violation_seed_SD_pp, fmt=symbol, color=color,
            markersize=3.9 if method == "fixed" else 3.1,
            markerfacecolor="white" if method == "fixed" else color,
            elinewidth=1.15 if method == "fixed" else .85,
            capthick=.85, capsize=2.2 if method == "fixed" else 1.45,
            label=label, zorder=3,
        )
    ax.legend(handles=[line, edge, *markers.values()], loc="upper right",
              frameon=False, borderpad=.3, handlelength=2.1, labelspacing=.32)
    save(fig, "submission_sed_ddqn_tradeoff")


def plot_topology_load(stacked):
    data = pd.read_csv(RESULTS / "topology_feasibility.csv").query("target == 0.005")
    if len(data) != 36 or data.duplicated(["servers", "load", "method"]).any():
        raise ValueError("Unexpected topology comparison grid")
    shape = (3, 1) if stacked else (1, 3)
    size = (3.35, 4.7) if stacked else (5.15, 2.0)
    fig, axes = plt.subplots(*shape, figsize=size, sharex=True, sharey=True)
    if stacked:
        fig.subplots_adjust(left=.17, right=.97, bottom=.105, top=.88, hspace=.39)
    else:
        fig.subplots_adjust(left=.11, right=.988, bottom=.21, top=.75, wspace=.18)
    styles = {
        3: ("#155688", "-", "o"),
        5: ("#a4471f", (0, (4, 2)), "s"),
        8: ("#167462", (0, (1.2, 1.5)), "^"),
    }
    for ax, method in zip(axes, ("adaptive", "fixed", "latency")):
        for servers, (color, linestyle, marker) in styles.items():
            series = data[(data.method == method) & (data.servers == servers)].sort_values("load")
            ax.plot(100 * series.load, 100 * series.mean_violation, linestyle=linestyle,
                    color=color, marker=marker, markerfacecolor="white",
                    markersize=3.05, markeredgewidth=.8, linewidth=1.2)
        ax.set_title(METHODS[method][0], fontsize=8, loc="left", pad=4)
        ax.set(xlim=(37, 93), ylim=(-.65, 26.1), xticks=[40, 60, 80, 90],
               yticks=[0, 5, 10, 15, 20, 25])
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#e5eaf0", linewidth=.55)
    handles = [
        Line2D([], [], color=color, ls=linestyle, marker=marker, markerfacecolor="white",
               markersize=3.05, markeredgewidth=.8, lw=1.2, label=f"{servers} servers")
        for servers, (color, linestyle, marker) in styles.items()
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.56 if stacked else .55, .99),
               ncol=3, frameon=False, borderaxespad=0, handlelength=2.0,
               handletextpad=.45, columnspacing=1.55)
    fig.supxlabel("Offered load (%)", x=.57 if stacked else .55, y=.015 if stacked else .027)
    fig.supylabel("Mean violation (%)", x=.015, y=.49 if stacked else .48)
    save(fig, "submission_load_vertical" if stacked else "submission_load_horizontal")


def plot_function_misses():
    data = pd.read_csv(RESULTS / "matched_training_functions_by_seed_day_reconstruction.csv")
    pooled = data.groupby(["func_id", "method"], as_index=False)[["n", "violations"]].sum()
    pooled["rate"] = 100 * pooled.violations / pooled.n
    order = np.arange(30)
    adaptive = pooled.loc[pooled.method.eq("adaptive")].set_index("func_id").violations
    shares = (100 * adaptive / adaptive.sum()).reindex(order)
    columns = ["adaptive", "fixed", "latency"]
    rates = pooled.pivot(index="func_id", columns="method", values="rate")[columns].reindex(order)
    observed = shares.notna()
    if rates.loc[observed].isna().any().any() or not rates.loc[~observed].isna().all().all():
        raise ValueError("A function is missing paired method data")

    fig = plt.figure(figsize=(3.35, 2.85))
    key = fig.add_axes([.683, .934, .276, .036])
    palette = LinearSegmentedColormap.from_list(
        "deadline_misses", ["#b1c9dc", "#dae8f1", "#ffffff", "#e7b4aa", "#a12b37"])
    palette.set_bad("#eceef0")
    limit = np.ceil(2 * rates.max().max()) / 2
    for start, bottom in [(0, .665), (10, .396), (20, .127)]:
        heat = fig.add_axes([.184, bottom, .805, .208])
        band = rates.iloc[start:start + 10].to_numpy().T
        cells = heat.pcolormesh(np.arange(11), np.arange(4), np.ma.masked_invalid(band),
                                cmap=palette, vmin=0, vmax=limit,
                                edgecolors="white", linewidth=.43)
        heat.set(xlim=(0, 10), ylim=(3, 0))
        heat.set_yticks([.5, 1.5, 2.5], ["Adaptive", "Fixed", "Latency-4M"], fontsize=6.5)
        heat.set_xticks(np.arange(10) + .5, [f"{i:02d}" for i in order[start:start + 10]],
                        fontsize=7)
        heat.tick_params(axis="both", length=0, pad=3)
        heat.spines[["left", "bottom"]].set_visible(False)
        for row, rate_row in enumerate(band):
            for column, rate in enumerate(rate_row):
                missing = np.isnan(rate)
                rgb = cells.cmap(cells.norm(rate))[:3] if not missing else (.93, .94, .95)
                brightness = np.dot(rgb, [.2126, .7152, .0722])
                label = "–" if missing else (f"{rate:.2f}" if rate else "0")
                heat.text(column + .5, row + .5, label, fontsize=6.5,
                          color="white" if brightness < .55 else "#202833",
                          ha="center", va="center")
    cb = fig.colorbar(cells, cax=key, orientation="horizontal", ticks=[0, limit])
    cb.ax.tick_params(labelsize=6.4, length=1.8, pad=1.3)
    cb.outline.set_linewidth(.4)
    fig.text(.02, .932, "Deadline misses (% of requests)", fontsize=7.3,
             fontweight="semibold", va="center")
    fig.text(.586, .023, "Function ID", ha="center", fontsize=7.15)
    save(fig, "submission_function_miss_rates")

    fig = plt.figure(figsize=(3.35, 1.89))
    share_plot = fig.add_axes([.131, .225, .852, .616])
    fig.text(.021, .937, "Share of adaptive misses (%)", fontsize=7.3,
             fontweight="semibold", va="center")
    largest = set(shares.nlargest(5).index)
    ids = order[observed]
    share_plot.bar(ids, shares.loc[ids], width=.73, linewidth=0,
                   color=["#326886" if i in largest else "#a7bcc8" for i in ids])
    share_plot.scatter(order[~observed], [.45] * (~observed).sum(), marker="x",
                       s=9, linewidths=.7, color="#76828c")
    share_plot.set(xlim=(-.58, 29.6), ylim=(0, 20), yticks=[0, 5, 10, 15, 20],
                   xlabel="Function ID")
    share_plot.set_xticks([0, 5, 10, 15, 20, 25, 29], ["00", "05", "10", "15", "20", "25", "29"])
    share_plot.set_axisbelow(True)
    share_plot.grid(axis="y", color="#e5eaf0", linewidth=.55)
    save(fig, "submission_adaptive_miss_share")


if __name__ == "__main__":
    plot_training_tradeoff()
    plot_topology_load(stacked=False)
    plot_topology_load(stacked=True)
    plot_function_misses()
