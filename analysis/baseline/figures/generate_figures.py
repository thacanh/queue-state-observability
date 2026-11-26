"""Build the manuscript’s empirical figures from verified replay exports."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator
import numpy as np
import pandas as pd


INK = "#202832"
GRID = "#e0e5e8"
BLUE = "#17698c"
ORANGE = "#a94718"
GRAY = "#626971"
WHITE = "#ffffff"
WIDTH = 6.5
BUDGET = 0.005
OBS = ["oracle", "estimated"]
FIXED = "Fixed-DDQN"
GUARDED = "Fixed-DDQN+Guard"
META = {"Creator": "generate_figures.py", "CreationDate": None, "ModDate": None}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.2,
    "axes.titlesize": 9.0,
    "axes.titleweight": "semibold",
    "axes.titlepad": 7,
    "axes.labelsize": 8.4,
    "axes.labelcolor": INK,
    "axes.edgecolor": GRAY,
    "axes.linewidth": 0.65,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.major.width": 0.65,
    "ytick.major.width": 0.65,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "axes.unicode_minus": False,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "legend.fontsize": 8.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def source(root: Path, experiment: str, file: str) -> pd.DataFrame:
    return pd.read_csv(root / "outputs" / experiment / file)


def save(fig, output: Path, name: str) -> None:
    fig.savefig(output / (name + ".pdf"), metadata=META)
    plt.close(fig)


def panel_title(ax, label: str, title: str) -> None:
    ax.set_title(f"({label}) {title}", loc="left")


def verify_rates(frame: pd.DataFrame) -> None:
    assert (frame["n"] > 0).all()
    np.testing.assert_allclose(frame["violation_rate"], frame["violations"] / frame["n"],
                               atol=1e-14, rtol=0)
    np.testing.assert_array_equal(frame["met_budget"], frame["violation_rate"] <= BUDGET)


def guard_effect(root: Path, output: Path) -> None:
    replay = source(root, "guard_evaluation", "paired.csv")
    replay = replay[replay.method.isin([FIXED, GUARDED])]
    verify_rates(replay)
    units = ["day", "reconstruction_seed", "trace_sha256", "seed", "model_sha256"]
    paired = replay.pivot(index=units, columns=["observation", "method"], values="met_budget")
    assert paired.shape == (295, 4) and paired.notna().all().all()
    assert replay["reconstruction_seed"].nunique() == 59 and replay["seed"].nunique() == 5
    stats = replay.groupby(["observation", "method"]).agg(
        miss=("violation_rate", "mean"), successes=("met_budget", "sum"), pairs=("met_budget", "size"),
    )
    stats["attainment"] = stats["successes"] / stats["pairs"]

    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.55))
    fig.subplots_adjust(left=0.088, right=0.988, bottom=0.16, top=0.78, wspace=0.32)
    methods = (FIXED, GUARDED)
    styles = (
        dict(facecolor=WHITE, edgecolor=GRAY, hatch="////", linewidth=0.75),
        dict(facecolor=BLUE, edgecolor=BLUE, linewidth=0.75),
    )
    shifts = (-0.176, 0.176)
    x = np.arange(2)
    for ax in axes:
        ax.set_xticks(x, [s.title() for s in OBS])
        ax.grid(axis="y")
        ax.set_xlim(-0.54, 1.54)

    for m, delta, style in zip(methods, shifts, styles):
        rows = stats.loc[[(obs, m) for obs in OBS]]
        bars = axes[0].bar(x + delta, 100 * rows["miss"], width=0.303, **style)
        for bar in bars:
            top = bar.get_height()
            label_y = max(top + 0.014, 0.513) if 0.45 <= top <= 0.50 else top + 0.014
            axes[0].text(bar.get_x() + bar.get_width() / 2, label_y, f"{top:.3f}",
                         ha="center", va="bottom", fontsize=8.2)
        bar = axes[1].bar(x + delta, 100 * rows["attainment"], width=0.303, **style)
        axes[1].bar_label(bar, labels=[f"{k}/{n}" for k, n in zip(rows["successes"], rows["pairs"])],
                          padding=2.5, fontsize=8.1)
    axes[0].axhline(100 * BUDGET, linestyle=(0, (3, 2)), color=ORANGE, lw=1.0, zorder=3)
    axes[0].set(ylim=(0, 0.815), ylabel="Mean workload miss rate (%)")
    axes[0].set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    axes[1].set(ylim=(0, 110), ylabel="Seed-workload pairs passing (%)")
    axes[1].set_yticks([0, 25, 50, 75, 100])
    panel_title(axes[0], "a", "Deadline misses")
    panel_title(axes[1], "b", "Budget attainment")
    handles = [Patch(**styles[0], label="Fixed-DDQN"), Patch(**styles[1], label="Fixed-DDQN + guard"),
               Line2D([], [], color=ORANGE, linestyle=(0, (3, 2)), label="0.5% miss budget")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.50, 0.99), ncol=3)
    save(fig, output, "guard_effect")


def crossed(root: Path, output: Path) -> None:
    replay = source(root, "crossed_observations", "factorial_pairs.csv")
    verify_rates(replay)
    units = ["day", "reconstruction_seed", "trace_sha256", "seed", "model_sha256"]
    paired = replay.pivot(index=units, columns=["policy_obs", "guard_obs"], values="violation_rate")
    assert paired.shape == (50, 6) and paired.notna().all().all()
    replay = replay[replay.guard_obs != "-"]
    stats = replay.groupby(["policy_obs", "guard_obs"]).agg(
        miss=("violation_rate", "mean"), passes=("met_budget", "sum"), n=("met_budget", "size"),
    )
    index = pd.MultiIndex.from_product([["O", "E"], ["O", "E"]])
    stats = stats.reindex(index)
    data = 100 * stats["miss"].to_numpy().reshape(2, 2)
    contrasts = source(root, "crossed_observations", "factorial_contrasts.csv")
    contrasts = contrasts[contrasts.metric == "violation_rate"].set_index("comparison")
    ordered = [
        ("base shift, oracle guard", "DDQN shift\nOracle guard"),
        ("base shift, estimated guard", "DDQN shift\nEstimated guard"),
        ("guard shift, oracle policy", "Guard shift\nOracle DDQN"),
        ("guard shift, estimated policy", "Guard shift\nEstimated DDQN"),
        ("observation interaction", "Interaction"),
    ]
    runs = [
        paired["E", "O"] - paired["O", "O"],
        paired["E", "E"] - paired["O", "E"],
        paired["O", "E"] - paired["O", "O"],
        paired["E", "E"] - paired["E", "O"],
        (paired["E", "E"] - paired["E", "O"]) - (paired["O", "E"] - paired["O", "O"]),
    ]
    tab = contrasts.loc[[k for k, label in ordered]]
    np.testing.assert_allclose(tab["delta"], [arr.mean() for arr in runs], atol=1e-14, rtol=0)

    fig, (left, right) = plt.subplots(1, 2, figsize=(WIDTH, 2.73), gridspec_kw={"width_ratios": (0.96, 1.0)})
    fig.subplots_adjust(left=0.149, right=0.982, bottom=0.19, top=0.87, wspace=0.73)
    panel_title(left, "a", "Guarded replays")
    scale = Normalize(vmin=0.31, vmax=0.65)
    colors = matplotlib.colormaps["Blues"]
    left.set(xlim=(-0.5, 1.5), ylim=(1.5, -0.5))
    left.set_xticks([0, 1], ["Oracle", "Estimated"])
    left.set_yticks([0, 1], ["Oracle", "Estimated"])
    left.set_xlabel("Guard queue view")
    left.set_ylabel("DDQN queue view", labelpad=3)
    left.tick_params(length=0)
    for edge in left.spines.values():
        edge.set_visible(False)
    for position, (_, record) in enumerate(stats.iterrows()):
        y, x = divmod(position, 2)
        tint = scale(data[y, x])
        left.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor=colors(tint),
                                 edgecolor=WHITE, linewidth=1.1))
        font = WHITE if tint > 0.69 else INK
        left.text(x, y - 0.10, f"{100*record['miss']:.3f}%", ha="center", va="center",
                  color=font, fontsize=11.7, fontweight="bold")
        left.text(x, y + 0.16,
                  f"{int(record['passes'])}/{int(record['n'])} ({record['passes']/record['n']:.0%})",
                  ha="center", va="center", color=font, fontsize=8.0)

    panel_title(right, "b", "Paired effects")
    right.axvline(0, color=GRAY, linestyle=(0, (3, 2)), lw=0.85)
    colors = [BLUE, BLUE, GRAY, GRAY, ORANGE]
    markers = ["o", "o", "s", "s", "D"]
    for y, (_, row), c, marker in zip(range(len(ordered)), tab.iterrows(), colors, markers):
        est = 100 * row["delta"]
        low = 100 * row["conditional_p025"]
        high = 100 * row["conditional_p975"]
        right.errorbar(est, y, xerr=[[est - low], [high - est]], color=c, fmt=marker, markersize=4.3,
                       capsize=2.3, elinewidth=1.1)
    right.set_yticks(range(len(ordered)), [label for _, label in ordered], fontsize=7.55)
    right.tick_params(axis="y", length=0, pad=3)
    right.spines["left"].set_visible(False)
    right.set(xlim=(-0.004, 0.224), ylim=(4.5, -0.5), xlabel="Miss-rate change (pp)")
    right.set_xticks([0, 0.05, 0.10, 0.15, 0.20])
    right.grid(axis="x")
    save(fig, output, "crossed")


def workload(root: Path, output: Path) -> None:
    trace = source(root, "crossed_observations", "workload_by_trace.csv")
    queue = source(root, "crossed_observations", "queue_by_trace_seed.csv")
    assert trace.groupby("corpus").size().to_dict() == {2019: 59, 2021: 5}
    assert queue.groupby("corpus").size().to_dict() == {2019: 50, 2021: 25}
    np.testing.assert_allclose(trace["offered_load"], 0.6, rtol=0, atol=1e-10)
    corpus = [2019, 2021]
    trace_mean = trace.groupby("corpus").mean(numeric_only=True).loc[corpus]
    queue_mean = queue.groupby("corpus").mean(numeric_only=True).loc[corpus]
    plot = [
        ("Peak-to-mean arrivals", "60 s peak / mean", trace_mean["peak60_arrival_ratio"], ".2f", False),
        ("Peak offered demand", "60 s demand / capacity", trace_mean["peak60_offered_load"], ".2f", True),
        ("Service-demand tail", "P99 demand (s)", trace_mean["service_p99_s"], ".2f", False),
        ("Local queue backlog", "P95 queued requests", queue_mean["backlog_p95"], ",.1f", False),
        ("Most-loaded server", "Routed work / capacity", queue_mean["max_routed_server_load"], ".2f", True),
        ("Feature saturation", "Values at upper bound (%)", 100 * queue_mean["saturation_upper"], ".2f", False),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH, 3.38))
    fig.subplots_adjust(left=0.046, right=0.984, bottom=0.15, top=0.80, hspace=1.0, wspace=0.39)
    styles = (
        dict(facecolor=WHITE, edgecolor=GRAY, hatch="////", linewidth=0.75),
        dict(facecolor=BLUE, edgecolor=BLUE, linewidth=0.75),
    )
    for panel, ax, (title, xlabel, values, display, capacity) in zip("abcdef", axes.flat, plot):
        panel_title(ax, panel, title)
        for pos, val, style in zip([1, 0], values, styles):
            bar = ax.barh(pos, val, height=0.51, **style)
            ax.bar_label(bar, labels=[format(val, display)], fontsize=7.8, padding=3.0)
        if capacity:
            ax.axvline(1.0, linestyle=(0, (3, 2)), color=ORANGE, lw=0.9, zorder=3)
        ax.set(xlim=(0, 1.49 * max(values)), ylim=(-0.57, 1.6), xlabel=xlabel)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.grid(axis="x")
    fig.legend(handles=[Patch(**s, label=f"Azure {c}") for s, c in zip(styles, corpus)],
               loc="upper center", bbox_to_anchor=(0.50, 0.985), ncol=2)
    save(fig, output, "workload")


def transfer(root: Path, output: Path) -> None:
    replay = source(root, "crossed_observations", "transfer_2021_by_day_seed.csv")
    verify_rates(replay)
    assert not replay["met_budget"].any()
    trace = source(root, "crossed_observations", "workload_by_trace.csv")
    bounds = trace[trace.corpus == 2021].set_index("day")
    bounds.index = bounds.index.astype(int)
    bounds = bounds.sort_index()
    assert (bounds["pooled_bound_fraction"] >= bounds["intrinsic_fraction"]).all()
    np.testing.assert_allclose(bounds["intrinsic_fraction"], bounds["intrinsic_misses"] / bounds["requests"])
    np.testing.assert_allclose(bounds["pooled_bound_fraction"], bounds["pooled_bound_misses"] / bounds["requests"])
    replay = replay[replay.observation == "estimated"].copy()
    replay["p95_s"] = replay["p95_ms"] / 1000
    main = replay.groupby(["method", "day"])
    styles = [
        (FIXED, FIXED, BLUE, "o", "-"),
        (GUARDED, "Fixed-DDQN + guard", ORANGE, "s", "--"),
        ("RiskOnly", "RiskOnly", GRAY, "v", ":"),
    ]
    days = bounds.index
    fig, (miss, p95, bound) = plt.subplots(1, 3, figsize=(WIDTH, 2.55))
    fig.subplots_adjust(left=0.087, right=0.984, bottom=0.25, top=0.76, wspace=0.58)
    method_handles = []
    for method, label, color, marker, line in styles:
        rows = main[["violation_rate", "p95_s"]].mean().loc[method].reindex(days)
        counts = main.size().loc[method].reindex(days)
        assert counts.eq(1 if method == "RiskOnly" else 5).all()
        style = dict(color=color, marker=marker, linestyle=line, markersize=4.1, lw=1.05)
        handle, = miss.plot(days, 100 * rows["violation_rate"], label=label, **style)
        p95.plot(days, rows["p95_s"], **style)
        method_handles.append(handle)
    panel_title(miss, "a", "Observed misses")
    panel_title(p95, "b", "Response latency")
    panel_title(bound, "c", "Feasibility floors")
    miss.set(ylabel="Mean miss rate (%)", ylim=(0, 53))
    miss.set_yticks([0, 10, 20, 30, 40, 50])
    p95.set(yscale="log", ylabel="Mean workload P95 (s)", ylim=(5, 8100))
    p95.yaxis.set_major_locator(FixedLocator([10, 100, 1000]))
    p95.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    p95.minorticks_off()
    cap, = bound.plot(days, 100 * bounds["pooled_bound_fraction"], marker="s", color=GRAY, linestyle=":",
                      markersize=5.6, markerfacecolor=WHITE, lw=1.05, label="Pooled")
    intrinsic, = bound.plot(days, 100 * bounds["intrinsic_fraction"], marker="o", color=INK,
                            linestyle="none", markersize=3.0, label="Intrinsic")
    budget = bound.axhline(100 * BUDGET, linestyle=(0, (3, 2)), color=ORANGE, lw=1.0, label="0.5% budget")
    bound.set(ylabel="Necessary misses (%)", ylim=(0, 0.81))
    bound.set_yticks([0, 0.25, 0.50, 0.75])
    for ax in (miss, p95, bound):
        ax.set_xticks(days)
        ax.set_xlim(days.min() - 0.20, days.max() + 0.20)
        ax.grid(axis="y")
        ax.set_xlabel("Held-out day")
    fig.legend(handles=method_handles, loc="upper center", bbox_to_anchor=(0.37, 0.987), ncol=3,
               columnspacing=1.0, handletextpad=0.5)
    fig.legend(handles=[intrinsic, cap, budget], loc="lower right", bbox_to_anchor=(0.99, 0.015),
               ncol=3, columnspacing=0.85, handletextpad=0.35, fontsize=7.2)
    save(fig, output, "transfer")


def main() -> None:
    default_output = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, default=default_output.parent,
                        help="directory containing the outputs/guard_evaluation and outputs/crossed_observations evidence trees")
    parser.add_argument("--output-dir", type=Path, default=default_output)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for build in (guard_effect, crossed, workload, transfer):
        build(args.evidence_root, args.output_dir)


if __name__ == "__main__":
    main()
