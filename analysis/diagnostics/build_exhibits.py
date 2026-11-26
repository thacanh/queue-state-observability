"""Build manuscript exhibits from the verified study tables without replaying workloads."""

import csv
import json
import os
from hashlib import file_digest
from pathlib import Path
from statistics import mean

os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                  MKL_THREADING_LAYER="SEQUENTIAL", OPENBLAS_NUM_THREADS="1")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "outputs/supervisor_strengthening"
BUILD = ROOT / "outputs/ieee_access_full_review"
SOURCE = BUILD / "source"
sources = {}


def read(relative):
    relative = Path(relative)
    path = STUDY / relative
    with path.open("rb") as f:
        sources[relative.as_posix()] = file_digest(f, "sha256").hexdigest()
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def add_value(macros, name, value):
    macros.append("\\newcommand{\\" + name + "}{" + str(value) + "}")


def set_plot_style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelsize": 9, "axes.titlesize": 9,
                         "legend.fontsize": 8.2, "pdf.fonttype": 42})


def budget(macros):
    counts = read("archived_analysis/a1_attainment_by_day.csv")
    for bank, tag in (("d08–d09", "Original"), ("d12–d14", "Later")):
        for arm, mode in (("oracle_on", "Oracle"), ("estimated_on", "Estimated")):
            row = next(r for r in counts if (r["bank"], r["day"], r["arm"], r["budget_pct"]) ==
                       (bank, "all", arm, "1.0"))
            add_value(macros, "FullBudget" + tag + mode, row["passes"] + "/" + row["n"])
    values = read("archived_analysis/a1_exact_ecdf.csv")
    colors = {"oracle": "#165d82", "estimated": "#a34831"}
    fig, plots = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(3.52, 3.7),
                              layout="constrained")
    for i, (ax, bank) in enumerate(zip(plots, ("d08–d09", "d12–d14"))):
        for mode, color in colors.items():
            for setting, style in (("off", "--"), ("on", "-")):
                xs = [(float(r["miss_pct"]), float(r["attainment"])*100) for r in values
                      if r["bank"] == bank and r["day"] == "all" and r["arm"] == f"{mode}_{setting}"]
                low = max((y for x, y in xs if x <= .3), default=0)
                high = max((y for x, y in xs if x <= 1), default=0)
                pts = [(.3, low)] + [(x, y) for x, y in xs if .3 < x < 1] + [(1, high)]
                ax.step(*zip(*pts), where="post", lw=1.3, color=color, ls=style,
                        label=f"{mode.title()}, guard {setting}")
        ax.axvline(.5, lw=.7, ls=":", color=".35")
        ax.grid(axis="y", color=".90", lw=.6)
        ax.set(title=f"({'ab'[i]}) {bank}", ylabel="Attainment (%)", ylim=(-1, 102),
               xlim=(.3, 1.0), yticks=[0, 25, 50, 75, 100])
    plots[0].legend(loc="lower right", frameon=True, framealpha=.9, ncol=1,
                    labelspacing=.18, borderpad=.25)
    plots[-1].set_xlabel("Miss budget (%)")
    fig.savefig(SOURCE / "figures/full_budget.pdf")
    plt.close(fig)


def calibration(macros):
    rows = read("T3/fixed_trajectory_calibration.csv")
    scores = {k: [float(r["mae_ms"]) for r in rows if r["fit"] == k and r["subset"] == "all"]
              for k in ("original", "pooled_train")}
    assert all(len(v) == 21 for v in scores.values())
    factors = {r["factor"] for r in rows if r["fit"] == "pooled_train"}
    assert len(factors) == 1
    add_value(macros, "FullWaitFactor", f"{float(factors.pop()):.5f}")
    add_value(macros, "FullWaitMAEBefore", f"{mean(scores['original']):.2f}")
    add_value(macros, "FullWaitMAEAfter", f"{mean(scores['pooled_train']):.2f}")
    rows = read("processed/t3_by_day.csv")
    points = {}
    for mode in ("oracle", "estimated"):
        for corrected in ("False", "True"):
            b = [r for r in rows if (r["mode"],r["guard"],r["corrected"]) == (mode,"True",corrected)]
            assert len(b) == 3 and set(r["replays"] for r in b) == {"4"}
            points[mode, corrected] = mean(float(r["p95_ms"]) for r in b), 100*mean(float(r["violation_rate"]) for r in b)
    for tag, corrected in (("Before", "False"), ("After", "True")):
        add_value(macros, "FullWaitGap" + tag,
                  f"{points['oracle',corrected][1]-points['estimated',corrected][1]:.4f}")
    fig, ax = plt.subplots(figsize=(3.53, 2.32), layout="constrained")
    for mode, color in (("oracle", "#165d82"), ("estimated", "#a34831")):
        a, b = points[mode, "False"], points[mode, "True"]
        ax.scatter(*a, marker="o", s=32, facecolors="white", edgecolors=color, zorder=4)
        ax.scatter(*b, marker="o", s=32, c=color, zorder=4)
        ax.annotate("", xy=b, xytext=a, arrowprops={"arrowstyle": "->", "lw": 1.2, "color": color,
                                                    "shrinkA": 4, "shrinkB": 5})
        ax.text(a[0]+1, a[1]+.010, mode.title() + " FT", color=color, fontsize=8.5)
    ax.set(xlabel="Mean workload P95 (ms)", ylabel="Mean miss rate (%)",
           ylim=(.58, .89), xlim=(1548, 1602))
    ax.grid(axis="y", color=".90", lw=.6)
    fig.savefig(SOURCE / "figures/full_calibration.pdf")
    plt.close(fig)


def tables(macros):
    contrasts = read("processed/t4_paired_by_bank.csv")
    arms = read("processed/t4_arms_by_bank.csv")
    cal_rows = []
    for key, day in (("Original", "d08--d09"), ("Later", "d12--d14")):
        for guard, label in (("False", "Off"), ("True", "On")):
            outcome = next(r for r in contrasts if (r["bank"],r["guard"],r["contrast"]) ==
                           (key,guard,"Estimated Numba minus Oracle Numba"))
            a, b = (next(r for r in arms if (r["bank"],r["guard"],r["observation"],r["reference_reused"]) ==
                         (key,guard,mode,"False")) for mode in ("oracle","estimated"))
            assert a["replays"] == b["replays"] == outcome["pairs"]
            line = (f"{day} & {label} & {100*float(outcome['delta_violation_rate']):+.4f} & "
                    f"{float(outcome['delta_p95_ms']):+.2f} & {a['passes']}/{a['replays']} "
                    rf"$\to$ {b['passes']}/{b['replays']}")
            cal_rows.append(line + r" \\")
    add_value(macros, "FullCalendarRows", "%\n" + "\n".join(cal_rows) + "\n")
    arms = read("processed/t8_arms.csv")
    contrast = read("processed/t8_paired_detail.csv")
    assert len(contrast) == 8
    assert all(float(r["delta_violation_rate"]) > 0 and float(r["delta_p95_ms"]) > 0 for r in contrast)
    assert sum(float(r["delta_p99_ms"]) < 0 for r in contrast) == 3
    assert all(r["passes"] == "0" and r["replays"] == "2" for r in arms)
    comparisons = []
    for day in ("7", "8"):
        for guard, label in (("False", "Off"), ("True", "On")):
            a, b = (next(r for r in arms if (r["day"],r["guard"],r["reference_reused"]) ==
                         (day,guard,archive)) for archive in ("True","False"))
            values = []
            for metric, scale in (("violation_rate",100),("p95_ms",.001)):
                val = [float(r[metric])*scale for r in (a,b)]
                values.append(r" $\to$ ".join(f"{v:.{3 if v < 1 and metric=='p95_ms' else 2}f}" for v in val))
            comparisons.append(" & ".join([day,label,*values]) + r" \\")
    add_value(macros, "FullTransferRows", "%\n" + "\n".join(comparisons) + "\n")


def events(macros):
    all_events = read("T6/replays.csv")
    count = sum(int(row["n"]) for row in all_events)
    assert len(all_events) == 40 and count == 871700 and all(r["archive_match"]=="True" for r in all_events)
    add_value(macros, "FullEventCount", f"{count:,}")
    data = read("processed/t6_whole_trajectory_and_snapshot_by_seed.csv")
    floors = read("archived_analysis/a3_workload_and_feasibility.csv")
    for (digit, name) in (("7","Seven"),("8","Eight")):
        rows = [r for r in data if r["day"] == digit and r["mode"] == "estimated"]
        assert len(rows) == 5
        for macro, field in (("LocalSaves","direct_saves"),("LocalLosses","direct_losses"),
                             ("TrajectoryMiss","total_miss_delta")):
            add_value(macros, "Full"+macro+name, f"{sum(int(r[field]) for r in rows):,}")
        floor = next(r for r in floors if (r["day"],r["corpus"],r["bank"]) == (digit,"2021","reserved"))
        for macro, field in (("Pooled","pooled_bound_fraction"),("Intrinsic","intrinsic_fraction")):
            add_value(macros, "Full"+macro+name, f"{float(floor[field])*100:.4f}")


if __name__ == "__main__":
    definitions = []
    set_plot_style()
    budget(definitions)
    calibration(definitions)
    tables(definitions)
    events(definitions)
    (SOURCE/"full_values.tex").write_text("\n".join(definitions)+"\n", encoding="utf-8")
    (BUILD/"exhibit_input_checksums.json").write_text(json.dumps(sources, indent=2)+"\n", encoding="utf-8")
