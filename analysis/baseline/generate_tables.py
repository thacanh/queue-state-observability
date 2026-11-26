"""Generate the LaTeX evidence rows from archived experiment records."""
import argparse
from pathlib import Path

import pandas as pd


def make_tables(root, output):
    evidence = root / "outputs"
    tables = {}

    def read(name, **kwargs):
        return pd.read_csv(evidence / name, **kwargs)

    def add(name, rows):
        tables[name] = "\n".join(" & ".join(map(str, cells)) + r" \\" for cells in rows)

    matched_training = read("training_evidence/matched_training/confirmation_paired.csv")
    assert len(matched_training) == 150 and matched_training.groupby(["method", "seed"]).size().eq(10).all()
    metrics = ["mean_ms", "p95_ms", "p99_ms", "violation_rate"]
    model_order = [("latency", "Latency-only"), ("fixed", r"Fixed $\lambda=2$"),
                   ("adaptive", r"Adaptive $\lambda$")]
    e2rows = []
    for method, label in model_order:
        trials = matched_training.loc[matched_training.method.eq(method)]
        seed_means = trials.groupby("seed")[metrics].mean()
        seed_means["violation_rate"] *= 100
        e2rows.append([label, *(f"{seed_means[c].mean():.{p}f} ({seed_means[c].std(ddof=1):.{p}f})"
                               for c, p in zip(metrics, [1, 1, 1, 4])),
                       f"{trials.violation_rate.le(.005).sum()}/50"])
    add("MatchedRows", e2rows)

    policies = ["Fixed-DDQN", "Fixed-DDQN+Guard", "RiskOnly"] + [
        "SED-" + q + g for q in ("P50", "P95", "P99") for g in ("", "+Guard")
    ]
    for name, file, scale, decimals in (
        ("GuardRows", "guard_evaluation/paired.csv", 1, 4),
        ("TransferRows", "guard_evaluation/external/paired_2021.csv", .001, 2),
    ):
        result = read(file)
        table_rows = []
        for method in policies:
            cells = [method.replace("+Guard", " + guard")]
            for mode in ("oracle", "estimated"):
                unit = result.loc[result.method.eq(method) & result.observation.eq(mode)]
                assert len(unit) in (5, 25, 59, 295)
                cells.extend([f"{100*unit.violation_rate.mean():.{decimals}f}",
                              f"{scale*unit.p95_ms.mean():.1f}"])
                if name == "GuardRows":
                    cells.append(f"{unit.met_budget.sum()}/{len(unit)}")
                else:
                    assert not unit.met_budget.any()
            table_rows.append(cells)
        add(name, table_rows)

    crossed = read("crossed_observations/factorial_pairs.csv", keep_default_na=False)
    labels = {"O": "Oracle", "E": "Estimated", "-": "Off"}
    rows = []
    for base, guard in (("O", "-"), ("E", "-"), ("O", "O"), ("O", "E"), ("E", "O"), ("E", "E")):
        subset = crossed.loc[crossed.policy_obs.eq(base) & crossed.guard_obs.eq(guard)]
        assert len(subset) == 50
        rows.append([labels[base], labels[guard] + (r"$^\dagger$" if guard != "-" and guard != base else ""),
                     f"{100*subset.violation_rate.mean():.3f}", f"{subset.p95_ms.mean():.1f}",
                     f"{subset.met_budget.sum()}/50"])
    add("CrossedRows", rows)
    bounds = read("crossed_observations/feasibility_2021_by_day.csv")
    add("BoundRows", [[t.day, f"{t.requests:,}", f"{t.time_multiplier:.3f}", f"{t.peak60_offered_load:.2f}",
                       f"{100*t.intrinsic_fraction:.3f}", f"{100*t.pooled_bound_fraction:.3f}"]
                      for t in bounds.itertuples()])

    characterization = read("crossed_observations/workload_by_trace.csv")
    telemetry = read("crossed_observations/queue_by_trace_seed.csv")
    telemetry["peak_util"] = telemetry[[f"server_{j}_util" for j in range(1, 4)]].max(axis=1)
    corpus = characterization.groupby("corpus").mean(numeric_only=True)
    corpus["service_max_s"] = characterization.groupby("corpus").service_max_s.max()
    corpus = corpus.join(telemetry.groupby("corpus").mean(numeric_only=True), rsuffix="_queue")
    add("WorkloadRows", [[label, *(f"{scale*corpus.loc[year, field]:,.{p}f}" for year in (2019, 2021))]
                         for label, field, p, scale in [
        ("Mean arrivals per second", "arrivals_per_s", 2, 1),
        ("Peak/mean arrivals, 60 s", "peak60_arrival_ratio", 2, 1),
        ("Peak demand/pool capacity, 60 s", "peak60_offered_load", 2, 1),
        ("Mean demand (s)", "service_mean_s", 3, 1),
        ("Demand P50 (s)", "service_p50_s", 3, 1),
        ("Demand P95 (s)", "service_p95_s", 2, 1),
        ("Demand P99 (s)", "service_p99_s", 2, 1),
        ("Maximum observed demand (s)", "service_max_s", 2, 1),
        ("Deadline P05 (s)", "slo_p05_s", 2, 1),
        ("Deadline P50 (s)", "slo_p50_s", 2, 1),
        ("Deadline P95 (s)", "slo_p95_s", 2, 1),
        ("P95 queue backlog (requests)", "backlog_p95", 1, 1),
        (r"Busiest server utilization (\%)", "peak_util", 1, 100),
        ("Maximum individual routed load", "max_routed_server_load", 2, 1),
        (r"Dynamic inputs at upper bound (\%)", "saturation_upper", 2, 100),
        (r"Dynamic inputs at lower bound (\%)", "saturation_lower", 2, 100),
    ]])
    loop = read("guard_evaluation/loopback.csv")
    add("CostRows", [[label.replace("+Guard", " + guard"),
                      *(f"{value:.1f}" for value in loop.loc[loop.method.eq(label),
                       [f"per_decision_us_p{q}" for q in (50, 95, 99)]].mean())]
                     for label in ("Fixed-DDQN", "Fixed-DDQN+Guard", "RiskOnly", "SED-P99", "SED-P99+Guard")])
    stress = read("guard_evaluation/diagnostics/robustness.csv")
    add("StressRows", [[label, *(f"{100*stress.loc[stress.scenario.eq(case) & stress.method.eq(method), 'violation_rate'].mean():.2f}"
                              for method in ("Fixed-DDQN", "Fixed-DDQN+Guard", "RiskOnly", "SED-P99+Guard"))]
                       for case, label in (("load40", r"Reference, 40\%"), ("load80", r"Reference, 80\%"),
                                           ("load90", r"Reference, 90\%"), ("R1", r"Burst, 60\%"),
                                           ("R3", r"Heavy tail, 60\%"))])
    text = "\n".join("\\newcommand{\\" + name + "}{%\n" + rows + "\n}\n" for name, rows in tables.items())
    if "nan" in text:
        raise ValueError("A configured method or scenario is absent from the source")
    output.mkdir(parents=True, exist_ok=True)
    (output / "evidence_tables.tex").write_text(text, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    make_tables(args.evidence_root, args.output_dir)
