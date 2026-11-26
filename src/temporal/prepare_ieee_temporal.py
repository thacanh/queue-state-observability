"""Audit archived outcomes independently and build the temporal publication exhibits."""

import argparse
from collections import Counter, defaultdict
import csv
from hashlib import sha256
from itertools import product
import json
import math
from pathlib import Path, PureWindowsPath

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "outputs/temporal_evaluation"
DRAFT = ROOT / "outputs/ieee_access_temporal"
DAYS = ("d12", "d13", "d14")
SEEDS = (42, 43, 44, 45, 46)
BASE_ARMS = ("oracle_off", "oracle_on", "estimated_off", "estimated_on")
CROSS_ARMS = ("O-", "E-", "OO", "OE", "EO", "EE")
FIELDS = ("met_budget", "violation_rate", "mean_ms", "p95_ms", "p99_ms")
DISPLAYS = ("attainment_pp", "miss_delta_pp", "mean_delta_ms", "p95_delta_ms", "p99_delta_ms")
UNITS = (100, 100, 1, 1, 1)
PRETTY = ("Oracle, off", "Oracle, on", "Estimated, off", "Estimated, on")


def digest(path):
    with path.open("rb") as stream:
        from hashlib import file_digest
        return file_digest(stream, "sha256").hexdigest()


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def records(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def avg(values):
    values = list(values)
    return math.fsum(values) / len(values)


def close(actual, expected, tolerance=1e-8):
    if not math.isclose(float(actual), float(expected), rel_tol=2e-12, abs_tol=tolerance):
        raise ValueError(f"Audit mismatch: {actual!r} versus {expected!r}")


def row_values(row):
    return np.array([float(row[k]) for k in FIELDS])


def variant(row, phase):
    if phase == "D":
        return row["arm"] if "arm" in row else row["policy_obs"] + row["guard_obs"]
    if "arm" in row:
        return row["arm"]
    mode = "oracle" if row.get("policy_training", row.get("training")) == "oracle" else "estimated"
    return mode + ("_on" if row["guard"] == "True" else "_off")


def read_runs(path, phase):
    rows = records(path)
    for row in rows:
        row["arm"] = variant(row, phase)
        row["seed"] = int(row["seed"])
        row["reconstruction_seed"] = int(row.get("reconstruction_seed", -1))
        row["n"] = int(row["n"])
        row["violations"] = int(row["violations"])
        pass_budget = (200 * row["violations"] <= row["n"])
        assert pass_budget == (row["met_budget"] == "True")
        row["met_budget"] = int(pass_budget)
        close(float(row["violation_rate"]), row["violations"] / row["n"], 1e-12)
        row["violation_rate"] = row["violations"] / row["n"]
        row["override_rate"] = float(row["override_rate"]) if "override_rate" in row else int(row["overrides"]) / row["n"]
        for k in ("mean_ms", "p95_ms", "p99_ms"):
            row[k] = float(row[k])
        assert row["p99_ms"] >= row["p95_ms"] >= 0
    return rows


def summarize(rows):
    return dict(n=len(rows), passed=sum(r["met_budget"] for r in rows),
                miss_pct=100 * avg(r["violation_rate"] for r in rows),
                p95_ms=avg(r["p95_ms"] for r in rows),
                mean_ms=avg(r["mean_ms"] for r in rows),
                p99_ms=avg(r["p99_ms"] for r in rows),
                override_pct=100 * avg(r["override_rate"] for r in rows))


def aggregate(rows, grouping):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[c] for c in grouping)].append(row)
    return {key: summarize(value) for key, value in groups.items()}


def audit_aggregates(evidence, rows, published):
    columns = {"arms": ["arm"], "day_arms": ["day", "arm"], "day_seed_arms": ["day", "seed", "arm"]}
    checked = 0
    for key, grouping in columns.items():
        actual = aggregate(rows, grouping)
        saved = records(evidence / f"analysis/{published}_{key}.csv")
        assert len(saved) == len(actual)
        for rec in saved:
            idx = tuple(int(rec[k]) if k == "seed" else rec[k] for k in grouping)
            a = actual[idx]
            assert a["n"] == int(rec["pairs"]) and a["passed"] == int(rec["passes"])
            close(100 * a["passed"] / a["n"], float(rec["attainment_pct"]))
            for name in ("p95_ms", "mean_ms", "p99_ms", "override_pct"):
                close(a[name], float(rec[name]))
            close(a["miss_pct"], float(rec["mean_miss_pct"]))
            checked += 8
    return checked


def effects(evidence, rows, phase, published, rules):
    arms = defaultdict(dict)
    sha_by_unit = {}
    for row in rows:
        unit = row["day"], row["seed"], row["reconstruction_seed"]
        assert row["trace_sha256"] == sha_by_unit.setdefault(unit, row["trace_sha256"])
        assert row["arm"] not in arms[unit]
        arms[unit][row["arm"]] = row
    definitions = rules["D_contrasts"] if phase == "D" else rules["B_transfer_contrasts"]
    pairs = {
        label: {k: (by_arm[a], by_arm[b]) for k, by_arm in arms.items()}
        for label, (a, b) in definitions.items()
    }
    differences = {
        label: {unit: row_values(a) - row_values(b) for unit, (a, b) in entries.items()}
        for label, entries in pairs.items()
    }
    if phase != "transfer":
        a, b, interaction = (
            ("guard shift, estimated policy", "guard shift, oracle policy", "observation interaction")
            if phase == "D" else ("guard_estimated_training", "guard_oracle_training", "training_by_guard_interaction")
        )
        differences[interaction] = {unit: differences[a][unit] - differences[b][unit] for unit in arms}
    saved = {(r["day"], r["contrast"]): r for r in records(evidence / f"analysis/{published}_paired_contrasts.csv")}
    ext = {}
    days = sorted({u[0] for u in arms})
    intervals = {}
    draws_by_day = {day: sorted({k[2] for k in arms if k[0] == day}) for day in days}
    for label, values in differences.items():
        for day in ["all", *days]:
            subset = [k for k in values if day == "all" or day == k[0]]
            stats = dict(zip(DISPLAYS, np.mean([values[k] for k in subset], axis=0) * UNITS))
            stats["pairs"] = len(subset)
            if label in pairs:
                pass_counts = Counter((pairs[label][k][1]["met_budget"], pairs[label][k][0]["met_budget"]) for k in subset)
                stats |= dict(fail_to_pass=pass_counts[0, 1], pass_to_fail=pass_counts[1, 0],
                              both_fail=pass_counts[0, 0], both_pass=pass_counts[1, 1],
                              lower_miss=sum(values[k][1] < 0 for k in subset),
                              higher_miss=sum(values[k][1] > 0 for k in subset),
                              lower_p95=sum(values[k][3] < 0 for k in subset),
                              higher_p95=sum(values[k][3] > 0 for k in subset),
                              lower_p95_higher_miss=sum(values[k][3] < 0 < values[k][1] for k in subset))
            for metric, value in stats.items():
                close(value, saved[day, label][metric])
            ext[day, label] = stats
        if phase == "transfer":
            continue
        matrices = [
            np.array([[values[day, seed, draw] for draw in draws_by_day[day]] for seed in SEEDS])
            for day in days
        ]
        rng = np.random.default_rng(20261124)
        boot = []
        for _ in range(4000):
            picked_seeds = rng.integers(5, size=5)
            chunks = []
            for i in rng.integers(len(days), size=len(days)):
                vals = matrices[i]
                picked_draws = rng.integers(vals.shape[1], size=vals.shape[1])
                chunks.append(vals[np.ix_(picked_seeds, picked_draws)].reshape(-1, 5))
            boot.append(np.vstack(chunks).mean(0))
        low, high = np.quantile(boot, [.025, .975], axis=0) * np.array(UNITS)
        intervals[label] = {metric: (lo, hi) for metric, lo, hi in zip(DISPLAYS, low, high, strict=True)}
    if phase != "transfer":
        for row in records(evidence / f"analysis/{published}_conditional_intervals.csv"):
            lo, hi = intervals[row["contrast"]][row["metric"]]
            close(lo, row["lower"])
            close(hi, row["upper"])
            close(ext["all", row["contrast"]][row["metric"]], row["delta"])
    return ext, intervals


def portable(path, ref):
    """Check frozen absolute file references at their packaged, relative locations."""
    return path.is_file() and path.stat().st_size == ref["bytes"] and digest(path) == ref["sha256"]


def verify_design(evidence, protocol, runs, archived_only=False):
    inputs_2019 = evidence / "inputs/2019_R0"
    inputs_2021 = evidence / "inputs/2021_reserved"
    bank_2019 = load(inputs_2019 / "manifest.json")["traces"]
    bank_2021 = load(inputs_2021 / "manifest.json")["traces"]
    phases = protocol["phases"]
    assert phases["D"]["workloads"] == [r for r in phases["B"]["workloads"] if r["draw_index"] < 5]
    assert len(bank_2019) == 30 and set(bank_2021) == {"elapsed07", "elapsed08"}
    sets = {}
    for phase, rows in runs.items():
        spec = phases[phase]
        path = evidence / phase
        manifest = load(path / "result_manifest.json")
        status = load(path / "execution.json")
        assert status["status"] == "COMPLETE" and status["wall_s"] <= status["approved_wall_s"]
        assert status["result_manifest_sha256"] == digest(path / "result_manifest.json")
        assert manifest["protocol_sha256"] == protocol["protocol_sha256"]
        assert portable(path / "replays.csv", manifest["output"])
        assert all(portable(path / "shards" / PureWindowsPath(r["path"]).name, r) for r in manifest["shards"])
        expected = {(r["key"], seed, arm) for r in spec["workloads"] for seed in SEEDS for arm in spec["arms"]}
        assert len(rows) == spec["runs"] == len(expected)
        assert {(r["key"], r["seed"], r["arm"]) for r in rows} == expected
        bank = bank_2021 if phase == "transfer" else bank_2019
        root = inputs_2021 if phase == "transfer" else inputs_2019
        for workload in spec["workloads"]:
            rec = bank[workload["key"]]
            workload_path = root / rec["filename"]
            if not archived_only or workload_path.is_file():
                assert digest(workload_path) == rec["file_sha256"]
        configs = {}
        for row in rows:
            arm = row["arm"]
            trace = bank[row["key"]]
            assert row["trace_sha256"] == trace["trace_sha256"]
            assert row["n"] == trace["meta"]["requests"]
            if phase == "D":
                mode, policy, guard = "oracle", arm[0], arm[1]
            else:
                mode, on = arm.split("_")
                policy, guard = "E", ("E" if on == "on" else "-")
            model = protocol["models"][f"{mode}/{row['seed']}"]
            assert row["model_sha256"] == model["sha256"]
            key = (row["trace_sha256"], row["model_sha256"], policy, guard)
            assert key not in configs
            configs[key] = row
        sets[phase] = configs
    overlap = sorted(sets["B"].keys() & sets["D"].keys())
    assert len(overlap) == 150
    assert not (sets["transfer"].keys() & (sets["B"].keys() | sets["D"].keys()))
    overlap_audit = []
    for key in overlap:
        a, b = sets["B"][key], sets["D"][key]
        assert (a["arm"], b["arm"]) in {("oracle_off", "E-"), ("oracle_on", "EE")}
        assert a["key"] == b["key"] and a["seed"] == b["seed"]
        for name in ("n", "violations", *FIELDS):
            assert a[name] == b[name]
        overlap_audit.append(dict(workload=a["key"], seed=a["seed"], B=a["arm"], D=b["arm"],
                                  trace_sha256=key[0], checkpoint_sha256=key[1]))
    assert len(set.union(*(set(group) for group in sets.values()))) == 940
    return overlap_audit


def audit_floor(evidence):
    data = records(evidence / "analysis/transfer_feasibility.csv")
    for row in data:
        with np.load(evidence / "inputs/2021_reserved" / f"elapsed{int(row['day']):02d}.npz") as stored:
            values = stored["values"]
        arrival, demand, slo = values[:, [0, 3, 4]].T
        intrinsic = demand > 1.5 * slo
        assert intrinsic.sum() == int(row["intrinsic_misses"])
        close(intrinsic.mean(), row["intrinsic_fraction"])
        close(arrival[-1], row["span_s"])
        close(arrival[-1] / float(row["raw_span_s"]), row["time_multiplier"])
        for partitions in (30, 180, 720):
            labels = np.minimum(np.floor(arrival / arrival[-1] * partitions).astype(int), partitions - 1)
            total = int(intrinsic.sum())
            for label in range(partitions):
                keep = (labels == label) & ~intrinsic
                if not keep.any():
                    continue
                deadlines = arrival[keep] + slo[keep]
                costs = demand[keep]
                cutoffs = np.unique(deadlines if len(costs) <= 32 else
                                    np.quantile(deadlines, [.01, .05, .1, .25, .5, .75, .9, .95, .99, 1.]))
                min_time = arrival[keep].min()
                worst = 0
                for cutoff in cutoffs:
                    selected = costs[deadlines <= cutoff]
                    work = math.fsum(selected)
                    available = 10.4 * (cutoff - min_time)
                    excess = work - available
                    tolerance = 1e-9 * max(work, available, 1.)
                    if excess > tolerance:
                        possible = np.searchsorted(np.cumsum(-np.sort(-selected)), excess - tolerance) + 1
                        worst = max(worst, int(possible))
                total += worst
            assert total == int(row[f"bound_bins_{partitions}"])
        floor = max(int(row[f"bound_bins_{k}"]) for k in (30, 180, 720))
        assert floor == int(row["pooled_bound_misses"]) and floor * 200 <= int(row["requests"])
        close(floor / len(values), row["pooled_bound_fraction"])
    return data


def near(rows):
    grouped = aggregate(rows, ["day", "arm"])
    output = {}
    for (day, arm) in grouped:
        vals = [r for r in rows if r["day"] == day and r["arm"] == arm]
        output[day, arm] = sum(200*r["violations"] > r["n"] and 2000*r["violations"] <= 11*r["n"] for r in vals)
    for arm in BASE_ARMS:
        output["all", arm] = sum(v for (d, a), v in output.items() if d != "all" and a == arm)
    return output


def latex_rows(items):
    return "\n".join(" & ".join(map(str, row)) + r" \\" for row in items)


def table(columns, headers, rows, title, label):
    return "\n".join([
        r"\begin{table}[!ht]", r"\centering\small", rf"\caption{{{title}}}\label{{{label}}}",
        rf"\begin{{tabular}}{{@{{}}{columns}@{{}}}}", r"\toprule",
        " & ".join(headers) + r" \\", r"\midrule", latex_rows(rows), r"\bottomrule",
        r"\end{tabular}", r"\end{table}", "",
    ])


def write_exhibits(destination, runs, agg, paired, bands, floor):
    src = destination / "source"
    audit_rows = []
    macros = {}

    def val(key, number, display, selector):
        assert key not in macros
        macros[key] = str(display)
        audit_rows.append(dict(tex_macro=key, printed_value=display, unrounded=number, origin=selector))

    def summary(phase, bank, day, arm):
        return agg[phase, bank][day, arm]

    def effect(phase, bank, day, contrast):
        return paired[phase, bank][0][day, contrast]

    for name, (phase, bank, arm) in {
        "OldOracleOn": ("B", "historical", "oracle_on"), "OldEstimatedOn": ("B", "historical", "estimated_on"),
        "NewOracleOn": ("B", "new", "oracle_on"), "NewEstimatedOn": ("B", "new", "estimated_on"),
        "NewEstimatedOff": ("B", "new", "estimated_off"),
        "NewBothOracle": ("D", "new", "OO"), "NewBothEstimated": ("D", "new", "EE"),
    }.items():
        a = summary(phase, bank, "all", arm)
        val(name, a["passed"], f"{a['passed']}/{a['n']}", f"{phase}/{bank}/all/{arm}: attainment")

    for prefix, bank in (("Old", "historical"), ("New", "new")):
        a = effect("B", bank, "all", "guard_estimated_training")
        val(prefix+"GuardCross", a["fail_to_pass"], f"{a['fail_to_pass']}/{a['pairs']}", f"B/{bank}: estimated guard crossings")
        for name, arm in (("Oracle", "oracle_off"), ("Estimated", "estimated_off")):
            n = summary("B", bank, "all", arm)["n"]
            count = bands[bank]["all", arm]
            val(prefix+name+"Near", count, f"{count}/{n}", f"B/{bank}/{arm}: rates in (0.50%,0.55%]")

    contrasts = {
        "NewTrainingMiss": ("B", "new", "all", "fine_tuning_guard_on", "miss_delta_pp", 3, 1),
        "NewTrainingP": ("B", "new", "all", "fine_tuning_guard_on", "p95_delta_ms", 2, 1),
        "NewTrainingAttain": ("B", "new", "all", "fine_tuning_guard_on", "attainment_pp", 2, 1),
        "NewOracleCross": ("B", "new", "all", "guard_oracle_training", "fail_to_pass", 0, 1),
        "NewGuardInteract": ("B", "new", "all", "training_by_guard_interaction", "attainment_pp", 2, 1),
        "NewBothShift": ("D", "new", "all", "both shift, guarded", "miss_delta_pp", 3, 1),
        "DayThirteenBothShift": ("D", "new", "d13", "both shift, guarded", "miss_delta_pp", 3, 1),
        "DayThirteenPolicyShift": ("D", "new", "d13", "base shift, no guard", "miss_delta_pp", 3, 1),
        "NewObsInteraction": ("D", "new", "all", "observation interaction", "miss_delta_pp", 3, 1),
        "DayThirteenInteraction": ("D", "new", "d13", "observation interaction", "miss_delta_pp", 3, 1),
        "ReservedDayEightMiss": ("transfer", "new", "8", "guard_estimated_training", "miss_delta_pp", 2, 1),
        "ReservedDayEightP": ("transfer", "new", "8", "guard_estimated_training", "p95_delta_ms", 1, .001),
        "ReservedDaySevenP": ("transfer", "new", "7", "guard_estimated_training", "p95_delta_ms", 1, .001),
        "ReservedAllP": ("transfer", "new", "all", "guard_estimated_training", "p95_delta_ms", 1, .001),
    }
    for name, (phase, bank, day, label, metric, precision, unit) in contrasts.items():
        v = effect(phase, bank, day, label)[metric] * unit
        val(name, v, f"{v:+.{precision}f}" if precision else f"{v:.0f}", f"{phase}/{bank}/{day}/{label}/{metric}")
    for mode in ("off", "on"):
        values = summary("transfer", "new", "all", f"estimated_{mode}")
        for suffix, metric, scale in (("Miss", "miss_pct", 1), ("P", "p95_ms", .001)):
            number = values[metric]*scale
            val("Reserved" + mode.title() + suffix, number, f"{number:,.1f}" if suffix == "P" else f"{number:.2f}",
                f"transfer/new/all/estimated_{mode}/{metric}")
    for info in floor:
        v = 100 * float(info["pooled_bound_fraction"])
        name = "Reserved" + {"7": "Seven", "8": "Eight"}[info["day"]] + "Floor"
        val(name, v, f"{v:.3f}", f"transfer/new/{info['day']}/pooled floor")
    val("TemporalExecuted", 1090, "1,090", "600 B + 450 D + 40 transfer archived executions")
    val("TemporalOverlap", 150, "150", "Identical effective configurations B and D")
    val("TemporalUnique", 940, "940", "Set union of workload/model/policy-view/guard-view identities")

    def serial(a, unit=1, digits=0):
        return rf"{a['miss_pct']:.3f}\,/\,{a['p95_ms']*unit:.{digits}f}\,/\,{a['passed']}/{a['n']}"

    def small_b_table():
        return table(
            "llrr", ["Day", "Fine-tuning", "Guard off", "Guard on"],
            [(day if mode == "oracle" else "", mode.title(), *(serial(summary("B", "new", day, f"{mode}_{g}")) for g in ("off", "on")))
             for day in (*DAYS, "all") for mode in ("oracle", "estimated")],
            r"Later Azure 2019 days under estimated evaluation. Each cell gives miss rate (\%), mean workload P95 (ms), and budget passes/total.",
            "tab:temporal-b",
        )

    def small_d_table():
        contrasts_d = ("base shift, no guard", "guard shift, oracle policy", "guard shift, estimated policy",
                       "both shift, guarded", "observation interaction")
        display = ("Policy only, guard off", "Guard only, policy O", "Guard only, policy E", "Both changed", "Interaction")
        return table("lrrrr", ["Contrast", "d12", "d13", "d14", "All"],
                     [(name, *(f"{effect('D', 'new', day, query)['miss_delta_pp']:+.3f}" for day in (*DAYS, "all")))
                      for name, query in zip(display, contrasts_d, strict=True)],
                     "Crossed changes in miss rate on later days (percentage points). Each day has 25 pairs per arm. O and E denote oracle and estimated state. The interaction contrasts the two guard shifts.",
                     "tab:temporal-d")

    def small_t_table():
        return table(
            "llrr", ["Day", "Fine-tuning", "Guard off", "Guard on"],
            [(day if mode == "oracle" else "", mode.title(),
              *(serial(summary("transfer", "new", day, f"{mode}_{g}"), .001, 1) for g in ("off", "on")))
             for day in ("7", "8") for mode in ("oracle", "estimated")],
            r"Azure 2021 reserved windows under the existing transfer protocol. Each cell gives miss rate (\%), mean workload P95 (s), and budget passes/total. Each arm uses estimated state.",
            "tab:reserved",
        )

    src.joinpath("temporal_values.tex").write_text("\n".join(rf"\newcommand{{\{k}}}{{{v}}}" for k, v in macros.items()) + "\n", encoding="utf-8")
    for key, content in (("b", small_b_table()), ("d", small_d_table()), ("transfer", small_t_table())):
        src.joinpath(f"temporal_table_{key}.tex").write_text(content, encoding="utf-8")
    write_csv(destination / "numerical_macro_audit.csv", audit_rows)

    # Keep the historical body tables intact. These compact supplement tables enumerate all new day-seed groups.
    section = []
    for phase in ("B", "D", "transfer"):
        arms = CROSS_ARMS if phase == "D" else BASE_ARMS
        seeds = aggregate(runs[phase, "new"], ["day", "seed", "arm"])
        unit = .001 if phase == "transfer" else 1.
        days = ("7", "8") if phase == "transfer" else DAYS
        for day in days:
            rows = []
            for seed in SEEDS:
                cells = [serial(seeds[day, seed, arm], unit, 1 if phase == "transfer" else 0) for arm in arms]
                if phase == "D":
                    cells = [r"\shortstack[r]{" + left + r"\\" + right + "}"
                             for left, right in (cell.rsplit(r"\,/\,", 1) for cell in cells)]
                rows.append((seed, *cells))
            note = "Each cell gives miss rate (\\%), mean workload P95 (" + ("s" if phase == "transfer" else "ms") + "), and passes/total."
            heading = "Crossed replay" if phase == "D" else "Fine-tuning and guard"
            section.append(table("l"+"r"*len(arms), ["Seed", *(tuple(a.replace("-", "$-$") for a in arms) if phase == "D" else PRETTY)],
                                 rows, f"{heading} on {'elapsed day ' if phase == 'transfer' else ''}{day}. {note}",
                                 f"tab:full-{phase.lower()}-{day}"))
    src.joinpath("temporal_seed_tables.tex").write_text("\n".join(section), encoding="utf-8")

    intervals_tables, auxiliary = [], []
    name_b = ("Guard with oracle FT", "Guard with estimated FT", "FT difference, guard off",
              "FT difference, guard on", "Training by guard interaction")
    name_d = ("Policy O to E, guard off", "Policy O to E, guard O", "Policy O to E, guard E",
              "Guard O to E, policy O", "Guard O to E, policy E", "Both O to E", "Observation interaction")
    for phase, labels in (("B", name_b), ("D", name_d)):
        point, intervals = paired[phase, "new"]
        rows = []
        for key, name in zip(intervals, labels, strict=True):
            cells = {m: f"{point['all', key][m]:+.3f} [{lo:+.3f}, {hi:+.3f}]"
                     for m, (lo, hi) in intervals[key].items()}
            rows.append((name, *(cells[m] for m in ("attainment_pp", "miss_delta_pp", "p95_delta_ms"))))
            auxiliary.append((phase, name, cells["mean_delta_ms"], cells["p99_delta_ms"]))
        content = table("lrrr", ("Contrast", "Pass (pp)", "Miss (pp)", "P95 (ms)"), rows,
                        ("Later-day fine-tuning" if phase == "B" else "Later-day crossed")+
                        " paired contrasts. Entries give point estimates and 95\\% descriptive grouped intervals.",
                        f"tab:temporal-ci-{phase.lower()}")
        intervals_tables.append(content.replace(r"\centering\small", r"\centering\footnotesize\setlength{\tabcolsep}{4pt}"))
    intervals_tables.append(table("llrr", ("Design", "Contrast", "Mean (ms)", "P99 (ms)"), auxiliary,
                                  "Paired mean-latency and P99 contrasts for the later 2019 matrices with 95\\% descriptive grouped intervals.",
                                  "tab:temporal-ci-aux"))
    src.joinpath("temporal_ci_tables.tex").write_text("\n".join(intervals_tables), encoding="utf-8")
    transition_tables = []
    for phase, names in (("B", name_b[:-1]), ("D", name_d[:-1]), ("transfer", name_b[:-1])):
        point, _ = paired[phase, "new"]
        keys = [k[1] for k in point if k[0] == "all" and "interaction" not in k[1]]
        row = []
        for (day, key), result in point.items():
            if key not in keys:
                continue
            row.append((day, names[keys.index(key)], result["fail_to_pass"], result["pass_to_fail"],
                        result["both_pass"], result["both_fail"]))
        transition_tables.append(table("llrrrr", ["Day", "Contrast", "Fail to pass", "Pass to fail", "Both pass", "Both fail"],
                                       row, f"Paired budget transitions for {'the reserved transfer windows' if phase == 'transfer' else 'the later-day '+phase+' matrix'}.",
                                       f"tab:temporal-transitions-{phase.lower()}"))
    src.joinpath("temporal_transition_tables.tex").write_text("\n".join(transition_tables), encoding="utf-8")
    table_floor = table(
        "rrrrrrr", ("Day", "Start minute", "Requests", "Arrival scale", "Intrinsic (\\%)", "Pooled (\\%)", "Passes"),
        [(r["day"], r["window_start_minute"], r["requests"], f"{float(r['time_multiplier']):.5f}",
          f"{100*float(r['intrinsic_fraction']):.4f}", f"{100*float(r['pooled_bound_fraction']):.4f}", "0/20") for r in floor],
        r"Necessary feasibility diagnostics on the reserved windows. The pooled floor takes the maximum over the predeclared 30, 180, and 720 bin partitions. Both floors leave feasibility open at the 0.5\% budget.",
        "tab:temporal-floor",
    )
    src.joinpath("temporal_floor_table.tex").write_text(table_floor, encoding="utf-8")
    return macros


def figure(destination, aggregate, paired):
    colors = ["#435971", "#158178"]
    orig, new = "#8198ae", "#126b91"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.1, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(3.48, 5.55))
    grid = fig.add_gridspec(3, 2, hspace=.79, wspace=.45, height_ratios=[1.15, 1.22, 1.04],
                          left=.17, right=.96, bottom=.075, top=.94)
    a, b, c, d = (fig.add_subplot(grid[i, j] if i == 2 else grid[i, :]) for i, j in ((0, 0), (1, 0), (2, 0), (2, 1)))
    bars = []
    for i, mode in enumerate(("oracle_on", "estimated_on")):
        values = []
        for bank in ("historical", "new"):
            rec = aggregate["B", bank]["all", mode]
            values.append(100 * rec["passed"] / rec["n"])
            bars.append(dict(panel="a", group=bank, variable=mode, value=values[-1], unit="percent"))
        series = a.bar(np.arange(2)+(i-.5)*.32, values, .30, label=("Oracle fine-tuning", "Estimated fine-tuning")[i], color=colors[i])
        a.bar_label(series, [f"{v:.1f}%" for v in values], padding=2, fontsize=6.8)
    a.set(xticks=[0, 1], xticklabels=["Original d08–d09", "Later d12–d14"], ylim=(0, 112), ylabel="Attainment (%)")
    a.set_title("(a) Guarded budget attainment", loc="left", pad=19, fontweight="semibold")
    a.legend(loc="lower left", bbox_to_anchor=(-.07, 1.), ncols=2, frameon=False, fontsize=6.5, columnspacing=.9)
    categories = ("d08–d09", "d12", "d13", "d14")
    shifts = [paired["D", "historical"][0]["all", "both shift, guarded"]["miss_delta_pp"]]
    shifts += [paired["D", "new"][0][day, "both shift, guarded"]["miss_delta_pp"] for day in DAYS]
    series = b.barh(range(len(categories)), shifts, .64, color=[orig, new, new, new])
    b.axvline(0, color="#4d5860", linewidth=.6)
    b.bar_label(series, [f"{v:+.3f}" for v in shifts], padding=2, fontsize=6.5)
    b.set(yticks=range(4), yticklabels=categories, xlim=(-.235, .337), xlabel="Change in misses (percentage points)")
    b.invert_yaxis()
    b.set_title("(b) Both queue views change from oracle to estimated", loc="left", pad=5, fontweight="semibold", fontsize=6.9)
    bars += [dict(panel="b", group=day, variable="EE_minus_OO", value=v, unit="percentage points")
             for day, v in zip(categories, shifts, strict=True)]
    categories_t = ("9–13", "7", "8")
    for axis, metric, unit, label, lim, ticks in ((c, "miss_delta_pp", 1, "Miss change (pp)", (0, 26), [0, 10, 20]),
                                                (d, "p95_delta_ms", .001, "P95 change (s)", (-2400, 1000), [-2000, 0, 1000])):
        values = [paired["transfer", bank][0][day, "guard_estimated_training"][metric] * unit
                  for bank, day in (("historical", "all"), ("new", "7"), ("new", "8"))]
        axis.barh(range(3), values, .62, color=[orig, new, new])
        axis.axvline(0, color="#4d5860", linewidth=.6)
        axis.set(ylim=(-.52, 2.52), xlabel=label, xticks=ticks, xlim=lim,
                 yticks=range(3), yticklabels=categories_t if axis is c else [])
        axis.invert_yaxis()
        bars += [dict(panel="c", group=day, variable=metric, value=value, unit="seconds" if unit == .001 else "percentage points")
                 for day, value in zip(categories_t, values, strict=True)]
    c.set_title("(c) Guard effect after estimated fine-tuning in transfer", loc="left", pad=5, fontweight="semibold", fontsize=6.9)
    for axis in (a, b, c, d):
        grid_axis = "y" if axis is a else "x"
        axis.grid(axis=grid_axis, color="#e3e7ea", linewidth=.5)
        axis.set_axisbelow(True)
        axis.tick_params(axis="both", length=2.3, labelsize=6.6)
    fig.savefig(destination / "source/figures/Fig6.pdf", bbox_inches="tight", pad_inches=.04,
                metadata={"Creator": "Matplotlib", "CreationDate": None})
    plt.close(fig)
    write_csv(destination / "temporal_figure_data.csv", bars)


def main(evidence, destination, historical, archived_only=False):
    (destination / "source/figures").mkdir(parents=True, exist_ok=True)
    protocol = load(evidence / "protocol.json")
    new = {phase: read_runs(evidence / phase / "replays.csv", phase) for phase in ("B", "D", "transfer")}
    overlap = verify_design(evidence, protocol, new, archived_only)
    write_csv(destination / "overlap_audit.csv", overlap)
    references = {phase: historical / f"{phase}.csv" if historical else Path(rec["path"])
                  for phase, rec in protocol["historical_refs"].items()}
    assert all(portable(path, protocol["historical_refs"][phase]) for phase, path in references.items())
    prior = {phase: read_runs(path, phase) for phase, path in references.items()}
    paired, aggregates, runs = {}, {}, {}
    checked = 0
    for phase in new:
        for bank, rows in (("new", new[phase]), ("historical", prior[phase])):
            name = phase if bank == "new" else f"{phase}_historical"
            checked += audit_aggregates(evidence, rows, name)
            paired[phase, bank] = effects(evidence, rows, phase, name, protocol["analysis"])
            aggregates[phase, bank] = aggregate(rows, ["day", "arm"]) | {
                ("all", k[0]): v for k, v in aggregate(rows, ["arm"]).items()
            }
            runs[phase, bank] = rows
    floor = audit_floor(evidence)
    bands = {bank: near(runs["B", bank]) for bank in ("new", "historical")}
    for bank, name in (("new", "B"), ("historical", "B_historical")):
        for row in records(evidence / f"analysis/{name}_near_budget.csv"):
            assert bands[bank][row["day"], row["arm"]] == int(row["near_above"])
    assert all(paired["B", "new"][0][day, "fine_tuning_guard_on"]["miss_delta_pp"] < 0 <
               paired["B", "new"][0][day, "fine_tuning_guard_on"]["p95_delta_ms"] for day in DAYS)
    assert all(aggregates["B", "new"][day, arm]["passed"] == 0 for day in ("d12", "d13") for arm in BASE_ARMS)
    assert paired["D", "new"][0]["d13", "observation interaction"]["miss_delta_pp"] < 0
    t = paired["transfer", "new"][0]
    assert t["8", "guard_estimated_training"]["higher_p95"] == t["8", "guard_estimated_training"]["higher_miss"] == 5
    assert sum(r["met_budget"] for r in new["transfer"]) == 0
    macros = write_exhibits(destination, runs, aggregates, paired, bands, floor)
    figure(destination, aggregates, paired)
    status = dict(protocol=protocol["protocol_sha256"], executions=1090, distinct_configurations=940,
                  repeated_B_D_configurations=150, published_aggregate_cells_checked=checked,
                  temporal_prose_macros=len(macros), passed=True)
    (destination / "numerical_audit_status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evidence", type=Path, default=EVIDENCE)
    p.add_argument("--draft", type=Path, default=DRAFT)
    p.add_argument("--historical", type=Path)
    p.add_argument("--archived-only", action="store_true",
                   help="Verify recorded replay and workload identities without every trace file.")
    a = p.parse_args()
    main(a.evidence, a.draft, a.historical, a.archived_only)
