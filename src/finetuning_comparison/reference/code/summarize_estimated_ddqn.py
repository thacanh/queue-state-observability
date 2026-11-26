"""Combine seed-wise validation replays and report paired observation effects."""

import argparse
import json
from pathlib import Path

import pandas as pd

from estimated_ddqn import save_metadata, sha
from evaluate_estimated_ddqn import write_csv
from guard_statistics import grouped_differences


KEYS = ["day", "reconstruction_seed", "seed"]


def paired(candidate, reference):
    result = candidate.merge(reference, on=KEYS, suffixes=("_candidate", "_comparator"),
                             validate="one_to_one")
    if len(candidate) != len(reference) or len(candidate) != len(result):
        raise ValueError("Missing paired replay")
    if not result.trace_sha256_candidate.equals(result.trace_sha256_comparator):
        raise ValueError("Paired trace hash mismatch")
    return result


def describe(data, group):
    return data.groupby(group, as_index=False).agg(
        pairs=("met_budget", "size"), passes=("met_budget", "sum"),
        miss_pct=("violation_rate", lambda x: 100*x.mean()),
        mean_ms=("mean_ms", "mean"), p95_ms=("p95_ms", "mean"), p99_ms=("p99_ms", "mean"),
        brier=("brier", "mean"), override_pct=("override_rate", lambda x: 100*x.mean()))


def summarize(root, output, seeds):
    frames, manifests = [], {}
    for seed in seeds:
        directory = root / f"validation_seed{seed}"
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        for name, digest in manifest["outputs"].items():
            if sha(directory / name) != digest:
                raise ValueError(f"Output verification failed for seed {seed}: {name}")
        data = pd.read_csv(directory / "paired.csv")
        if len(data) != 236 or set(data.seed) != {seed}:
            raise ValueError(f"Unexpected evaluation matrix for seed {seed}")
        frames.append(data)
        manifests[seed] = manifest
    for key in ("validation_bank_sha256", "baseline_sha256", "evaluation_source_sha256"):
        if len({m[key] for m in manifests.values()}) != 1:
            raise ValueError(f"Different replay protocol across seeds: {key}")
    training = [m["training_manifests"][str(s)] for s, m in manifests.items()]
    for key in ("source", "runtime"):
        if any(m[key] != training[0][key] for m in training):
            raise ValueError(f"Different training protocol across seeds: {key}")
    full = pd.concat(frames, ignore_index=True)
    method = ["policy_training", "guard"]
    allmetrics = {"summary": describe(full, method),
                  "by_seed": describe(full, method + ["seed"]),
                  "by_day_seed": describe(full, method + ["day", "seed"])}
    groups = {key: df for key, df in full.groupby(method)}
    intervals, transitions = [], []
    for label, candidate, comparator in (
        ("fine-tuning without guard", ("estimated_fine_tuning", False), ("oracle", False)),
        ("fine-tuning with guard", ("estimated_fine_tuning", True), ("oracle", True)),
        ("guard after estimated fine-tuning", ("estimated_fine_tuning", True), ("estimated_fine_tuning", False)),
        ("guard after oracle fine-tuning", ("oracle", True), ("oracle", False)),
    ):
        diff = paired(groups[candidate], groups[comparator])
        intervals.extend({"contrast": label, **value} for value in grouped_differences(diff))
        for (day, seed), p in diff.groupby(["day", "seed"]):
            a, b = p.met_budget_candidate, p.met_budget_comparator
            transitions.append({
                "contrast": label, "day": day, "seed": seed, "pairs": len(p),
                "gained": int((a & ~b).sum()), "lost": int((~a & b).sum()),
                "candidate_miss_pct": 100*p.violation_rate_candidate.mean(),
                "reference_miss_pct": 100*p.violation_rate_comparator.mean(),
                "mean_p95_change_ms": (p.p95_ms_candidate-p.p95_ms_comparator).mean(),
            })
    output.mkdir(parents=True, exist_ok=True)
    for name, value in {**allmetrics, "paired": full, "paired_intervals": pd.DataFrame(intervals),
                        "paired_transitions": pd.DataFrame(transitions)}.items():
        write_csv(value, output / f"{name}.csv")
    save_metadata(output / "manifest.json", {
        "seeds": seeds,
        "scope": "Matched development follow-up on 59 reconstructions of two inspected days, d08 and d09",
        "validation_bank_sha256": manifests[seeds[0]]["validation_bank_sha256"],
        "replay_manifests": {str(s): sha(root / f"validation_seed{s}/manifest.json") for s in seeds},
        "outputs": {p.name: sha(p) for p in output.glob("*.csv")},
    })
    print(allmetrics["summary"].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    args = parser.parse_args()
    summarize(args.root, args.output, args.seeds)
