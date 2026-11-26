"""Calculate paired training comparisons with shared workload resampling."""
from pathlib import Path

import pandas as pd
from guard_statistics import grouped_differences


def main():
    root = Path(__file__).resolve().parents[1]
    results = pd.read_csv(root / "outputs/training_evidence/matched_training/confirmation_paired.csv")
    results["met_budget"] = results.violation_rate.le(.005)
    keys = ["seed", "day", "reconstruction_seed", "trace_sha256"]
    adaptive = results.loc[results.method.eq("adaptive")]
    fixed = results.loc[results.method.eq("fixed")]
    pairs = adaptive.merge(fixed, on=keys, suffixes=("_candidate", "_comparator"), validate="one_to_one")
    if len(pairs) != len(adaptive) or len(pairs) != len(fixed):
        raise ValueError("Fine-tuning records do not share the same workload and seed keys.")
    contrasts = pd.DataFrame(grouped_differences(pairs))
    target = root / "regenerated/training_reference_intervals.csv"
    target.parent.mkdir(exist_ok=True)
    contrasts.to_csv(target, index=False)
    print(contrasts.to_string(index=False))


if __name__ == "__main__":
    main()
