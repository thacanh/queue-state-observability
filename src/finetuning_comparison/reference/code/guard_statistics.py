"""Verify and summarize guard_evaluation results, with paired estimates conditional on source days."""

from pathlib import Path
import json

import replay_io as base
import numpy as np
import pandas as pd

OUT = base.OUT
METRICS = ["mean_ms", "p95_ms", "p99_ms", "violation_rate", "override_rate",
           "wait_MAE_ms", "wait_bias_ms", "brier", "risk_mean", "risk_vs_oracle_MAE",
           "oracle_state_brier"]


def summary(data, keys):
    sizes = data.groupby(keys, dropna=False).agg(passes=("met_budget", "sum"), rows=("met_budget", "size"),
                                               requests=("n","sum"), misses=("violations","sum"))
    combined = sizes.join(data.groupby(keys, dropna=False)[METRICS].mean()).reset_index()
    combined["attainment"] = combined.passes / combined.rows
    combined["request_weighted_violation"] = combined.misses / combined.requests
    return combined


def frontier(data, x="p95_ms", y="violation_rate"):
    frame = data.copy()
    frame["on_frontier"] = [
        not (((frame[x] <= row[x]) & (frame[y] <= row[y])) &
             ((frame[x] < row[x]) | (frame[y] < row[y]))).any()
        for _, row in frame.iterrows()
    ]
    return frame


def pair(frame, observation, comparator, candidate="Fixed-DDQN+Guard"):
    frame = frame[frame.observation.eq(observation)]
    keys = ["day", "reconstruction_seed"]
    a = frame[frame.method.eq(candidate)]
    b = frame[frame.method.eq(comparator)]
    if b.seed.ne(-1).any():
        keys = keys+["seed"]
    pairs = a.merge(b, on=keys, suffixes=("_candidate", "_comparator"), validate="many_to_one")
    assert pairs.trace_sha256_candidate.equals(pairs.trace_sha256_comparator)
    assert len(pairs) == len(a)
    return pairs


def grouped_differences(pairs):
    """Resample crossed seeds and days; reconstruct jointly across policies within day."""
    rng = np.random.default_rng(20261124)
    sample_metrics = ["met_budget", "mean_ms", "p95_ms", "p99_ms", "violation_rate"]
    ids = [c for c in ("seed", "seed_candidate") if c in pairs][0]
    seed_values = sorted(pairs[ids].unique())
    days = sorted(pairs.day.unique())
    grouped = []
    for day in days:
        group = pairs[pairs.day.eq(day)].copy()
        draw_ids = sorted(group.reconstruction_seed.unique())
        index = pd.MultiIndex.from_product([seed_values,draw_ids], names=[ids,"reconstruction_seed"])
        group = group.set_index([ids, "reconstruction_seed"]).reindex(index)
        arr = np.stack([(group[m+"_candidate"].astype(float)-group[m+"_comparator"].astype(float)).to_numpy()
                        for m in sample_metrics], axis=1)
        grouped.append(arr.reshape(len(seed_values),len(draw_ids),len(sample_metrics)))
    boot = np.empty((4000,len(sample_metrics)))
    for i in range(len(boot)):
        seeds = rng.integers(len(seed_values), size=len(seed_values))
        selected = []
        for d in rng.integers(len(days), size=len(days)):
            values = grouped[d]
            draws = rng.integers(values.shape[1],size=values.shape[1])
            selected.append(values[seeds[:,None],draws[None,:],:].reshape(-1,len(sample_metrics)))
        boot[i] = np.concatenate(selected).mean(axis=0)
    return [{"metric":m,
            "delta": float((pairs[m+"_candidate"].astype(float)-pairs[m+"_comparator"].astype(float)).mean()),
            "conditional_p025":float(np.quantile(boot[:,i], .025)),
            "conditional_p975":float(np.quantile(boot[:,i], .975))}
            for i,m in enumerate(sample_metrics)]


def calibrate():
    data = pd.read_json(OUT/"calibration_workloads.jsonl", lines=True)
    rows = []
    for (obs, method), group in data.groupby(["observation","method"]):
        sums = np.sum(np.stack(group.calibration.map(np.asarray)), axis=0)
        for low,high,(n, pred, truth, oracle) in zip(base.BINS[:-1], base.BINS[1:], sums):
            rows.append({"observation":obs, "method":method, "bin_low":low, "bin_high":min(1.,high),
                         "requests":int(n), "forecast_sum":pred, "misses":int(truth),
                         "oracle_forecast_sum":oracle,"predicted_mean":pred/n if n else np.nan,
                         "observed_rate":truth/n if n else np.nan})
    result = pd.DataFrame(rows)
    result.to_csv(OUT/"reliability.csv",index=False)
    totals = []
    for (obs, method), group in result.groupby(["observation","method"]):
        n = group.requests.sum()
        totals.append({"observation":obs,"method":method,"request_count":n,
                       "forecast_mean":group.forecast_sum.sum()/n, "observed_rate":group.misses.sum()/n,
                       "ece":(group.forecast_sum-group.misses).abs().sum()/n,
                       "oracle_conditional_forecast_mean":group.oracle_forecast_sum.sum()/n})
    pd.DataFrame(totals).to_csv(OUT/"calibration_summary.csv",index=False)


def main():
    state = base.load_json(OUT/"status.json")
    if base.digest(OUT/"paired.csv") != state["paired_sha256"]:
        raise ValueError("guard_evaluation result digest changed")
    frame = pd.read_csv(OUT/"paired.csv")
    assert len(frame)==state["rows"] and frame.reconstruction_seed.nunique()==59
    assert np.allclose(frame.violations/frame.n,frame.violation_rate)
    assert (frame.met_budget==(frame.violation_rate<=.005)).all()
    report = summary(frame,["observation","method"])
    pd.concat([frontier(g) for _,g in report.groupby("observation")]).to_csv(OUT/"summary.csv",index=False)
    summary(frame,["observation","day","method"]).to_csv(OUT/"by_day.csv",index=False)
    summary(frame,["observation","seed","method"]).to_csv(OUT/"by_seed.csv",index=False)
    diff, transition = [], []
    for observation in ("oracle","estimated"):
        for baseline in report.method.unique():
            if baseline == "Fixed-DDQN+Guard":
                continue
            pairs = pair(frame, observation, baseline)
            diff.extend(dict(observation=observation,baseline=baseline,**row)
                        for row in grouped_differences(pairs))
            for label,g in [("both",pairs),*pairs.groupby("day")]:
                a,b = g.met_budget_candidate,g.met_budget_comparator
                transition.append(dict(observation=observation,baseline=baseline,day=label,
                                       fail_to_pass=int((a&~b).sum()),pass_to_fail=int((~a&b).sum()),
                                       both_pass=int((a&b).sum()),both_fail=int((~a&~b).sum())))
    pd.DataFrame(diff).to_csv(OUT/"paired_conditional_intervals.csv",index=False)
    pd.DataFrame(transition).to_csv(OUT/"paired_transitions.csv",index=False)
    calibrate()
    for filename, src, keys in [
        ("cross_corpus_summary.csv",OUT/"external/paired_2021.csv",["observation","method"]),
        ("cross_corpus_by_day.csv",OUT/"external/paired_2021.csv",["observation","day","method"]),
        ("checkpoint_control_summary.csv",OUT/"diagnostics/checkpoint_controls.csv",["observation","method"]),
        ("robustness_summary.csv",OUT/"diagnostics/robustness.csv",["scenario","observation","method"]),
        ("threshold_bias_summary.csv",OUT/"diagnostics/sensitivity.csv",["observation","method","bias","trigger","reduction"]),
    ]:
        if src.exists():
            summary(pd.read_csv(src),keys).to_csv(OUT/filename,index=False)
    print(report[["observation","method","passes","rows","p95_ms","violation_rate","wait_MAE_ms"]].to_string(index=False))


if __name__ == "__main__":
    main()
