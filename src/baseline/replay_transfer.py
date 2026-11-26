"""Chronological Azure 2021 transfer replication with a frozen 2019 checkpoint."""

from pathlib import Path
import hashlib
import json

import replay_io as base
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from tqdm import tqdm

from routing_simulator import replay

EXT = base.OUT / "external"
SOURCE = EXT / "AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt"
ARCHIVE = SOURCE.with_suffix(".rar")
DAY = 86400


def prepare(source):
    raw = pd.read_csv(source)
    required = ["app", "func", "end_timestamp", "duration"]
    if not set(required) <= set(raw.columns):
        raise ValueError(f"Unexpected public schema: {raw.columns.tolist()}")
    norm = raw[required].copy()
    valid = np.isfinite(norm.end_timestamp) & np.isfinite(norm.duration)
    valid &= norm.duration.gt(0) & norm.end_timestamp.ge(norm.duration)
    valid &= norm.app.notna() & norm.func.notna()
    norm = norm.loc[valid].copy()
    norm["start"] = norm.end_timestamp - norm.duration
    norm["day"] = (norm.start // DAY).astype(np.int64)
    norm["id"] = norm.app.astype(str) + "/" + norm.func.astype(str)
    first = norm[norm.day.between(0, 6) & (norm.end_timestamp < 7*DAY)]
    ranking = first.groupby("id").agg(n=("duration", "size"), days=("day", "nunique"))
    ranking = ranking[(ranking.n >= 200) & (ranking.days >= 3)].reset_index()
    cohort = ranking.sort_values(["n", "id"], ascending=[False, True]).id.head(30)
    if len(cohort) < 5:
        raise ValueError("Too few functions satisfy the prespecified training-only rule")
    train = first[first.id.isin(cohort)]
    names = {key: i for i, key in enumerate(sorted(cohort))}
    blocks = [np.sort(train.loc[train.id == key, "duration"].to_numpy()) for key in names]
    bank = np.concatenate(blocks)
    offsets = np.r_[0, np.cumsum([len(a) for a in blocks])]
    prefix = np.r_[0., np.cumsum(bank)]
    means = np.array([a.mean() for a in blocks])
    profiles = np.array([np.quantile(a, [.5, .95, .99]) for a in blocks])
    return norm, names, bank, offsets, prefix, means, profiles, len(raw)-len(norm)


def day_workload(frame, day, names, profiles):
    pool = frame[(frame.day == day) & frame.id.isin(names)].sort_values("start", kind="stable")
    minutes = ((pool.start.to_numpy() - day*DAY) // 60).astype(int)
    counts = np.bincount(minutes, minlength=1440)
    idx = np.convolve(counts, np.ones(180, np.int64), mode="valid").argmax()
    keep = pool.iloc[np.flatnonzero((minutes >= idx) & (minutes < idx+180))]
    if len(keep) < 100:
        raise ValueError(f"Insufficient requests on predeclared day {day}")
    ids = keep.id.map(names).to_numpy(np.int64)
    a = keep.start.to_numpy()
    demand = keep.duration.to_numpy()
    p50,p95,p99 = profiles[ids].T
    target_span = demand.sum() / (np.dot(base.SPEED, base.WORKERS) * .6)
    arr = (a-a[0]) * target_span / (a[-1]-a[0])
    return np.ascontiguousarray(np.column_stack([arr,p50,p99,demand,np.maximum(.1,2*p99)])), ids, {
        "p50": p50.copy(), "p95": p95.copy(), "p99": p99.copy()}, int(idx)


def main():
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    models, _, _ = base.audited_inputs()
    fixed = {seed: models["fixed", seed][1] for seed in base.SEEDS}
    protocol = {
        "source": "Microsoft Azure Functions Invocation Trace 2021; attribution Zhang et al., SOSP 2021",
        "archive_sha256": base.digest(ARCHIVE), "member_sha256": base.digest(SOURCE),
        "code": {n: base.digest(base.ROOT/"work"/n) for n in ("replay_transfer.py", "replay_io.py", "routing_simulator.py")},
        "unit": "one real source elapsed-day window per draw; durations as service proxy, start=end-duration, timestamps modified by publisher",
        "split": {"profile_fit": "elapsed days 0-6, invocations ending before day 7",
                  "reserved_unused": [7, 8], "held_out": [9, 10, 11, 12, 13]},
        "cohort": "top 30 (app,func) by valid training count; minimum 200 requests over at least 3 training days; lexicographic tie",
        "test_replay": "earliest busiest 180-minute window per day in fixed cohort; preserve order and intervals up to global rescale to 60% offered load on three modeled servers",
        "checkpoints": fixed,
        "models": "No retraining; fixed-DDQN weights and feature scales frozen from 2019. Profiles, guard CDF and deadlines built from 2021 training prefix. Cross-corpus transfer, not an unchanged full-stack checkpoint test.",
        "deadline": "twice training P99, floored at 100 ms", "budget": .005,
        "guard_thresholds": [.005,.005], "branches": ["oracle", "estimated"],
        "no_test_tuning": True,
    }
    identity = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    lock = EXT/"protocol.json"
    if lock.exists() and base.load_json(lock)["identity"] != identity:
        raise ValueError("External protocol differs from the locked run")
    base.publish_json(lock, {"identity": identity, **protocol})
    base.setup()
    frame, names, bank, offsets, prefix, means, profiles, excluded = prepare(SOURCE)
    dummy, models = base.CONTEXT[6:8]
    arms = [(label, -1, method, guard, pred, dummy) for label, method, guard, pred in base.METHODS]
    for seed in base.SEEDS:
        arms.extend([("Fixed-DDQN",seed,0,False,"p50",models["fixed",seed]),
                     ("Fixed-DDQN+Guard",seed,0,True,"p50",models["fixed",seed])])
    rows = []
    with threadpool_limits(limits=1):
        for day in tqdm(protocol["split"]["held_out"], desc="Azure 2021 held-out days"):
            save = EXT/f"heldout_{day:02d}.json"
            if save.exists():
                cached = base.load_json(save)
                if cached["identity"] != identity:
                    raise ValueError("External checkpoint identity mismatch")
                rows.extend(cached["rows"])
                continue
            x, ids, preds, start_minute = day_workload(frame, day, names, profiles)
            sha = hashlib.sha256(x.tobytes()+ids.tobytes()).hexdigest()
            current = []
            for mode in ("oracle", "estimated"):
                for label, seed, method, guard, pred, (scales, weights, model_sha) in arms:
                    out = replay(x, ids, preds[pred], base.SPEED, base.WORKERS, scales, weights,
                                 method, guard, mode == "estimated", 1., bank, offsets, prefix, means)
                    stats, _ = base.results(x,out)
                    current.append(dict(day=day, window_start_minute=start_minute, seed=seed,
                                        method=label, model_sha256=model_sha, trace_sha256=sha,
                                        observation=mode, **stats))
            base.publish_json(save, {"identity":identity,"rows":current})
            rows.extend(current)
    assert len(rows) == 5*2*len(arms)
    pd.DataFrame(rows).to_csv(EXT/"paired_2021.csv",index=False)
    base.publish_json(EXT/"status.json",{"status":"VERIFIED_2021_HELDOUT_TRANSFER","identity":identity,
                      "rows":len(rows),"cohort_size":len(names),"excluded_source_records":excluded,
                      "paired_sha256":base.digest(EXT/"paired_2021.csv")})


if __name__ == "__main__":
    main()
