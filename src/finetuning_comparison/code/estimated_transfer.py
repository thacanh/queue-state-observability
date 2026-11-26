"""Run the locked Azure 2021 extension with both training observation modes."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def verify(path, expected):
    if digest(path) != expected:
        raise ValueError(f"Digest mismatch: {path.name}")


def atomic_json(path, content):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(content, indent=2) + "\n", encoding="utf8")
    tmp.replace(path)


def run(root, checkpoints, output):
    sys.path.insert(0, str(root.resolve() / "work"))
    import replay_io as base
    import replay_transfer as external
    from replay_crossed import queue_metrics, verify_old
    from crossed_router import crossed_kernel
    import torch
    from threadpoolctl import threadpool_limits

    seeds, days = list(range(42, 47)), list(range(9, 14))
    original = external.EXT
    protocol = read(original / "protocol.json")
    old_status = read(original / "status.json")
    if protocol["identity"] != old_status["identity"] or protocol["split"]["held_out"] != days:
        raise ValueError("Unexpected transfer protocol")
    verify(external.ARCHIVE, protocol["archive_sha256"])
    verify(external.SOURCE, protocol["member_sha256"])
    original_pairs = original / "paired_2021.csv"
    verify(original_pairs, old_status["paired_sha256"])
    historic = pd.read_csv(original_pairs).query(
        "observation == 'estimated' and method in ['Fixed-DDQN', 'Fixed-DDQN+Guard']"
    ).set_index(["day", "seed", "method"], verify_integrity=True)
    if len(historic) != 50:
        raise ValueError("Historical replay bank is incomplete")
    original_models, _, _ = base.audited_inputs()
    models = {}
    for seed in seeds:
        original_path, original_hash = original_models["fixed", seed]
        if original_hash != protocol["checkpoints"][str(seed)]:
            raise ValueError("Historical checkpoint differs from the locked transfer protocol")
        new_path, = checkpoints.glob(f"seed-{seed}-*/status.json")
        new = read(new_path)
        if (new["seed"], new["status"], new["observation"], new["actual_transitions"],
                new["optimizer_updates"], new["oracle_model_sha256"]) != (
                seed, "COMPLETE", "estimated", 2_000_128, 7695, original_hash):
            raise ValueError("Estimated training manifest is incompatible")
        for train, model_path, sha in (
                ("oracle", original_path, original_hash),
                ("estimated", new_path.with_name("final.pt"), new["model_sha256"])):
            verify(model_path, sha)
            ck = torch.load(model_path, weights_only=True, map_location="cpu")
            if train == "estimated" and ck["identity"] != new["id"]:
                raise ValueError("Estimated checkpoint identity differs from its manifest")
            scales = np.array([ck["scales"][k] for k in ("p50", "p99", "unc", "ia", "res")])
            weights = tuple(np.ascontiguousarray(ck["model"][f"net.{k}"].numpy(), dtype=np.float32)
                            for k in ("0.weight", "0.bias", "2.weight", "2.bias", "4.weight", "4.bias"))
            models[train, seed] = scales, weights, sha
        np.testing.assert_array_equal(models["oracle", seed][0], models["estimated", seed][0])

    frame, names, bank, offsets, prefix, means, profiles, excluded = external.prepare(external.SOURCE)
    windows = {day: external.day_workload(frame, day, names, profiles) for day in days}
    fingerprints = {day: hashlib.sha256(x.tobytes() + ids.tobytes()).hexdigest()
                    for day, (x, ids, *_) in windows.items()}
    for day in days:
        if set(historic.loc[day].trace_sha256) != {fingerprints[day]}:
            raise ValueError(f"Window {day} differs from the historical trace")
    bounds_path = root / "outputs/crossed_observations/feasibility_2021_by_day.csv"
    bounds = pd.read_csv(bounds_path).set_index("day", verify_integrity=True)
    for day, (x, *_rest) in windows.items():
        assert len(x) == bounds.loc[day, "requests"]
        assert np.count_nonzero(x[:, 3]/max(base.SPEED) > x[:, 4]) == bounds.loc[day, "intrinsic_misses"]
        np.testing.assert_allclose(x[:, 3].sum()/((x[-1, 0]-x[0, 0])*np.dot(base.SPEED, base.WORKERS)), .6)
    output.mkdir(parents=True, exist_ok=True)
    lock = {
        "source_protocol": protocol["identity"], "source_sha256": protocol["member_sha256"],
        "historical_pairs_sha256": digest(original_pairs), "bounds_sha256": digest(bounds_path),
        "extension_code_sha256": digest(Path(__file__)),
        "historical_code": protocol["code"],
        "released_code": {name: digest(root/"work"/name) for name in protocol["code"]},
        "telemetry_code": {name: digest(root/"work"/name) for name in ("crossed_router.py", "replay_crossed.py")},
        "workloads": {str(day): {"sha256": fingerprints[day], "n": len(windows[day][0]),
                                "start_minute": windows[day][-1]} for day in days},
        "cohort_size": len(names), "excluded_invalid_source_records": excluded,
        "profile_fit": "Only 2021 invocations starting on elapsed days 0–6 and ending before day 7",
        "training": {f"{train}/{seed}": value[2] for (train, seed), value in models.items()},
        "evaluation_observation": "estimated", "guard_thresholds": [.005, .005],
        "residual_scale": 1., "budget": .005, "speeds": base.SPEED.tolist(),
        "workers": base.WORKERS.tolist(), "offered_load": .6,
        "scope": "Frozen external-transfer extension on five previously inspected windows",
    }
    lock_path = output / "transfer_protocol.json"
    if lock_path.exists() and read(lock_path) != lock:
        raise ValueError("Previously locked transfer inputs differ")
    atomic_json(lock_path, lock)

    kernel = crossed_kernel()
    jobs = [(day, seed, train, guard) for day in days for seed in seeds
            for train in ("oracle", "estimated") for guard in (False, True)]

    def replay(item):
        day, seed, train, guard = item
        x, ids, pred, minute = windows[day]
        scale, weights, sha = models[train, seed]
        response = kernel(x, ids, pred["p50"], base.SPEED, base.WORKERS, scale, weights,
                          0, guard, True, 1., bank, offsets, prefix, means, True, .005, .005)
        metrics, _ = base.results(x, response[:, :10])
        if train == "oracle":
            old = historic.loc[day, seed, "Fixed-DDQN+Guard" if guard else "Fixed-DDQN"]
            verify_old(metrics, old, fingerprints[day], sha)
        return dict(day=day, seed=seed, window_start_minute=minute, training=train, guard=guard,
                    trace_sha256=fingerprints[day], model_sha256=sha, **metrics, **queue_metrics(x, response),
                    intrinsic_fraction=bounds.loc[day, "intrinsic_fraction"],
                    pooled_bound_fraction=bounds.loc[day, "pooled_bound_fraction"])

    with threadpool_limits(limits=1):
        first = replay(jobs[0])
        with ThreadPoolExecutor(max_workers=5) as pool:
            rows = [first, *tqdm(pool.map(replay, jobs[1:]), total=99, desc="Frozen transfer replays")]
    data = pd.DataFrame(rows)
    keys = ["day", "seed", "training", "guard"]
    assert len(data) == 100 and not data.duplicated(keys).any()
    assert data.groupby(["day", "seed"]).trace_sha256.nunique().eq(1).all()
    assert data.groupby(["day", "seed", "training"]).model_sha256.nunique().eq(1).all()
    all_metrics = ["violation_rate", "mean_ms", "p95_ms", "p99_ms", "override_rate", "wait_bias_ms",
                   "wait_MAE_ms", "brier", "max_routed_server_load", "saturation_upper", "backlog_p95"]
    groups = data.groupby(["training", "guard"])
    summary = groups[all_metrics].mean().join(groups.met_budget.agg(passes="sum", pairs="size")).reset_index()
    day_groups = data.groupby(["day", "training", "guard"])
    per_day = day_groups[all_metrics].mean().join(day_groups.met_budget.agg(passes="sum", pairs="size")).reset_index()
    outputs = {"transfer_paired.csv": data, "transfer_by_day_seed.csv": data[keys + all_metrics + [
                   "met_budget", "intrinsic_fraction", "pooled_bound_fraction"]],
               "transfer_summary.csv": summary, "transfer_by_day.csv": per_day}
    for name, result in outputs.items():
        temp = (output/name).with_suffix(".tmp")
        result.to_csv(temp, index=False)
        temp.replace(output/name)
    atomic_json(output/"transfer_manifest.json", {
        "protocol_sha256": digest(lock_path), "historical_replays_verified": 50, "new_replays": 50,
        "outputs": {name: digest(output/name) for name in outputs},
    })
    print(summary.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("artifact", "checkpoints", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    options = parser.parse_args()
    run(options.artifact, options.checkpoints, options.output)
