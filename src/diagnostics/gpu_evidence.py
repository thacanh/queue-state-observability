"""Verify returned Colab jobs and summarize the predeclared paired comparisons."""

from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile
import os

import replay_study as study

os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                  MKL_THREADING_LAYER="SEQUENTIAL")

import numpy as np
import pandas as pd

from analyze_new import OUT, compare, fig_save, grouped, plt, save, verified


STAGES = ("T4", "T8")
GPU = study.BASE / "gpu_results"
MATCH_KEYS = ["key", "day", "seed", "guard", "trace_sha256"]
QUANTITIES = ["delta_violation_rate", "delta_p95_ms", "delta_p99_ms",
              "delta_met_budget", "delta_override_rate"]


def planned(phase):
    plan = study.read(study.BASE / phase / "planned_protocol.json")
    identity = plan.pop("protocol_sha256")
    assert study.fingerprint(plan) == identity
    return plan


def jobs(phase):
    return [f"{mode}_{seed}_lambda2" for seed, mode in planned(phase)["jobs"]]


def import_results(source):
    allowed = {"batch_status.json"}
    for phase in STAGES:
        allowed.add(f"{phase}/training_protocol.json")
        for job in jobs(phase):
            for name in ("status.json", "final.pt"):
                allowed.add(f"{phase}/{job}/{name}")
    with ZipFile(source) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) - allowed or archive.testzip() is not None:
            raise ValueError("Colab result archive has unexpected, repeated, or corrupted files")
        for name in names:
            target = GPU / name
            target.parent.mkdir(parents=True, exist_ok=True)
            data = archive.read(name)
            if target.is_file() and study.sha256(data).hexdigest() == study.digest(target):
                continue
            if target.is_file():
                old = target.with_name(f"{target.name}.{study.digest(target)[:12]}.superseded")
                if not old.exists():
                    old.write_bytes(target.read_bytes())
            tmp = target.with_suffix(".tmp")
            tmp.write_bytes(data)
            tmp.replace(target)
    study.write(GPU / "import_record.json", {"archive_sha256": study.digest(source),
        "archive_bytes": source.stat().st_size, "files": names,
        "imported_utc": datetime.now(timezone.utc).isoformat()})


def check_training(phase):
    plan = planned(phase)
    root = GPU / phase
    runtime_plan = study.read(root / "training_protocol.json")
    assert runtime_plan["phase"] == phase
    assert runtime_plan["source"]["gpu_training.py"] == plan["source_sha256"]
    budget = runtime_plan["training_budget"]
    assert budget == {"transitions": plan["states_per_job"], "updates": plan["updates_per_job"],
                      "active_session_cap_s": plan["budget_limits_s"][0]}
    assert runtime_plan["jobs"] == [[*job, 2] for job in plan["jobs"]]
    assert runtime_plan["runtime"]["gpu"] == "Tesla T4"
    manifest_hash = study.fingerprint(runtime_plan)
    records = []
    for (seed, mode), folder in zip(plan["jobs"], jobs(phase)):
        marker = root / folder / "status.json"
        m = study.read(marker)
        if m["status"] != "COMPLETE":
            raise ValueError(f"{phase} {folder} is {m['status']}: {m.get('error', '')}")
        weights = root / folder / "final.pt"
        assert m["seed"] == seed and m["observation"] == mode and m["lambda"] == 2
        assert m["phase"] == phase and m["protocol"] == manifest_hash
        assert m["model_sha256"] == study.digest(weights)
        assert m["runtime"] == runtime_plan["runtime"]
        assert m["calendar"] == "shared Numba float32" and not m["guard_in_training"]
        assert m["actual_transitions"] == plan["states_per_job"] and m["optimizer_updates"] == plan["updates_per_job"]
        assert m["config"]["checkpoint_every"] == plan["same_save_cadence_batches"]
        assert m["config"]["target_update"] == plan["target_update_batches"]
        initial = plan["training"][str(seed)]["stage1"] if phase == "T4" else plan["initialization"][str(seed)]
        assert m["initial_checkpoint_sha256"] == initial
        if phase == "T4":
            assert m["input_identity"] == plan["training"][str(seed)]
        else:
            prefix = plan["training_prefix_only"]
            assert m["input_identity"] == {"train": prefix["training_csv_sha256"],
                "bank": prefix["training_bank_sha256"], "normalization": plan["fit_scales_sha256"]}
        rss = [s["rss_bytes"] for s in m["resource_samples"]]
        gpu_used = [s["gpu_memory_mib"] for s in m["resource_samples"] if "gpu_memory_mib" in s]
        gpu_util = [s["gpu_util_pct"] for s in m["resource_samples"] if "gpu_util_pct" in s]
        records.append(dict(phase=phase, seed=seed, mode=mode, gpu=m["runtime"]["gpu"],
            transitions=m["actual_transitions"], optimizer_updates=m["optimizer_updates"],
            active_s=m["total_active_seconds"], last_session_cpu_s=m["cpu_seconds"],
            peak_sampled_rss_mib=max(rss, default=0)/2**20,
            peak_torch_reserved_mib=m["peak_reserved_cuda_bytes"]/2**20,
            peak_sampled_gpu_mib=max(gpu_used, default=0),
            mean_sampled_gpu_util_pct=float(np.mean(gpu_util)) if gpu_util else None,
            checkpoint_bytes=weights.stat().st_size, checkpoint_sha256=m["model_sha256"],
            manifest_sha256=study.digest(marker), identity=m["identity"]))
    assert sum(r["active_s"] for r in records) < plan["budget_limits_s"][0]
    return records


def audit_checkpoint_tensors(records):
    import torch

    def payload(weights):
        return {key: {"dtype": str(value.dtype), "shape": list(value.shape),
                      "sha256": study.sha256(value.contiguous().numpy().tobytes()).hexdigest()}
                for key, value in weights.items()}

    calibration = study.read(study.BASE / "T7/calibration.json")["scales"]
    audit = []
    for row in records:
        phase, mode, seed = row["phase"], row["mode"], row["seed"]
        path = GPU / phase / f"{mode}_{seed}_lambda2" / "final.pt"
        old_path = study.ARTIFACT / "inputs/checkpoints" / mode / f"seed{seed}.pt"
        old_sha = study.digest(old_path)
        expected = planned(phase)
        assert old_sha == (expected["historical_comparators"][f"{mode}/{seed}"] if phase == "T4"
                           else expected["initialization"][str(seed)])
        a = torch.load(path, map_location="cpu", weights_only=True)
        b = torch.load(old_path, map_location="cpu", weights_only=True)
        assert a["meta"]["updates"] == row["optimizer_updates"]
        assert a["meta"]["steps"] * a["config"]["nenv"] == row["transitions"]
        assert a["identity"] == a["config"]["identity"] == row["identity"]
        assert all(a["scales"][k] == (calibration.get(k, b["scales"][k]) if phase == "T8" else b["scales"][k])
                   for k in study.KEYS)
        assert a["model"].keys() == b["model"].keys()
        first, second = payload(a["model"]), payload(b["model"])
        audit.append(dict(phase=phase, mode=mode, seed=seed,
                          candidate_checkpoint_sha256=study.digest(path), archived_checkpoint_sha256=old_sha,
                          candidate_tensor_sha256=study.fingerprint(first), archived_tensor_sha256=study.fingerprint(second),
                          tensors_byte_identical=first == second, tensor_count=len(first),
                          max_abs_weight_difference=max(float((a["model"][k] - b["model"][k]).abs().max())
                                                        for k in first)))
    save(pd.DataFrame(audit), "gpu_checkpoint_tensor_audit")


def paired_summary(pairs, labels):
    fields = {key: (key, "mean") for key in QUANTITIES}
    return pairs.groupby(labels, as_index=False).agg(
        pairs=("key", "size"), fail_to_pass=("fail_to_pass", "sum"),
        pass_to_fail=("pass_to_fail", "sum"), **fields)


def conditional_intervals(data):
    """Resample source days, seeds and shared draws within each day."""
    rng = np.random.default_rng(20261127)
    count = 4000
    rows = []
    for (bank, guard, contrast), sub in data.groupby(["bank", "guard", "contrast"]):
        days = []
        for _, day in sub.groupby("day"):
            a = day.sort_values(["key", "seed"])
            assert a.groupby("key").seed.nunique().eq(2).all()
            days.append(a[QUANTITIES].to_numpy().reshape(-1, 2, len(QUANTITIES)))
        day_indices = rng.integers(len(days), size=(count, len(days)))
        seeds = rng.integers(2, size=(count, 2))
        totals = np.zeros((count, len(QUANTITIES)))
        workload_counts = np.zeros(count)
        for idx in day_indices.T:
            for i, day in enumerate(days):
                which = np.flatnonzero(idx == i)
                n = len(day)
                draws = rng.integers(n, size=(len(which), n))
                totals[which] += day[draws[:, :, None], seeds[which, None, :]].sum(axis=(1, 2))
                workload_counts[which] += n * 2
        samples = totals / workload_counts[:, None]
        lo, hi = np.quantile(samples, [.025, .975], axis=0)
        for j, metric in enumerate(QUANTITIES):
            rows.append(dict(bank=bank, guard=guard, contrast=contrast, metric=metric,
                             estimate=sub[metric].mean(), lower=lo[j], upper=hi[j],
                             replicates=count, days=len(days), seed_count=2))
    return pd.DataFrame(rows)


def analyze_t4():
    data, protocol = verified("T4")
    plan = planned("T4")
    assert {key: v["trace_sha256"] for key, v in protocol["workloads"].items()} == plan["workloads"]
    data["bank"] = np.where(data.day.isin(["d08", "d09"]), "Original", "Later")
    data["observation"] = data["mode"].str.split("_").str[0]
    assert len(data) == 1424 and data.reference_reused.sum() == 712 and data.key.nunique() == 89
    np.testing.assert_allclose(data.violation_rate, data.violations / data.n, atol=1e-15, rtol=0)
    assert data.groupby(MATCH_KEYS + ["observation", "reference_reused"]).size().eq(1).all()
    keys = MATCH_KEYS + ["bank"]
    old, new = data[data.reference_reused], data[~data.reference_reused]
    for (mode, seed), rows in old.groupby(["observation", "seed"]):
        assert rows.model_sha256.eq(plan["historical_comparators"][f"{mode}/{seed}"]).all()
    for (mode, seed), rows in new.groupby(["observation", "seed"]):
        assert rows.model_sha256.eq(protocol["checkpoints"][f"{mode}_lambda2/{seed}"]).all()
    conditions = [
        ("Estimated Numba minus Oracle Numba", new[new.observation.eq("estimated")], new[new.observation.eq("oracle")]),
        ("Oracle Numba minus archived Oracle Torch", new[new.observation.eq("oracle")], old[old.observation.eq("oracle")]),
        ("Estimated Numba minus archived Estimated Numba", new[new.observation.eq("estimated")], old[old.observation.eq("estimated")]),
    ]
    pairs = pd.concat([compare(a, b, keys, []).assign(contrast=name) for name, a, b in conditions], ignore_index=True)
    assert len(pairs) == 1068
    save(pairs, "t4_paired_detail")
    save(paired_summary(pairs, ["bank", "day", "seed", "guard", "contrast"]), "t4_paired_by_seed_day")
    save(paired_summary(pairs, ["bank", "day", "guard", "contrast"]), "t4_paired_by_day")
    summary = paired_summary(pairs, ["bank", "guard", "contrast"])
    save(summary, "t4_paired_by_bank")
    save(conditional_intervals(pairs), "t4_conditional_intervals")
    save(grouped(data, ["bank", "day", "seed", "guard", "observation", "reference_reused"]), "t4_arms")
    save(grouped(data, ["bank", "guard", "observation", "reference_reused"]), "t4_arms_by_bank")
    return new, pairs


def analyze_t8():
    data, protocol = verified("T8")
    plan = planned("T8")
    assert {key: v["trace_sha256"] for key, v in protocol["workloads"].items()} == plan["workloads"]
    reference, _ = verified("T7")
    assert study.read(study.BASE / "T7/execution.json")["protocol"] == plan["reference_protocol"]
    assert len(data) == 16 and data.reference_reused.sum() == 8
    np.testing.assert_allclose(data.violation_rate, data.violations / data.n, atol=1e-15, rtol=0)
    new, old = data[~data.reference_reused], data[data.reference_reused]
    for seed, rows in new.groupby("seed"):
        assert rows.model_sha256.eq(protocol["checkpoints"][f"estimated_lambda2/{seed}"]).all()
    for seed, rows in old.groupby("seed"):
        assert rows.model_sha256.eq(plan["initialization"][str(seed)]).all()
    columns = MATCH_KEYS + ["model_sha256", "n", "violations", "met_budget", "p95_ms", "p99_ms"]
    locked = reference[reference.recalibrated & reference["mode"].eq("estimated")]
    canonical = lambda frame: frame[columns].sort_values(MATCH_KEYS).reset_index(drop=True)
    pd.testing.assert_frame_equal(canonical(old), canonical(locked), check_exact=False, rtol=1e-12, atol=1e-12)
    pair = compare(new, old, MATCH_KEYS, [])
    save(pair, "t8_paired_detail")
    save(paired_summary(pair, ["day", "guard"]), "t8_paired_by_day")
    save(grouped(data, ["day", "guard", "reference_reused"]), "t8_arms")
    return data, pair


def draw_comparisons(t4, t8):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.55))
    for ax, bank in zip(axes, ("Original", "Later")):
        day = t4[t4.bank.eq(bank)]
        for mode, color in (("oracle", "#0c739b"), ("estimated", "#ad5135")):
            for guard, marker in ((False, "x"), (True, "o")):
                rows = day[day.observation.eq(mode) & day.guard.eq(guard)]
                for i, s in rows.groupby("seed"):
                    ax.scatter(s.p95_ms.mean(), s.violation_rate.mean()*100, marker=marker, color=color,
                               label=f"{mode.title()} FT, {'guard' if guard else 'no guard'}" if i == 42 else None)
        ax.axhline(.5, lw=.8, ls=":", c=".45")
        ax.set(title=bank + " Azure 2019 bank", xlabel="Mean workload P95 (ms)", ylabel="Mean miss rate (%)")
        ax.legend(fontsize=7)
    fig_save(fig, "t4_held_calendar_tradeoff")
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.4))
    for ax, day in zip(axes, (7, 8)):
        x = t8[t8.day.eq(day)]
        for guard, color in ((False, "#0c739b"), (True, "#ad5135")):
            rows = x[x.guard.eq(guard)]
            for _, s in rows.groupby("seed"):
                p = s.sort_values("reference_reused", ascending=False)
                ax.plot(p.p95_ms / 1000, p.violation_rate * 100, color=color, lw=1.0)
            for archived, marker in ((True, "o"), (False, "^")):
                s = rows[rows.reference_reused.eq(archived)]
                ax.scatter(s.p95_ms / 1000, s.violation_rate * 100, marker=marker, color=color,
                           label=f"{'Frozen' if archived else 'Prefix FT'}, {'guard' if guard else 'no guard'}",
                           s=26)
        ax.axhline(.5, lw=.8, ls=":", c=".45")
        ax.set(title=f"Azure 2021 elapsed day {day}", xlabel="Workload P95 (s)", ylabel="Miss rate (%)")
        ax.legend(fontsize=7)
    fig_save(fig, "t8_prefix_finetuning_tradeoff")


def main():
    cli = ArgumentParser(description=__doc__)
    cli.add_argument("--import-zip", type=Path)
    cli.add_argument("--training-only", action="store_true")
    args = cli.parse_args()
    if args.import_zip:
        import_results(args.import_zip)
    batch = GPU / "batch_status.json"
    if batch.is_file():
        assert study.read(batch) == {p: 0 for p in STAGES}
    marks = [r for p in STAGES for r in check_training(p)]
    OUT.mkdir(exist_ok=True)
    save(pd.DataFrame(marks), "gpu_training_resources")
    print(f"Verified {len(marks)} completed Tesla T4 checkpoints.")
    if not args.training_only:
        audit_checkpoint_tensors(marks)
        t4, _ = analyze_t4()
        t8, _ = analyze_t8()
        draw_comparisons(t4, t8)


if __name__ == "__main__":
    main()
