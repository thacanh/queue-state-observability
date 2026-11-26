"""Run observation perturbations, event logging, and training-only calibration."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from pathlib import Path
import time

import replay_study as study


def base(phase, frozen, keys, models, description, plan):
    return study.base_protocol(phase, frozen, keys, sorted({s for _, s in models}), models, description,
                               plan | {"diagnostic_code": study.digest(Path(__file__)),
                                       "kernel_adapter": study.digest(Path(__file__).with_name("diagnostic_kernel.py"))})


def diagnostic_args(inputs, checkpoint, guard, bank, noise, channel=0, bias=0., sigma=0.,
                    scale=1., method=0):
    return (*study.arguments(inputs, checkpoint, guard, bank, scale=scale, method=method),
            channel, bias, sigma, noise)


def blank():
    return np.zeros((1, len(study.WORKERS), int(max(study.WORKERS))))


def warm_check(kernel, frozen, templates, key):
    inputs = frozen.workload(key)
    checkpoint = templates["oracle", 42]
    params = diagnostic_args(inputs, checkpoint, True, frozen.bank, blank())
    kernel.compile(tuple(study.typeof(v) for v in params))
    out = kernel(*params)
    reference = study.archived_outcome(frozen, key, 42, "oracle", True, checkpoint[2])
    stats = study.results(inputs[0], out[:, :10])[0]
    if frozen.transfer:
        stats.update(queue_metrics(inputs[0], out))
    study.check_outcome(stats, reference)


def row(frozen, key, seed, mode, ck):
    return dict(key=key, day=frozen.traces[key]["day"], seed=seed, mode=mode, model_sha256=ck[2],
                trace_sha256=frozen.traces[key]["trace_sha256"])


def dose_response(kernel):
    frozen = study.Frozen()
    keys = frozen.later_keys
    models = {(mode, s): frozen.checkpoint(mode, s) for mode, s in product(study.MODES, (42, 43))}
    doses = [("zero", 0., 0.)] + [("bias", x, 0.) for x in (-30, 30, 60)] + [
        ("jitter", 0., x) for x in (15, 30, 60)]
    spec = base("T5", frozen, keys, models,
        "Show every setting. Within each channel, mean outcomes by day and paired changes vs zero "
        "and genuine estimator. Errors over all valid workers on each arm trajectory. No population inference.",
        {"noise_seed": 9261105, "noise_generator": "NumPy PCG64 standard normal; one tape per workload",
         "channels": {"policy_only": "perturb oracle residuals, guard off",
                      "guard_only": "estimated policy, perturb guard oracle residuals"},
         "doses_bias_sigma_ms": [list(q) for q in doses], "expected_new": 336, "expected_reused": 48,
         "idle_worker_policy": "preserve valid idle workers and padding, truncate active residuals at zero",
         "anchor": "use unchanged genuine estimated policy and, when enabled, estimated guard"})

    def tasks(key):
        inputs = frozen.workload(key)
        child = np.random.SeedSequence([9261105, int(key[1:3]), int(key[-2:])])
        tape = np.random.default_rng(child).standard_normal(
            size=(len(inputs[0]), len(study.WORKERS), int(max(study.WORKERS))))
        return [(key, inputs, mode, s, channel, kind, b, d, tape)
                for (mode, s), channel, (kind, b, d) in product(models, (1, 2), doses + [("estimated", 0., 0.)])]

    def evaluate(item):
        key, inputs, mode, s, channel, kind, b, d, tape = item
        ck = models[mode, s]
        guard = channel == 2
        reference = kind == "estimated"
        if reference:
            stats = study.archived_outcome(frozen, key, s, mode, guard, ck[2])
        else:
            out = kernel(*diagnostic_args(inputs, ck, guard, frozen.bank, tape, channel, b/1000, d/1000))
            stats = study.results(inputs[0], out[:, :10])[0]
            stats.update(queue_metrics(inputs[0], out))
            pos = 27 if channel == 1 else 29
            stats.update(worker_bias_ms=1000*float(out[:, pos].mean()),
                         worker_mae_ms=1000*float(out[:, pos+1].mean()))
        return row(frozen, key, s, mode, ck) | dict(channel="guard" if guard else "policy", kind=kind,
              bias_ms=b, sigma_ms=d, guard=guard, reference_reused=reference, **stats)

    study.run_shards("T5", spec, keys, tasks, evaluate, lambda: warm_check(kernel, frozen, models, keys[0]),
                     336, 48)


def event_trace(key, inputs, mode, seed, guard, out):
    path = study.BASE / "T6" / "events" / f"{key}_{mode}_{seed}_{int(guard)}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    x, ids, _ = inputs
    arr = x[:, 0]
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, request_id=np.arange(len(x), dtype=np.int32), function_id=ids.astype(np.int16),
             arrival_s=arr, deadline_s=x[:, 4], demand_s=x[:, 3], proposal=out[:, 2].astype(np.int8),
             selected=out[:, 1].astype(np.int8), guard_activated=out[:, 26].astype(bool),
             override=(out[:, 2]!=out[:, 1]), proposal_risk=out[:, 4], selected_risk=out[:, 3],
             estimated_wait_s=out[:, [31,38,45]], actual_wait_s=out[:, [34,41,48]],
             queued=out[:, [35,42,49]].astype(np.int32), estimated_pending_demand=out[:, [36,43,50]],
             actual_pending_demand=out[:, [37,44,51]], start_s=out[:, 24], completion_s=out[:, 25],
             miss=(out[:, 0] > x[:, 4]), proposed_miss_same_state=out[:, 6].astype(bool))
    tmp.replace(path)
    return {"event_file": path.name, "event_sha256": study.digest(path), "event_bytes": path.stat().st_size}


def transfer_events(kernel):
    frozen = study.Frozen(transfer=True)
    keys = frozen.window_keys
    models = {(mode,s): frozen.checkpoint(mode,s) for mode,s in product(study.MODES, range(42,47))}
    spec = base("T6", frozen, keys, models,
        "Verify every row against archive before writing telemetry. Compare same request ID on matched "
        "on/off runs; group by elapsed day, training configuration and checkpoint seed. Event times descriptive.",
        {"fixed_eval_arms": ["oracle_off","oracle_on","estimated_off","estimated_on"],
         "horizons_s": [1,10,60], "future_request_counts": [10,100],
         "server_load": "work routed within horizon divided by target server speed × workers × horizon",
         "snapshot": "proposed/actual route miss on the guarded run's predecision state",
         "storage_cap_bytes": 250_000_000, "source_archive": study.digest(study.TEMPORAL/"transfer/replays.csv"),
         "expected_new": 40})

    def tasks(key):
        inputs = frozen.workload(key)
        return [(key, inputs, mode, seed, guard) for (mode, seed), guard in product(models, (False,True))]

    def evaluate(item):
        key, inputs, mode, seed, guard = item
        ck = models[mode, seed]
        out = kernel(*diagnostic_args(inputs, ck, guard, frozen.bank, blank()))
        stats = study.results(inputs[0], out[:, :10])[0]
        stats.update(queue_metrics(inputs[0], out))
        study.check_outcome(stats, study.archived_outcome(frozen, key, seed, mode, guard, ck[2]))
        event = event_trace(key, inputs, mode, seed, guard, out)
        return row(frozen, key, seed, mode, ck) | dict(guard=guard, archive_match=True, **event, **stats)

    study.run_shards("T6", spec, keys, tasks, evaluate, lambda: warm_check(kernel, frozen, models, keys[0]), 40)
    stored = sum(p.stat().st_size for p in (study.BASE/"T6/events").glob("*.npz"))
    assert stored <= spec["plan"]["storage_cap_bytes"]


def training_map(frame, archive, path):
    """Verify day boundaries and feature provenance independently of evaluation."""
    q_cols = [f"percentile_Average_{i}" for i in (1, 25, 50, 75, 99)]
    keys = ["HashOwner","HashApp","HashFunction"]
    sel_path = archive/"processed/selected_functions_d01_v2.csv"
    selected = pd.read_csv(sel_path).drop_duplicates(keys).head(30)[["func_id", *keys]]
    labels = [f"d{i:02d}" for i in range(1,8)]
    gaps = frame.arrival.diff().to_numpy()
    cuts = np.flatnonzero(np.isclose(gaps, 600, atol=1e-9, rtol=0))
    assert len(frame)==868593 and len(cuts)==6 and not (np.delete(gaps[1:], cuts-1) >= 600).any()
    sources = {str(sel_path.relative_to(study.ROOT)): study.digest(sel_path)}
    profiles = []
    for label in labels:
        file = archive/"data"/f"function_durations_percentiles.anon.{label}.csv"
        sources[str(file.relative_to(study.ROOT))] = study.digest(file)
        raw = pd.read_csv(file, usecols=keys+q_cols, low_memory=False)
        view = raw.merge(selected, on=keys, how="inner")
        for col in q_cols:
            view[col] = pd.to_numeric(view[col], errors="coerce")
        grouped = view.groupby(["func_id"], as_index=False)[q_cols].median()
        q = grouped[q_cols].to_numpy()
        valid = np.isfinite(q).all(1) & (q>0).all(1) & (np.diff(q, axis=1)>=0).all(1)
        profiles.append(grouped.loc[valid])
    intervals = list(zip(np.r_[0,cuts], np.r_[cuts,len(frame)]))
    records = {}
    for i, (label, (lo, hi)) in enumerate(zip(labels, intervals)):
        prev = pd.concat(profiles[:max(i, 1)]).groupby("func_id")[q_cols].median()
        own = profiles[i].set_index("func_id")
        part = frame.iloc[lo:hi]
        for feature, column in (("est_p50",q_cols[2]),("est_p99",q_cols[4])):
            np.testing.assert_allclose(part[feature], part.func_id.map(prev[column])/1000, atol=1e-12, rtol=0)
        lo_d = part.func_id.map(own[q_cols[0]]).to_numpy()/1000
        hi_d = part.func_id.map(own[q_cols[4]]).to_numpy()/1000
        assert ((part.service_demand.to_numpy() >= lo_d-1e-12) & (part.service_demand.to_numpy() <= hi_d+1e-12)).all()
        records[label] = {"lo": int(lo), "hi": int(hi), "rows": int(hi-lo),
                          "table_hash": study.sha256(pd.util.hash_pandas_object(part,index=True).values.tobytes()).hexdigest()}
    return {"trace_bytes_sha256": study.digest(path), "boundary_gap_s": 600,
            "source_days": records, "profile_sources": sources,
            "provenance_rules": "Legacy ordered train days, six 600-s concatenation gaps, each day independently checked against preceding-day profile features and same-day duration support."}


def train_calibrate(kernel):
    from scipy.optimize import minimize_scalar
    frozen = study.Frozen()
    keys = frozen.later_keys
    models = {(mode,s): frozen.checkpoint(mode,s) for mode,s in product(study.MODES, (42,43))}
    source = study.ROOT/"outputs/shield_verified/source_data"
    path = source/"multiday_final_5seeds/traces/train_d01_d07.csv"
    expected = frozen.protocol["inputs"]["2019_training_bank"]["sha256"]
    assert study.digest(path) == expected
    frame = pd.read_csv(path, float_precision="round_trip")
    provenance = training_map(frame, source, path)
    spec = base("T3", frozen, keys, models,
        "Fit and report training-only fixed-trajectory same-state labels, then compare fully re-routed later evaluation "
        "trajectories by workload and seed against exact archived original estimator. Report signed and absolute wait, "
        "miss and tail, attained budget and guard behavior. Positive waits and all-state metrics reported separately.",
        {"training": provenance, "bank": frozen.bank_sha, "model": "v_corrected = c * v; zero remains zero",
         "fit": "minimize equal-day/equal-server Huber loss over estimated wait > 0; Huber delta .05s; c in [0.25,2.5]",
         "witness": "Estimated SED-P50, guard off. All training arrivals and valid servers; reset at every source day.",
         "expected_new": 48, "expected_training_telemetry": 7})
    folder = study.BASE/"T3"
    ident = study.lock(folder/"protocol.json", spec)
    ck = models["oracle", 42]
    labels = list(provenance["source_days"])
    clock = time.perf_counter()

    def collect(label):
        segment = provenance["source_days"][label]
        sub = frame.iloc[segment["lo"]:segment["hi"]]
        x = np.ascontiguousarray(sub[study.XCOLS].to_numpy(np.float64))
        x[:,0] -= x[0,0]
        ids = sub.func_id.astype(str).map(frozen.names).to_numpy(np.int64)
        out = kernel(*diagnostic_args((x,ids,x[:,1].copy()), ck, False, frozen.bank, blank(), method=1))
        return label, out[:,[31,38,45]].copy(), out[:,[34,41,48]].copy()

    def cached_collect(label):
        file = folder/f"{label}_witness.npz"
        if file.exists():
            with np.load(file, allow_pickle=False) as data:
                assert str(data["protocol"])==ident
                return label, data["estimated"], data["truth"]
        label, estimate, truth = collect(label)
        tmp = file.with_suffix(".tmp")
        with tmp.open("wb") as f:
            np.savez_compressed(f, protocol=ident, estimated=estimate, truth=truth)
        tmp.replace(file)
        return label, estimate, truth

    with study.threadpool_limits(limits=1):
        warm_check(kernel, frozen, models, keys[0])
        with ThreadPoolExecutor(max_workers=4) as pool:
            data = list(pool.map(cached_collect, labels))
    groups = [(a[:,j][a[:,j]>0], b[:,j][a[:,j]>0]) for _,a,b in data for j in range(a.shape[1])]
    assert all(len(a) for a,b in groups)

    def objective(c, subsets=groups):
        values = []
        for a,b in subsets:
            e = np.abs(a*c-b)
            h = np.minimum(e,.05)
            values.append(np.mean(.5*h*h + .05*(e-h)))
        return np.mean(values)

    fitted = minimize_scalar(objective, method="bounded", bounds=(.25,2.5), options={"xatol":1e-9})
    if not fitted.success:
        raise ValueError(fitted.message)
    scalar = float(fitted.x)
    calibration = []
    for i,(day,estimate,truth) in enumerate(data):
        other_days = groups[:3*i]+groups[3*(i+1):]
        fold = minimize_scalar(lambda c: objective(c, other_days), method="bounded", bounds=(.25,2.5),
                               options={"xatol": 1e-9})
        for j in range(estimate.shape[1]):
            for subset, mask in (("all",np.ones(len(estimate),bool)), ("positive",estimate[:,j]>0)):
                for fit, factor in (("original",1.),("pooled_train",scalar),("other_train_days",float(fold.x))):
                    e = estimate[:,j][mask]*factor-truth[:,j][mask]
                    calibration.append(dict(day=day,server=j+1,fit=fit,subset=subset,n=int(len(e)),
                                            factor=factor, bias_ms=1000*float(e.mean()), mae_ms=1000*float(np.abs(e).mean())))
    pd.DataFrame(calibration).to_csv(folder/"fixed_trajectory_calibration.csv",index=False)
    study.write(folder/"calibration.json", {"protocol":ident, "scale":scalar, "loss_before":float(objective(1.)),
                "loss_after":float(fitted.fun), "elapsed_s":time.perf_counter()-clock,
                "witness_files":{f.name:study.digest(f) for f in folder.glob("*_witness.npz")}})

    def tasks(key):
        inputs = frozen.workload(key)
        return [(key,inputs,mode,seed,guard,corrected) for (mode,seed),guard,corrected
                in product(models,(False,True),(False,True))]

    def evaluate(task):
        key, inputs, mode, seed, guard, corrected = task
        ck = models[mode,seed]
        if corrected:
            out = kernel(*diagnostic_args(inputs, ck, guard, frozen.bank, blank(), scale=scalar))
            stats = study.results(inputs[0],out[:,:10])[0]
        else:
            stats = study.archived_outcome(frozen,key,seed,mode,guard,ck[2])
        return row(frozen,key,seed,mode,ck) | dict(guard=guard,corrected=corrected,
            reference_reused=not corrected,scale=(scalar if corrected else 1.), **stats)

    study.run_shards("T3", spec, keys, tasks, evaluate, lambda:None, 48, 48)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=["T3","T5","T6"])
    args = cli.parse_args()
    study.dependencies()
    global np, pd, queue_metrics
    from replay_crossed import queue_metrics
    from diagnostic_kernel import make_kernel
    np, pd = study.np, study.pd
    phase = {"T3":train_calibrate, "T5":dose_response, "T6":transfer_events}[args.phase]
    phase(make_kernel())


if __name__ == "__main__":
    main()
