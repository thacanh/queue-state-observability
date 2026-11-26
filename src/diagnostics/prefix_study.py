"""Fit feature scales solely on the Azure 2021 profile prefix."""

from concurrent.futures import ThreadPoolExecutor
from itertools import product
from pathlib import Path
import time

import diagnostics as diag
import replay_study as study


DAYS = range(7)
FIT_DAYS = (0, 1, 2, 3)
SECONDS_PER_DAY = 86400


def prepare_prefix(frozen):
    import replay_transfer as prior

    evidence = frozen.protocol["inputs"]["2021_source"]
    source = study.ROOT / "outputs/guard_evaluation/external" / Path(
        evidence["path"].replace("\\", "/")).name
    assert study.digest(source) == evidence["sha256"]
    fields = ["app", "func", "end_timestamp", "duration"]
    blocks = []
    for chunk in pd.read_csv(source, usecols=fields, chunksize=500_000):
        end, duration = chunk.end_timestamp, chunk.duration
        use = (np.isfinite(end) & np.isfinite(duration) & (duration>0) &
               (end>=duration) & (end<7*SECONDS_PER_DAY) & chunk.app.notna() & chunk.func.notna())
        part = chunk.loc[use].copy()
        part["start"] = part.end_timestamp - part.duration
        part["day"] = (part.start // SECONDS_PER_DAY).astype(np.int64)
        part["id"] = part.app.astype(str)+"/"+part.func.astype(str)
        blocks.append(part)
    frame = pd.concat(blocks, ignore_index=True)
    assert frame.day.between(0,6).all() and frame.end_timestamp.lt(7*SECONDS_PER_DAY).all()
    ranking = frame.groupby("id").agg(n=("duration","size"),days=("day","nunique")).reset_index()
    candidates = ranking.loc[(ranking.n>=200)&(ranking.days>=3)].sort_values(["n","id"],ascending=[False,True])
    names = {name: i for i,name in enumerate(sorted(candidates.id.head(30)))}
    frame = frame.loc[frame.id.isin(names)].copy()
    svc = [np.sort(frame.loc[frame.id==label,"duration"].to_numpy()) for label in names]
    bank = np.concatenate(svc)
    offsets = np.r_[0,np.cumsum([len(values) for values in svc])]
    np.testing.assert_array_equal(bank, frozen.bank[0])
    np.testing.assert_array_equal(offsets, frozen.bank[1])
    profiles = np.array([np.quantile(values,[.5,.95,.99]) for values in svc])
    with np.load(frozen.bank_path, allow_pickle=False) as previous:
        np.testing.assert_array_equal(profiles,previous["profiles"])
    traces = {}
    for d in DAYS:
        x,ids,_,minute = prior.day_workload(frame,d,names,profiles)
        assert np.all((ids>=0)&(ids<len(names)))
        traces[d] = (x,ids,x[:,1].copy(),int(minute))
    return names, traces


def save_prefix(frozen, names, traces):
    folder = study.BASE / "prefix_inputs"
    folder.mkdir(parents=True, exist_ok=True)
    streams = []
    metadata = {}
    shift = 0.
    for d,(x,ids,_,start) in traces.items():
        part = pd.DataFrame(x,columns=study.XCOLS)
        part["arrival"] += shift
        part["func_id"] = ids.astype(str)
        streams.append(part)
        shift = float(part.arrival.iloc[-1])+600.
        metadata[str(d)] = {"n":len(x),"window_start_minute":start,
                            "trace_sha256":study.sha256(x.tobytes()+ids.tobytes()).hexdigest()}
    path = folder/"prefix_train.csv"
    pd.concat(streams,ignore_index=True).to_csv(path.with_suffix(".tmp"),index=False)
    path.with_suffix(".tmp").replace(path)
    target = folder/"prefix_bank.npz"
    with target.with_suffix(".tmp").open("wb") as f:
        np.savez(f,bank=frozen.bank[0],offsets=frozen.bank[1],
                 functions=np.array([str(i) for i in range(len(names))]))
    target.with_suffix(".tmp").replace(target)
    result = {"source_sha256":frozen.protocol["inputs"]["2021_source"]["sha256"],
              "source_cutoff":"arrival in elapsed days 0–6 and completion strictly before start of day 7",
              "days":metadata,"frozen_bank_value_match":True,
              "training_csv_sha256":study.digest(path),"training_bank_sha256":study.digest(target)}
    study.lock(folder/"manifest.json", result)
    return result


def raw_features(x,out):
    p50 = np.maximum(x[:,1],1e-6)
    p99 = np.maximum(p50,x[:,2])
    req = np.column_stack([np.log1p(p50),np.log1p(p99),p99/p50,np.log1p(np.diff(x[:,0],prepend=x[0,0]))])
    queue = np.concatenate([out[:,j:j+3]/np.array([1,1,2]) for j in (31,38,45)],axis=1)
    return tuple(req.T)+(queue.ravel(),)


def main():
    started = time.perf_counter()
    study.dependencies()
    global np,pd
    np,pd = study.np,study.pd
    diag.np, diag.pd = np,pd
    from replay_crossed import queue_metrics
    from diagnostic_kernel import make_kernel
    diag.queue_metrics = queue_metrics
    kernel = make_kernel()
    frozen = study.Frozen(transfer=True)
    keys = frozen.window_keys
    seeds = (42,43)
    models = {(mode,s):frozen.checkpoint(mode,s) for mode,s in product(study.MODES,seeds)}
    names,traces = prepare_prefix(frozen)
    prefix = save_prefix(frozen,names,traces)
    spec = diag.base("T7",frozen,keys,models,
        "Use 16 unguarded prefix-only calibration trajectories. Apply one fitted scale vector to unchanged frozen "
        "models. Compare to archived identical checkpoint and guard at original normalization. Report by fixed window.",
        {"fit_days":list(FIT_DAYS),"prefix_identity":prefix, "prefix_days":list(DAYS),
         "scale_rule":"95th percentile of each raw request feature and pooled queue residual min, mean and max/2; equal numbers per day and model; floor 1e-6",
         "sampling_rule":"uniformly spaced indices by floor(linspace) within each trace; 2000 request indices per calibration day and model",
         "checkpoint_weights":"frozen; 2019 scales used to produce training-prefix calibration trajectories",
         "expected_calibration_runs":16,"expected_diagnostic_replays":16,"expected_reference_reuses":16,
         "script_sha256":study.digest(Path(__file__))})
    folder = study.BASE/"T7"
    ident = study.lock(folder/"protocol.json", spec)
    cal = []
    measurements = []
    with study.threadpool_limits(limits=1):
        first = traces[0][:3]
        params = diag.diagnostic_args(first,models["oracle",42],False,frozen.bank,diag.blank())
        kernel.compile(tuple(study.typeof(v) for v in params))
        for day in FIT_DAYS:
            inp = traces[day][:3]
            indices = np.linspace(0,len(inp[0])-1,2000).astype(np.int64)
            def one(item):
                mode, s = item
                out = kernel(*diag.diagnostic_args(inp,models[mode,s],False,frozen.bank,diag.blank()))
                sample = raw_features(inp[0],out)
                summary = study.results(inp[0],out[:,:10])[0]
                summary.update(queue_metrics(inp[0],out))
                return (tuple(a[indices] for a in sample[:4])+(sample[4].reshape(len(inp[0]),9)[indices].ravel(),),
                        dict(day=day,seed=s,mode=mode,model_sha256=models[mode,s][2],**summary))
            with ThreadPoolExecutor(max_workers=4) as ex:
                for f,r in ex.map(one,models):
                    cal.append(f)
                    measurements.append(r)
    assert len(cal)==16
    scales = np.array([max(1e-6,float(np.quantile(np.concatenate(column),.95))) for column in zip(*cal)])
    pd.DataFrame(measurements).to_csv(folder/"prefix_trajectories.csv",index=False)
    study.write(folder/"calibration.json",{"protocol":ident,"scales":dict(zip(study.KEYS,scales.tolist())),
                   "prefix_run_count":len(cal),"fit_only_elapsed_0_to_6":True,
                   "fit_preparation_wall_s":time.perf_counter()-started})

    def tasks(key):
        inp = frozen.workload(key)
        return [(key,inp,mode,seed,guard,recal) for (mode,seed),guard,recal in product(models,(False,True),(False,True))]

    def evaluate(task):
        key,inp,mode,seed,guard,recal = task
        ck=models[mode,seed]
        if recal:
            adjusted=(scales,ck[1],ck[2])
            out=kernel(*diag.diagnostic_args(inp,adjusted,guard,frozen.bank,diag.blank()))
            stats=study.results(inp[0],out[:,:10])[0]
            stats.update(queue_metrics(inp[0],out))
        else:
            stats=study.archived_outcome(frozen,key,seed,mode,guard,ck[2])
        return diag.row(frozen,key,seed,mode,ck) | dict(guard=guard,recalibrated=recal,
                    reference_reused=not recal,**stats)

    study.run_shards("T7", spec, keys, tasks, evaluate, lambda:None, 16,16)


if __name__ == "__main__":
    main()
