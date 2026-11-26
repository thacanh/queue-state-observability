"""Replay frozen workloads under crossed queue observations."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import runpy

import replay_io as base
import replay_transfer as ext
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from tqdm import tqdm

from crossed_router import crossed_kernel


OUT = base.ROOT / "outputs/crossed_observations"
KINDS = ["p50", "p99", "ratio", "interarrival", "residual_min", "residual_mean", "residual_max"]
ARMS = [("O", "-"), ("E", "-"), ("O", "O"), ("O", "E"), ("E", "O"), ("E", "E")]
PARTITIONS = (30, 180, 720)
CAPACITY = float(np.dot(base.SPEED, base.WORKERS))


def publish_csv(path, rows):
    temp = path.with_suffix(".tmp")
    pd.DataFrame(rows).to_csv(temp, index=False)
    temp.replace(path)


def describe(x, corpus, day, seed, scale=1., orig_span=None):
    a, demand, deadline = x[:, 0], x[:, 3], x[:, 4]
    span = float(a[-1]-a[0])
    load = float(demand.sum() / (span*CAPACITY))
    np.testing.assert_allclose(load, .6, rtol=0, atol=1e-10)
    bins = np.floor((a-a[0])/60).astype(int)
    rate = len(x)/span
    q = np.quantile(demand, [.5,.95,.99])
    slo = np.quantile(deadline, [.05,.5,.95])
    frame = pd.DataFrame(dict(arrival=a, service_demand=demand, slo=deadline))
    intrinsic = int((demand/max(base.SPEED)>deadline).sum())
    diagnostics = []
    for count in PARTITIONS:
        bounded = bound(mark_bins(frame, bins=count), base.SPEED, base.WORKERS)[0]
        assert bounded["fastest_intrinsic_misses"] == intrinsic
        diagnostics.append(int(bounded["necessary_misses"]))
    return dict(corpus=corpus, day=day, reconstruction_seed=seed, requests=len(x), span_s=span,
        raw_span_s=orig_span, time_multiplier=scale, offered_load=load, arrivals_per_s=rate,
        peak60_arrival_ratio=float(np.bincount(bins).max()/(60*rate)),
        peak60_offered_load=float(np.bincount(bins,weights=demand).max()/(60*CAPACITY)),
        service_mean_s=float(demand.mean()),service_p50_s=q[0],service_p95_s=q[1],service_p99_s=q[2],
        service_max_s=float(demand.max()),slo_p05_s=slo[0],slo_p50_s=slo[1],slo_p95_s=slo[2],
        intrinsic_misses=intrinsic, intrinsic_fraction=intrinsic/len(x),
        pooled_bound_misses=max(diagnostics),pooled_bound_fraction=max(diagnostics)/len(x),
        **{f"bound_bins_{k}":v for k,v in zip(PARTITIONS,diagnostics)})


def queue_metrics(x,out):
    a, demand = x[:,0], x[:,3]
    start = a+out[:,7]
    action=out[:,1].astype(int)
    wait_q=[]
    overloaded=[]
    row={"wait_p95_s":float(np.quantile(out[:,7],.95))}
    for j,(speed,w) in enumerate(zip(base.SPEED,base.WORKERS)):
        at=action==j

        backlog=np.searchsorted(a[at],a,side="right")-np.searchsorted(np.sort(start[at]),a,side="right")
        wait_q.append(backlog)
        end=start[at]+demand[at]/speed
        util=np.maximum(0,np.minimum(end,a[-1])-np.maximum(start[at],a[0])).sum()/((a[-1]-a[0])*w)
        offered=demand[at].sum()/((a[-1]-a[0])*speed*w)
        row.update({f"server_{j+1}_util":util,f"server_{j+1}_routed_load":offered,
                    f"server_{j+1}_backlog_p95":float(np.quantile(backlog,.95))})
        overloaded.append(offered)
    total=np.stack(wait_q).sum(axis=0)
    row.update(backlog_p95=float(np.quantile(total,.95)),backlog_max=int(total.max()),
               max_routed_server_load=max(overloaded),
               saturation_upper=float(out[:,10:17].mean()), saturation_lower=float(out[:,17:24].mean()))
    row.update({f"upper_{name}":float(out[:,i+10].mean()) for i,name in enumerate(KINDS)})
    row.update({f"lower_{name}":float(out[:,i+17].mean()) for i,name in enumerate(KINDS)})
    return row


def verify_old(new, old, digest, model_hash):
    assert old.trace_sha256==digest and old.model_sha256==model_hash
    for k in ("n","violations","overrides","mean_ms","p95_ms","p99_ms","brier"):
        np.testing.assert_allclose(new[k],old[k],rtol=0,atol=1e-6,err_msg=k)


def run(x,ids,pred,seed,obs,guard,bank):
    scales,weights,_=models["fixed",seed]
    result=kernel(x,ids,pred,base.SPEED,base.WORKERS,scales,weights,0,guard!="-",obs=="E",
                  1.,*bank, "E"==(obs if guard=="-" else guard))
    return result


def core(key):
    day,seed=key
    frame,sha=base.read_workload(day,seed,manifest)
    desc=describe(frame[base.XCOLS].to_numpy(), "2019", day, seed)
    if key not in subset:
        return desc,[],[]
    dest=OUT/"shards"/f"{day}_{seed}.json"
    if dest.exists():
        cache=base.load_json(dest)
        if cache["identity"]!=identity: raise ValueError("Crossed-observation replay snapshot differs")
        return desc,cache["factorial"],cache["queue"]
    x,ids,preds=base.workload_arrays(frame)
    results,queues=[],[]
    for model_seed in base.SEEDS:
        for policy,guard in ARMS:
            method = "Fixed-DDQN" if guard=="-" else "Fixed-DDQN+Guard"
            mode = "oracle" if policy=="O" else "estimated"
            if guard=="-" or policy==guard:
                record=old_core.loc[(mode,day,seed,method,model_seed)]
                stat={k:record[k].item() for k in
                      ("n","violations","violation_rate","met_budget","mean_ms","p95_ms","p99_ms","overrides")}
            else:
                record=None

            if record is None or (policy,guard)==("E","-") or (key==first and model_seed==42):
                o=run(x,ids,preds["p50"],model_seed,policy,guard,bank_2019)
                stat=base.results(x,o[:,:10])[0]
                if record is not None:
                    verify_old(stat,record,sha,models["fixed",model_seed][2])
                if (policy,guard)==("E","-"):
                    queues.append(dict(corpus="2019",day=day,reconstruction_seed=seed,seed=model_seed,
                                       **queue_metrics(x,o)))
            keep=("n","violations","violation_rate","met_budget","mean_ms","p95_ms","p99_ms","overrides")
            results.append(dict(day=day,reconstruction_seed=seed,seed=model_seed,policy_obs=policy,
                                guard_obs=guard,trace_sha256=sha,
                                model_sha256=models["fixed",model_seed][2],**{k:stat[k] for k in keep}))
    base.publish_json(dest,{"identity":identity,"factorial":results,"queue":queues})
    return desc,results,queues


def ext_work():
    frame, names, b, offsets, prefix, means, prof, _=ext.prepare(ext.SOURCE)
    bank=(b,offsets,prefix,means)
    workload,queues=[],[]
    for day in range(9,14):
        x,ids,pred,minute=ext.day_workload(frame,day,names,prof)
        keep=frame.loc[(frame.day==day)&frame.id.isin(names)&
                        frame.start.ge(day*ext.DAY+minute*60)&frame.start.lt(day*ext.DAY+(minute+180)*60)]
        orig=float(keep.start.max()-keep.start.min())
        assert len(keep)==len(x)
        np.testing.assert_array_equal(np.sort(keep.duration.to_numpy()),np.sort(x[:,3]))
        ratio=(x[-1,0]-x[0,0])/orig
        np.testing.assert_allclose((np.sort(keep.start.to_numpy())-keep.start.min())*ratio,x[:,0])
        sha=hashlib.sha256(x.tobytes()+ids.tobytes()).hexdigest()
        workload.append(describe(x,"2021",str(day),-1,ratio,orig))
        for seed in base.SEEDS:
            out=run(x,ids,pred["p50"],seed,"E","-",bank)
            row=old_ext.loc[("estimated",day,"Fixed-DDQN",seed)]
            verify_old(base.results(x,out[:,:10])[0],row,sha,models["fixed",seed][2])
            queues.append(dict(corpus="2021",day=str(day),reconstruction_seed=-1,seed=seed,
                               **queue_metrics(x,out)))
    return workload,queues


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument("--workers",type=int,default=6)
    args=parser.parse_args(argv)
    OUT.mkdir(exist_ok=True)
    (OUT/"shards").mkdir(exist_ok=True)
    global subset,manifest,identity,models,kernel,bank_2019,old_core,old_ext,bound,mark_bins,first
    prior=base.load_json(base.OUT/"status.json")
    external=base.load_json(ext.EXT/"status.json")
    assert prior["paired_sha256"]==base.digest(base.OUT/"paired.csv")
    assert external["paired_sha256"]==base.digest(ext.EXT/"paired_2021.csv")
    transfer_protocol=base.load_json(ext.EXT/"protocol.json")
    assert transfer_protocol["archive_sha256"]==base.digest(ext.ARCHIVE) and transfer_protocol["member_sha256"]==base.digest(ext.SOURCE)
    plan=[tuple(k) for k in base.load_json(base.BANK_PATH/"protocol.json")["plan"]]
    subset={k for d in ("d08","d09") for k in [k for k in plan if k[0]==d][:5]}
    first=next(iter(plan))
    manifest=base.load_json(base.BANK_PATH/"inputs/manifest.json")
    protocol={"stage":"Crossed observation replay with frozen checkpoints","data":{"2019":prior["identity"],"2021":external["identity"]},
        "factorial_inputs":[list(k) for k in plan if k in subset],"armlist":ARMS,
        "queue_telemetry":"Estimated-mode frozen Fixed-DDQN, all five seeds, 10 R0 draws and five 2021 days",
        "saturation":"share of clipped input features exactly at boundary; per-server residual features averaged over servers",
        "feasibility":"intrinsic fastest-server floor plus max of valid disjoint pooled capacity relaxations",
        "pooled_partitions":PARTITIONS,"training":"Frozen-checkpoint evaluation.",
        "files":{f:base.digest(base.ROOT/"work"/f) for f in
            ("routing_simulator.py","crossed_router.py","replay_crossed.py","risk_feasibility.py","replay_transfer.py")}}
    identity=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    lock=OUT/"protocol.json"
    if lock.exists() and base.load_json(lock)["identity"]!=identity: raise ValueError("Crossed-observation protocol changed")
    base.publish_json(lock,{"identity":identity,**protocol})
    scope=runpy.run_path(str(base.ROOT/"work/risk_feasibility.py"),
                         init_globals={"np":np,"pd":pd,"njit":base.njit})
    bound,mark_bins=scope["risk_deadline_necessary_bound"],scope["risk_mark_bins"]
    base.setup()
    _,b,o,p,m,*_=base.CONTEXT
    bank_2019=(b,o,p,m)
    models=base.CONTEXT[7]
    old_core=pd.read_csv(base.OUT/"paired.csv").set_index(["observation","day","reconstruction_seed","method","seed"])
    old_ext=pd.read_csv(ext.EXT/"paired_2021.csv").set_index(["observation","day","method","seed"])
    kernel=crossed_kernel()
    with threadpool_limits(limits=1):

        work,res,queues=core(plan[0])
        workloads=[work]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for work,rows,q in tqdm(pool.map(core,plan[1:]),total=len(plan)-1,desc="Azure 2019 crossed-observation replay"):
                workloads.append(work)
                res.extend(rows)
                queues.extend(q)
        w,q=ext_work()
    workloads.extend(w)
    queues.extend(q)
    assert len(workloads)==64 and len(res)==300 and len(queues)==75
    publish_csv(OUT/"workload_by_trace.csv",workloads)
    publish_csv(OUT/"factorial_pairs.csv",res)
    publish_csv(OUT/"queue_by_trace_seed.csv",queues)
    base.publish_json(OUT/"status.json",{"status":"VERIFIED_CROSSED_OBSERVATIONS","identity":identity,
        "rows":{"workload":len(workloads),"factorial":len(res),"telemetry":len(queues)},
        "sha256":{f:base.digest(OUT/f) for f in ("workload_by_trace.csv","factorial_pairs.csv","queue_by_trace_seed.csv")}})


if __name__=="__main__":
    main()
