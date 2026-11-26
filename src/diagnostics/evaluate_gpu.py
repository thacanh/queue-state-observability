"""Evaluate predeclared GPU-study checkpoints once validated results are returned."""

from argparse import ArgumentParser
from itertools import product
from pathlib import Path
from shutil import copyfile

import replay_study as study


PENALTIES = (0,5,20,80)
SEEDS = (42,43)
HISTORY = study.ROOT/"outputs/shield_verified/source_data/runs/deadline_guard_v1/inputs"
REGISTRY = study.ROOT/"outputs/verified_final/e2/registry.json"


def audit_t2():
    registry=study.read(REGISTRY)["jobs"]
    local=study.ROOT/"work/estimated-local/multiday_final_5seeds"
    output=[]
    for seed,lam in product(SEEDS,PENALTIES):
        rec=registry[f"fixed_seed{seed}_lambda{lam}"]["record"]
        primary=registry[f"fixed_seed{seed}_lambda2"]["record"]
        assert rec["seed"]==seed and rec["method"]=="fixed" and rec["fixed_lambda"]==lam
        assert rec["realized_transitions"]==2_000_128 and rec["actual_optimizer_updates"]==7695
        assert rec["input_sha256"]==primary["input_sha256"] and rec["source_sha256"]==primary["source_sha256"]
        stage1=local/"models"/f"latency_ddqn_seed{seed}.pt"
        assert study.digest(stage1)==rec["input_sha256"]["stage1"]
        assert study.digest(local/"traces/train_d01_d07.csv")==rec["input_sha256"]["train"]
        assert study.digest(local/"traces/val_d08_d09.csv")==rec["input_sha256"]["validation"]
        output.append(dict(seed=seed, penalty=lam, training_observation="oracle",
                      expected_binary_sha256=rec["model_sha256"], stage1_sha256=rec["input_sha256"]["stage1"],
                      source_sha256=rec["source_sha256"], recorded_transitions=rec["realized_transitions"],
                      recorded_updates=rec["actual_optimizer_updates"], gpu_active_session_s=rec["session_train_seconds"],
                      recovered_local_files=[str(p.relative_to(study.ROOT)) for p in (study.ROOT/"outputs").rglob(
                      Path(rec["model_path"]).parent.name+"/final.pt") if study.digest(p)==rec["model_sha256"]],
                      project_relative_original_path=rec["model_path"]))
    study.write(study.BASE/"T2/checkpoint_audit.json",{"checkpoints":output,"registry_sha256":study.digest(REGISTRY)})


def stage_recovered_t2():
    registry=study.read(REGISTRY)["jobs"]
    rows=[]
    for seed,lam in product(SEEDS,PENALTIES):
        rec=registry[f"fixed_seed{seed}_lambda{lam}"]["record"]
        source=study.BASE/"recovered_t2"/Path(rec["model_path"]).parent.name
        manifest=study.read(source/"training.json")
        assert manifest==rec and study.digest(source/"final.pt")==rec["model_sha256"]
        ck=study.torch.load(source/"final.pt",map_location="cpu",weights_only=True)
        fields={"seed":seed,"method":"fixed","fixed_lambda":lam,"transitions":2_000_000}
        assert ck["identity"]==rec["identity"] and all(ck["config"][k]==v for k,v in fields.items())
        assert ck["config"]["device"]=="cuda" and ck["meta"]["updates"]==7695
        assert ck["meta"]["steps"]*ck["config"]["nenv"]==2_000_128
        folder=study.BASE/"gpu_results"/"T2"/f"oracle_{seed}_lambda{lam}"
        folder.mkdir(parents=True,exist_ok=True)
        target=folder/"final.pt"
        if not target.exists():
            copyfile(source/"final.pt",target.with_suffix(".tmp"))
            target.with_suffix(".tmp").replace(target)
        assert study.digest(target)==rec["model_sha256"]
        row={"status":"COMPLETE","origin":"recovered historical checkpoint","seed":seed,"lambda":lam,
            "model_sha256":rec["model_sha256"],"identity":rec["identity"],"observation":"oracle",
            "metadata_sha256":study.digest(source/"training.json"),"registry_sha256":study.digest(REGISTRY),
            "provenance":study.read(source/"provenance.json"),"history":rec}
        study.write(folder/"status.json",row)
        rows.append(row)
    study.lock(study.BASE/"T2/recovery_manifest.json",{"sources":rows,"new_training_jobs":0})


def checkpoint(folder, phase, mode, seed, lam):
    directory=folder/phase/f"{mode}_{seed}_lambda{lam}"
    status=study.read(directory/"status.json")
    assert status["status"]=="COMPLETE" and status["seed"]==seed and status["lambda"]==lam
    recovered=status["origin"]=="recovered historical checkpoint"
    if recovered:
        record=study.read(REGISTRY)["jobs"][f"fixed_seed{seed}_lambda{lam}"]["record"]
        assert phase=="T2" and status["model_sha256"]==record["model_sha256"]
    else:
        assert status["actual_transitions"]==2_000_128 and status["optimizer_updates"]==7695
        assert status["observation"]==mode and not status["guard_in_training"]
        if phase in ("T2","T4"):
            rec=study.read(REGISTRY)["jobs"][f"fixed_seed{seed}_lambda2"]["record"]
            assert status["initial_checkpoint_sha256"]==rec["input_sha256"]["stage1"]
            assert status["input_identity"]==rec["input_sha256"]
        if phase=="T4":
            assert status["calendar"]=="shared Numba float32"
    return study.model(directory/"final.pt",status["model_sha256"],status["identity"]),status


def earlier_manifest(frozen):
    index=study.read(HISTORY/"manifest.json")
    if "traces" in index:
        index=index["traces"]
    earlier=study.csv_rows(study.HISTORICAL)
    assert len(earlier)==1180
    keys=sorted({f"{x['day']}_{x['reconstruction_seed']}" for x in earlier})
    expected={f"{x['day']}_{x['reconstruction_seed']}":x["trace_sha256"] for x in earlier}
    for k in keys:
        assert expected[k]==index[k]["frame_sha256"]
        frozen.traces[k]={"trace_sha256":expected[k],"file_sha256":index[k]["sha256"],
                          "day":k.split("_")[0]}
    old={((f"{r['day']}_{r['reconstruction_seed']}"),r["policy_training"],int(r["seed"]),r["guard"]=="True"):r
         for r in earlier}
    return index,keys,old


def t4_workload(frozen, key, index):
    if key not in index:
        return frozen.workload(key)
    rec=index[key]
    path=HISTORY/rec["filename"]
    assert study.digest(path)==rec["sha256"]
    frame=study.pd.read_csv(path,dtype=rec["dtypes"],float_precision="round_trip")
    fingerprint=study.sha256(study.pd.util.hash_pandas_object(frame,index=True).values.tobytes()).hexdigest()
    assert fingerprint==rec["frame_sha256"]
    x=study.np.ascontiguousarray(frame[study.XCOLS].to_numpy(study.np.float64))
    ids=frame.func_id.astype(str).map(frozen.names).to_numpy(study.np.int64)
    return x,ids,x[:,1].copy()


def evaluate_training(phase, folder):
    frozen=study.Frozen(transfer=phase=="T8")
    index,hist_keys,old={},[],{}
    if phase=="T4":
        index,hist_keys,old=earlier_manifest(frozen)
    keys=frozen.window_keys if phase=="T8" else hist_keys+list(frozen.traces.keys())[:30] if phase=="T4" else frozen.later_keys
    if phase=="T4":
        assert len(keys)==len(set(keys))==89
    instructions=(list(product(("oracle",),SEEDS,PENALTIES)) if phase=="T2" else
                  list(product(study.MODES if phase=="T4" else ("estimated",),SEEDS,(2,))))
    statuses, models={},{}
    for mode,seed,lam in instructions:
        label=f"{mode}_lambda{lam}"
        models[label,seed], statuses[f"{label}/{seed}"]=checkpoint(folder,phase,mode,seed,lam)
    if phase=="T4":
        assert len({study.fingerprint(s["runtime"]) for s in statuses.values()})==1
        assert len({s["protocol"] for s in statuses.values()})==1
        for seed in SEEDS:
            a,b=(statuses[f"{m}_lambda2/{seed}"] for m in study.MODES)
            assert a["config"]==b["config"] and a["input_identity"]==b["input_identity"]
    refs={}
    for mode,seed in product(study.MODES,SEEDS):
        refs[mode,seed]=frozen.checkpoint(mode,seed)
    calib=None
    if phase=="T8":
        calib=study.pd.read_csv(study.BASE/"T7/replays.csv")
        calib=calib[calib.recalibrated & calib["mode"].eq("estimated")]
        assert len(calib)==8
    run_count={"T2":96,"T4":712,"T8":8}[phase]
    reused={"T2":48,"T4":712,"T8":8}[phase]
    spec=study.base_protocol(phase,frozen,keys,SEEDS,models,
        "Report within-day paired miss, mean workload P95/P99, and attainment. Keep original/later and transfer "
        "trace scope separate. T4 contrasts both symmetric controls and both matched historical same-mode models.",
        {"run_count":run_count,"archived_reference_count":reused,"inference_guard":[False,True],
         "training_manifests":statuses,"evaluation_code":study.digest(__file__),
         "reference_bank":study.digest(study.HISTORICAL) if phase=="T4" else None,
         "t8_normalization":study.digest(study.BASE/"T7/calibration.json") if phase=="T8" else None})

    def tasks(key):
        inp=t4_workload(frozen,key,index) if phase=="T4" else frozen.workload(key)
        new=[(key,inp,mode,seed,guard,False) for (mode,seed),guard in product(models,(False,True))]
        reference=[(key,inp,mode,seed,guard,True) for (mode,seed),guard in product(refs,(False,True))
                   if phase!="T8" or mode=="estimated"]
        return new+reference

    def evaluate(task):
        key,inp,mode,seed,guard,reference=task
        if not reference:
            ck=models[mode,seed]
            stats=study.results(inp[0],study.replay(*study.arguments(inp,ck,guard,frozen.bank)))[0]
        else:
            ck=refs[mode,seed]
            if phase=="T8":
                m=calib[(calib.key==key)&(calib.seed==seed)&(calib.guard==guard)]
                assert len(m)==1 and m.model_sha256.iloc[0]==ck[2]
                stats=m.iloc[0].drop(labels=["key","day","seed","mode","model_sha256","trace_sha256",
                      "guard","recalibrated","reference_reused"]).to_dict()
            elif key in hist_keys:
                a=old[key,"estimated_fine_tuning" if mode=="estimated" else mode,seed,guard]
                assert a["trace_sha256"]==frozen.traces[key]["trace_sha256"] and a["model_sha256"]==ck[2]
                stats={k:float(a[k]) for k in ("violation_rate","mean_ms","p95_ms","p99_ms","override_rate")}
                stats.update(n=int(a["n"]),violations=int(a["violations"]),met_budget=a["met_budget"]=="True")
            else:
                stats=study.archived_outcome(frozen,key,seed,mode,guard,ck[2])
        return dict(day=frozen.traces[key]["day"],key=key,seed=seed,mode=mode,guard=guard,
                    trace_sha256=frozen.traces[key]["trace_sha256"],model_sha256=ck[2],reference_reused=reference,**stats)

    def warm():
        _,inp,mode,seed,guard,_=tasks(keys[0])[0]
        params=study.arguments(inp,models[mode,seed],guard,frozen.bank)
        study.replay.compile(tuple(study.typeof(p) for p in params))

    study.run_shards(phase,spec,keys,tasks,evaluate,warm,run_count,reused)


if __name__=="__main__":
    cli=ArgumentParser(description=__doc__)
    cli.add_argument("task",choices=["audit","recover","T2","T4","T8"])
    cli.add_argument("--gpu-results",type=Path,default=study.BASE/"gpu_results")
    args=cli.parse_args()
    if args.task=="audit":
        audit_t2()
    else:
        study.dependencies()
        if args.task=="recover":
            stage_recovered_t2()
        else:
            evaluate_training(args.task,args.gpu_results)
