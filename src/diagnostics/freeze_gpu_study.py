"""Freeze GPU study input identities and planned comparisons before training."""

from itertools import product

import replay_study as study


def main():
    prior=study.read(study.TEMPORAL/"protocol.json")
    later=study.read(study.TEMPORAL/"inputs/2019_R0/manifest.json")["traces"]
    earlier={f"{r['day']}_{r['reconstruction_seed']}":r["trace_sha256"]
             for r in study.csv_rows(study.HISTORICAL)}
    assert len(earlier)==59 and len(later)==30
    train=study.ROOT/"work/estimated-local/multiday_final_5seeds"
    registry=study.read(study.ROOT/"outputs/verified_final/e2/registry.json")["jobs"]
    seeds=(42,43)
    records={s:registry[f"fixed_seed{s}_lambda2"]["record"]["input_sha256"] for s in seeds}
    prefix=study.read(study.BASE/"prefix_inputs/manifest.json")
    cap={"T4":[1800,900],"T8":[900,600]}
    common={"states_per_job":2_000_128,"updates_per_job":7695,"guard_in_training":False,
            "lambda":2,"seeds":list(seeds),"same_device_per_phase":True,"same_save_cadence_batches":1000,
            "target_update_batches":500,"source_sha256":study.digest(study.BASE/"code/gpu_training.py"),
            "replay_source_sha256":study.digest(study.BASE/"code/evaluate_gpu.py"),
            "deadline_budget":.005,"evaluation_guard":[False,True],"evaluation_observation":"estimated",
            "budget_s":"first GPU active training, second CPU evaluation",
            "safe_stop":"record phase elapsed time, stop at the cap, preserve atomic checkpoints",
            "analysis":"all paired seeds and days; report miss fraction, workload P95/P99, attainment and guard "
                       "override; intervals are conditional, with no cross-pair independence or calendar equivalence claim"}
    spec4=common|{"phase":"T4","budget_limits_s":cap["T4"],"jobs":[list(i) for i in product(seeds,study.MODES)],
        "training":{str(seed):row for seed,row in records.items()},
        "workloads":earlier|{name:r["trace_sha256"] for name,r in later.items()},
        "replays_new":712,"historical_comparators":{n:v["sha256"] for n,v in prior["models"].items() if int(n[-2:]) in seeds},
        "contrasts":["Estimated Numba − Oracle Numba","Oracle Numba − Oracle Torch",
                     "New Estimated Numba − archived Estimated Numba"]}
    reserved=study.read(study.TEMPORAL/"inputs/2021_reserved/manifest.json")["traces"]
    spec8=common|{"phase":"T8","budget_limits_s":cap["T8"],"training_prefix_only":prefix,
        "fit_scales_sha256":study.digest(study.BASE/"T7/calibration.json"),
        "jobs":[list(i) for i in product(seeds,("estimated",))],
        "initialization":{str(s):prior["models"][f"estimated/{s}"]["sha256"] for s in seeds},
        "workloads":{name:r["trace_sha256"] for name,r in reserved.items()},
        "replays_new":8,"replays_reused":8,"reference_protocol":study.read(study.BASE/"T7/protocol.json")["protocol_sha256"],
        "scope":"fixed diagnostic windows with prior outcomes; same normalization for learned and frozen weights"}
    for phase,spec in (("T4",spec4),("T8",spec8)):
        study.lock(study.BASE/phase/"planned_protocol.json",spec)
    for seed,record in records.items():
        assert study.digest(train/"models"/f"latency_ddqn_seed{seed}.pt")==record["stage1"]
        assert study.digest(train/"traces/train_d01_d07.csv")==record["train"]
        assert study.digest(train/"traces/val_d08_d09.csv")==record["validation"]


if __name__=="__main__":
    main()
