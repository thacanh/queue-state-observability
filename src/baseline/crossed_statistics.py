"""Summarize crossed_observations and verify all contrasts use paired units."""

import pandas as pd
import numpy as np

import replay_io as base
from guard_statistics import grouped_differences

OUT=base.ROOT/"outputs/crossed_observations"
KEYS=["day","reconstruction_seed","seed","trace_sha256","model_sha256"]
METRICS=["met_budget","mean_ms","p95_ms","p99_ms","violation_rate"]


def paired(a,b):
    joined=a.merge(b,on=KEYS,suffixes=("_candidate","_comparator"),validate="one_to_one")
    assert len(a)==len(b)==len(joined)
    return joined


def effect(rows,group):
    summary=[]
    for key,g in rows.groupby(group):
        a,b=g.met_budget_candidate,g.met_budget_comparator
        summary.append(dict(zip(group,key if isinstance(key,tuple) else (key,))) | dict(
            pairs=len(g), candidate_pass=int(a.sum()), comparator_pass=int(b.sum()),
            fail_to_pass=int((a&~b).sum()),pass_to_fail=int((~a&b).sum()),
            miss_delta_pp=100*float((g.violation_rate_candidate-g.violation_rate_comparator).mean()),
            p95_delta_ms=float((g.p95_ms_candidate-g.p95_ms_comparator).mean())))
    return pd.DataFrame(summary)


def main():
    status=base.load_json(OUT/"status.json")
    assert all(base.digest(OUT/name)==sha for name,sha in status["sha256"].items())
    work=pd.read_csv(OUT/"workload_by_trace.csv")
    queue=pd.read_csv(OUT/"queue_by_trace_seed.csv")
    fact=pd.read_csv(OUT/"factorial_pairs.csv",keep_default_na=False)
    assert set(work.corpus)=={2019,2021} and set(fact.policy_obs)=={"O","E"}
    assert len(work)==64 and len(queue)==75 and len(fact)==300
    w=work.groupby("corpus").mean(numeric_only=True)
    w["service_max_s"]=work.groupby("corpus").service_max_s.max()
    q=queue.groupby("corpus").mean(numeric_only=True)
    combined=w.join(q.drop(columns=["reconstruction_seed","seed"],errors="ignore"),rsuffix="_telemetry")
    combined.to_csv(OUT/"workload_summary.csv")
    work.query("corpus == 2021").to_csv(OUT/"feasibility_2021_by_day.csv",index=False)
    fact_summary=fact.groupby(["policy_obs","guard_obs"]).agg(
        passes=("met_budget","sum"), pairs=("met_budget","size"), mean_ms=("mean_ms","mean"),
        p95_ms=("p95_ms","mean"),p99_ms=("p99_ms","mean"),miss_rate=("violation_rate","mean"),
        overrides=("overrides","sum"))
    assert fact_summary.pairs.eq(50).all()
    fact_summary.to_csv(OUT/"factorial_summary.csv")
    blocks={k:g for k,g in fact.groupby(["policy_obs","guard_obs"])}
    specs=[("base shift, no guard",("E","-"),("O","-")),
           ("base shift, oracle guard",("E","O"),("O","O")),
           ("base shift, estimated guard",("E","E"),("O","E")),
           ("guard shift, oracle policy",("O","E"),("O","O")),
           ("guard shift, estimated policy",("E","E"),("E","O")),
           ("both shift, guarded",("E","E"),("O","O"))]
    contrasts=[]
    for label,a,b in specs:
        r=paired(blocks[a],blocks[b])
        contrasts.extend({"comparison":label,**v} for v in grouped_differences(r))
    oo,oe,eo,ee=(blocks[k].set_index(KEYS).sort_index() for k in [("O","O"),("O","E"),("E","O"),("E","E")])
    assert oo.index.equals(oe.index) and oo.index.equals(eo.index) and oo.index.equals(ee.index)
    interaction=oo[[]].reset_index()
    for k in METRICS:
        interaction[k+"_candidate"]=(ee[k].astype(float)-eo[k].astype(float)
                                      -oe[k].astype(float)+oo[k].astype(float)).to_numpy()
        interaction[k+"_comparator"]=0.
    contrasts.extend({"comparison":"observation interaction",**v} for v in grouped_differences(interaction))
    pd.DataFrame(contrasts).to_csv(OUT/"factorial_contrasts.csv",index=False)
    old=pd.read_csv(base.OUT/"paired.csv")
    changes=[]
    shifts=[]
    guard_intervals=[]
    for mode in ("oracle","estimated"):
        arms=old[old.observation.eq(mode)]
        r=paired(arms[arms.method.eq("Fixed-DDQN+Guard")],arms[arms.method.eq("Fixed-DDQN")])
        guard_intervals.extend(dict(observation=mode,baseline="Fixed-DDQN",**v)
                               for v in grouped_differences(r))
        for keys in (["day"],["seed"],["day","seed"]):
            changes.append(effect(r,keys).assign(observation=mode,grouping="/".join(keys)))
    for m in ("Fixed-DDQN","Fixed-DDQN+Guard"):
        q=old[old.method.eq(m)]
        r=paired(q[q.observation.eq("estimated")],q[q.observation.eq("oracle")])
        for keys in (["day"],["seed"],["day","seed"]):
            shifts.append(effect(r,keys).assign(method=m,grouping="/".join(keys)))
    pd.concat(changes,ignore_index=True).to_csv(OUT/"guard_paired_by_seed_day.csv",index=False)
    pd.concat(shifts,ignore_index=True).to_csv(OUT/"observation_shift_by_seed_day.csv",index=False)
    prior_intervals=pd.read_csv(base.OUT/"paired_conditional_intervals.csv").query('baseline=="Fixed-DDQN"')
    intervals=pd.DataFrame(guard_intervals)
    pd.testing.assert_frame_equal(intervals,prior_intervals.reset_index(drop=True))
    intervals.to_csv(OUT/"guard_paired_intervals.csv",index=False)
    ext=pd.read_csv(base.OUT/"external/paired_2021.csv")
    assert len(ext)==170 and not ext.met_budget.any()
    ext[["day","seed","observation","method","n","violations","violation_rate","met_budget","mean_ms",
         "p95_ms","p99_ms","trace_sha256","model_sha256"]].sort_values(
             ["day","observation","method","seed"]).to_csv(OUT/"transfer_2021_by_day_seed.csv",index=False)
    matched_training=pd.read_csv(base.BASELINE)
    controls=pd.read_csv(base.OUT/"diagnostics/checkpoint_controls.csv")
    lat=matched_training[matched_training.method.eq("latency")]
    ref=controls[controls.method.eq("Latency-4M")&controls.observation.eq("oracle")]
    assert set(lat.reconstruction_seed).isdisjoint(ref.reconstruction_seed)
    assert lat.groupby("seed").model_sha256.first().equals(ref.groupby("seed").model_sha256.first())
    assert int((lat.violation_rate<=.005).sum())==13 and int(ref.met_budget.sum())==9
    adaptive=matched_training[matched_training.method.eq("adaptive")].assign(passed=lambda x:x.violation_rate<=.005)
    adaptive_passes=adaptive.groupby("seed").passed.sum().tolist()
    assert adaptive_passes==[9,7,8,1,9]
    audit={"matched_training_latency_P99_SD_ms":float(lat.groupby("seed").p99_ms.mean().std(ddof=1)),
            "SD_rule":"sample SD across five checkpoint seed means, each averaged over ten reconstructions",
            "matched_training_adaptive_passes_by_seed_42_to_46":adaptive_passes,
            "matched_training_passes":13,"guard_bank_oracle_latency_passes":9,
            "comparison":"same five model hashes, disjoint reconstruction seeds, same two source days",
            "matched_training_reconstruction_seeds":sorted(lat.reconstruction_seed.unique().tolist()),
            "guard_bank_reconstruction_seeds":sorted(ref.reconstruction_seed.unique().tolist())}
    base.publish_json(OUT/"numeric_audit.json",audit)
    print(combined[["arrivals_per_s","peak60_arrival_ratio","peak60_offered_load","service_mean_s","service_p99_s",
                    "service_max_s","slo_p50_s","intrinsic_fraction","pooled_bound_fraction","backlog_p95","saturation_upper",
                    "max_routed_server_load"]].to_string())
    print(work.query("corpus == 2021")[["day","intrinsic_fraction","pooled_bound_fraction","time_multiplier",
                                     "peak60_offered_load"]].to_string(index=False))
    print(fact_summary.to_string())


if __name__=="__main__":
    main()
