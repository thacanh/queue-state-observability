"""Reconstruct the A1–A4 evidence from archived rows, with no routing calls."""

from pathlib import Path
import os
import time

import replay_study as study

os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil

OUT = study.BASE / "archived_analysis"
SEEDS = [42,43,44,45,46]
ARMS = ["oracle_off", "oracle_on", "estimated_off", "estimated_on"]
LABELS = dict(zip(ARMS, ["Oracle FT", "Oracle FT + guard", "Estimated FT", "Estimated FT + guard"]))
COLORS = dict(zip(ARMS, ["#586a7e", "#0e7490", "#ad5a3a", "#842d24"]))
BUDGETS = np.array([30,40,50,60,75,100])
CONTRASTS = (
    ("Guard after Oracle FT", "oracle_on", "oracle_off"),
    ("Guard after Estimated FT", "estimated_on", "estimated_off"),
    ("FT with guard", "estimated_on", "oracle_on"),
    ("FT without guard", "estimated_off", "oracle_off"),
)
plt.rcParams.update({"font.size":9, "axes.spines.top":False, "axes.spines.right":False,
                     "legend.frameon":False, "pdf.fonttype":42})


def save(data,name):
    data.to_csv(OUT/f"{name}.csv",index=False)


def write_fig(fig,name):
    fig.tight_layout()
    fig.savefig(OUT/f"{name}.pdf",bbox_inches="tight")
    fig.savefig(OUT/f"{name}.png",dpi=160,bbox_inches="tight")
    plt.close(fig)


def read(rel):
    return pd.read_csv(study.ROOT/"outputs"/rel)


def fine_tuning_banks():
    historical=read("estimated_ddqn/validation/paired.csv")
    historical["arm"]=historical.policy_training.map({"oracle":"oracle","estimated_fine_tuning":"estimated"})+"_"+historical.guard.map({True:"on",False:"off"})
    historical["key"]=historical.day+"_"+historical.reconstruction_seed.astype(str)
    later=read("temporal_evaluation/B/replays.csv")
    assert (len(historical),len(later))==(1180,600)
    for f in (historical,later):
        assert (f.violations/f.n<=.005).eq(f.met_budget).all()
        assert f.groupby(["key","seed"]).size().eq(4).all()
        assert f.groupby("key").trace_sha256.nunique().eq(1).all()
        assert f.groupby("key").n.nunique().eq(1).all()
    return {"d08–d09":historical,"d12–d14":later}


def paired_intervals(wide):
    groups=[x.droplevel("day").sort_index().to_numpy().reshape(5,-1,len(BUDGETS))
            for _,x in wide.groupby(level="day")]
    rand=np.random.default_rng(1127)
    simulations=np.empty((4000,len(BUDGETS)))
    for i in range(len(simulations)):
        days=rand.integers(len(groups),size=len(groups))
        seeds=rand.integers(5,size=5)
        pieces=[]
        for d in days:
            group=groups[d]
            draw=rand.integers(group.shape[1],size=group.shape[1])
            pieces.append(group[seeds[:,None],draw].reshape(-1,len(BUDGETS)))
        simulations[i]=np.concatenate(pieces).mean(axis=0)*100
    return np.quantile(simulations,[.025,.975],axis=0)


def budget_analysis(banks):
    counts,changes,points,intervals=[],[],[],[]
    fig,ax=plt.subplots(1,2,figsize=(8.4,3.25),sharey=True)
    for axis,(bank,frame) in zip(ax,banks.items()):
        for day,subset in [("all",frame),*frame.groupby("day")]:
            for arm,m in subset.groupby("arm"):
                ratios=np.sort(m.violations.to_numpy()/m.n.to_numpy()*100)
                checkpoints,frequency=np.unique(ratios,return_counts=True)
                points.extend(dict(bank=bank,day=day,arm=arm,miss_pct=rate,
                                   attainment=win/len(m),n=len(m)) for rate,win in zip(checkpoints,np.cumsum(frequency)))
                if day=="all":
                    view=np.r_[.3,checkpoints[(checkpoints>.3)&(checkpoints<1.)],1.]
                    vals=np.searchsorted(ratios,view,side="right")/len(m)*100
                    axis.step(view,vals,where="post",color=COLORS[arm],label=LABELS[arm],
                              linestyle="-" if arm.endswith("on") else "--")
                for b in BUDGETS:
                    hit=m.violations*10000<=m.n*b
                    counts.append(dict(bank=bank,day=day,arm=arm,budget_pct=b/100,n=len(m),
                                       passes=int(hit.sum()),attainment=float(hit.mean())))
            for name,a,b in CONTRASTS:
                k=["day","seed","key","trace_sha256","n"]
                left=subset[subset.arm.eq(a)][k+["violations"]]
                right=subset[subset.arm.eq(b)][k+["violations"]]
                paired=left.merge(right,on=k,validate="one_to_one",suffixes=("_candidate","_reference"))
                assert len(paired)*4==len(subset)
                hits_c=paired.violations_candidate.to_numpy()[:,None]*10000<=paired.n.to_numpy()[:,None]*BUDGETS
                hits_r=paired.violations_reference.to_numpy()[:,None]*10000<=paired.n.to_numpy()[:,None]*BUDGETS
                delta=hits_c.astype(int)-hits_r.astype(int)
                for j,th in enumerate(BUDGETS):
                    changes.append(dict(bank=bank,day=day,contrast=name,budget_pct=th/100,n=len(paired),
                                        fail_to_pass=int((delta[:,j]>0).sum()),pass_to_fail=int((delta[:,j]<0).sum()),
                                        both_pass=int((hits_c[:,j]&hits_r[:,j]).sum()),
                                        both_fail=int((~hits_c[:,j]&~hits_r[:,j]).sum()),
                                        change_pp=float(delta[:,j].mean()*100)))
                if day=="all":
                    wide=pd.DataFrame(delta,index=pd.MultiIndex.from_frame(paired[["day","seed","key"]]))
                    bounds=paired_intervals(wide)
                    for t,lo,hi,estimate in zip(BUDGETS,*bounds,delta.mean(axis=0)*100):
                        intervals.append(dict(bank=bank,contrast=name,budget_pct=t/100,estimate_pp=estimate,low_pp=lo,high_pp=hi))
        axis.set(xlabel="Workload miss budget (%)",title=bank,xlim=(.30,1.00),ylim=(0,103))
        axis.axvline(.5,color=".55",linestyle=":",linewidth=1)
        axis.grid(axis="y",alpha=.2)
    ax[0].set_ylabel("Workload attainment (%)")
    ax[0].legend(loc="lower right",fontsize=7)
    save(pd.DataFrame(points),"a1_exact_ecdf")
    save(pd.DataFrame(counts),"a1_attainment_by_day")
    save(pd.DataFrame(changes),"a1_paired_transitions")
    save(pd.DataFrame(intervals),"a1_conditional_intervals")
    write_fig(fig,"a1_budget_sensitivity")


def points(frame, grouping):
    return (frame.groupby(grouping,dropna=False,as_index=False)
            .agg(replays=("violation_rate","size"),miss_pct=("violation_rate",lambda s:s.mean()*100),
                 mean_ms=("mean_ms","mean"),p95_ms=("p95_ms","mean"),p99_ms=("p99_ms","mean"),
                 passes=("met_budget","sum")))


def tradeoffs(banks):
    frames=[]
    fig,axes=plt.subplots(2,2,figsize=(8.5,6.0))
    for axis,(bank,frame) in zip(axes[0],banks.items()):
        sums=points(frame,["arm"])
        frames.append(sums.assign(bank=bank))
        for item in sums.itertuples():
            axis.scatter(item.p95_ms,item.miss_pct,color=COLORS[item.arm],label=LABELS[item.arm],s=26)
        axis.set_title(bank+" · estimated evaluation")
    e2=read("verified_final/e2/pilot_paired.csv")
    e2=e2[e2.method.eq("fixed")].copy()
    assert len(e2)==180
    e2["met_budget"]=e2.violations*200<=e2.n
    sums=points(e2,["fixed_lambda"])
    for item in sums.itertuples():
        axes[1,0].scatter(item.p95_ms,item.miss_pct,color="#586a7e",s=24)
        axes[1,0].annotate(f"λ={item.fixed_lambda:g}",(item.p95_ms,item.miss_pct),
                           xytext=(3,2),textcoords="offset points",fontsize=7)
    axes[1,0].set_title("Oracle evaluation · fixed-penalty pilot")
    save(sums,"a2_oracle_penalty_pilot")
    grid=read("e5/threshold_bias_summary.csv")
    grid=grid[grid.method.eq("Fixed-DDQN+Guard")&grid.bias.eq(1.0)].copy()
    assert len(grid)==9
    for item in grid.itertuples():
        axes[1,1].scatter(item.p95_ms,item.violation_rate*100,color="#0e7490",s=24)
    axes[1,1].set_title("Estimated evaluation · guard grid, seed 42")
    axes[0,0].legend(fontsize=6.9)
    for axis in axes.flat:
        axis.set(xlabel="Mean workload P95 (ms)",ylabel="Mean workload miss (%)")
        axis.grid(alpha=.2)
    save(pd.concat(frames,ignore_index=True),"a2_fine_tuning_operating_points")
    save(grid,"a2_guard_grid")
    write_fig(fig,"a2_archived_operating_points")


def archived_transfer():
    source=read("e5/external/paired_2021.csv")
    new=read("temporal_evaluation/transfer/replays.csv")
    old=read("e6/transfer_2021_by_day_seed.csv")
    workloads=pd.concat([read("e6/feasibility_2021_by_day.csv").assign(bank="original"),
                         read("temporal_evaluation/analysis/transfer_feasibility.csv").assign(bank="reserved")])
    save(source,"a3_original_external_all_arms")
    save(old,"a3_original_fine_tuning_transfer")
    save(new,"a3_reserved_transfer")
    save(workloads,"a3_workload_and_feasibility")
    telemetry=read("e6/queue_by_trace_seed.csv")
    save(telemetry[telemetry.corpus.astype(str).eq("2021")],"a3_original_queue_telemetry")
    metrics=[c for c in new.columns if c in ("violations","violation_rate","p95_ms","p99_ms","override_rate",
                                             "overrides","direct_saves","direct_losses","backlog_p95","backlog_max")
             or c.startswith(("server_","saturation_","upper_","lower_"))]
    baseline=new[new.arm.str.endswith("_off")].assign(mode=lambda f:f.arm.str.split("_").str[0])
    guard=new[new.arm.str.endswith("_on")].assign(mode=lambda f:f.arm.str.split("_").str[0])
    key=["key","day","seed","mode","trace_sha256","model_sha256"]
    buf=guard[key+metrics].merge(baseline[key+metrics],on=key,suffixes=("_on","_off"),validate="one_to_one")
    for c in metrics:
        buf["delta_"+c]=buf[c+"_on"]-buf[c+"_off"]
    save(buf,"a3_reserved_trajectory_pairs")
    save(buf.groupby(["day","mode"],as_index=False)[[c for c in buf if c.startswith("delta_")]].mean(),
         "a3_reserved_trajectory_by_window")


def robustness():
    f=read("verified_final/runs/robustness.csv")
    assert len(f)==5400
    f["met_budget"]=f.violation_rate.le(f.target)
    g=["servers","target","load","variant","protocol","method","fixed_lambda"]
    save(points(f,g),"a4_robustness_complete")
    main=f[f.servers.isin([3,5])&f.target.eq(.005)&f.load.isin([.4,.6,.8])&f.method.eq("fixed")]
    save(points(main,g),"a4_main_exhibit")
    for filename in ("state_ablations","reward_clipping"):
        source=study.ROOT/"outputs/verified_final/runs"/f"{filename}.csv"
        if source.exists():
            data=pd.read_csv(source)
            save(data,f"a4_{filename}_rows")
            if {"p95_ms","mean_ms","violation_rate","target"}<=set(data):
                data["met_budget"]=data.violation_rate.le(data.target)
                grouping=[k for k in ("method","mask","reward_clip","clipping","variant","protocol","clip_reward") if k in data]
                if grouping:
                    save(points(data,grouping),f"a4_{filename}_summary")
    for key,file in (("guard_screen","e5/robustness_summary.csv"),("feasibility","risk_verified/runs/feasibility_paired.csv")):
        save(read(file),f"a4_{key}")


def main():
    start=time.perf_counter()
    OUT.mkdir(exist_ok=True)
    refs=["estimated_ddqn/validation/paired.csv","temporal_evaluation/B/replays.csv",
          "temporal_evaluation/transfer/replays.csv","verified_final/runs/robustness.csv",
          "e5/external/paired_2021.csv","e6/queue_by_trace_seed.csv","e6/feasibility_2021_by_day.csv",
          "verified_final/e2/pilot_paired.csv","e5/threshold_bias_summary.csv",
          "e6/transfer_2021_by_day_seed.csv","e5/robustness_summary.csv","risk_verified/runs/feasibility_paired.csv"]
    refs.extend(f"verified_final/runs/{name}.csv" for name in ("state_ablations","reward_clipping")
                if (study.ROOT/"outputs/verified_final/runs"/f"{name}.csv").exists())
    key=study.lock(OUT/"protocol.json", {"script_sha256":study.digest(__file__),
        "inputs":{name:study.digest(study.ROOT/"outputs"/name) for name in refs},
        "bootstrap":{"seed":1127,"replicates":4000,"estimand":"conditional descriptive, paired day/seed/draw"},
        "budgets_integer_per_10000":BUDGETS.tolist(),"new_replays":0})
    banks=fine_tuning_banks()
    budget_analysis(banks)
    tradeoffs(banks)
    archived_transfer()
    robustness()
    mem=psutil.Process().memory_info()
    study.write(OUT/"execution.json",{"status":"COMPLETE","protocol":key,"new_replays":0,
        "wall_s":time.perf_counter()-start,"peak_rss_bytes":getattr(mem,"peak_wset",mem.rss),
        "outputs":{p.name:study.digest(p) for p in OUT.iterdir() if p.suffix in (".csv",".png",".pdf")}})


if __name__=="__main__":
    main()
