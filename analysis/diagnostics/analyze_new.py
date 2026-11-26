"""Summarize every predeclared arm and the matched transfer event logs."""

from pathlib import Path
import os

import replay_study as study

os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT=study.BASE/"processed"
METRICS=["violation_rate","mean_ms","p95_ms","p99_ms","met_budget","override_rate","wait_bias_ms","wait_MAE_ms"]
PALETTE={"oracle":"#0c739b","estimated":"#ad5135"}
plt.rcParams.update({"font.size":9, "axes.spines.top":False, "axes.spines.right":False,
                     "legend.frameon":False, "pdf.fonttype":42})


def verified(phase):
    folder=study.BASE/phase
    ex=study.read(folder/"execution.json")
    pr=study.read(folder/"protocol.json")
    assert ex["status"]=="COMPLETE" and ex["replays_sha256"]==study.digest(folder/"replays.csv")
    sha=pr.pop("protocol_sha256")
    assert sha==ex["protocol"]==study.fingerprint(pr)
    for name,h in ex["shards"].items():
        assert h==study.digest(folder/"shards"/name)
    data=pd.read_csv(folder/"replays.csv")
    assert len(data)==ex["recorded_rows"]
    np.testing.assert_array_equal(data.violations*200<=data.n,data.met_budget)
    return data,pr


def fig_save(fig,name):
    fig.tight_layout()
    fig.savefig(OUT/f"{name}.pdf",bbox_inches="tight")
    fig.savefig(OUT/f"{name}.png",bbox_inches="tight",dpi=160)
    plt.close(fig)


def save(f,name):
    f.to_csv(OUT/f"{name}.csv",index=False)


def grouped(data,groups):
    metrics=[m for m in METRICS if m in data]
    metrics+= [m for m in data if m.startswith(("worker_","upper_","lower_"))]
    return data.groupby(groups,dropna=False,as_index=False).agg(
        replays=("n","size"),passes=("met_budget","sum"),**{m:(m,"mean") for m in metrics})


def compare(candidate,base,join,metadata):
    metrics=[m for m in METRICS if m in candidate and m in base]
    cols=list(dict.fromkeys(join+metadata+metrics+["model_sha256","n"]))
    c=candidate[cols].merge(base[join+metrics+["model_sha256","n"]],on=join,validate="many_to_one",
                             suffixes=("_candidate","_ref"))
    assert len(c)==len(candidate) and c.n_candidate.eq(c.n_ref).all()
    for m in metrics:
        c["delta_"+m]=c[m+"_candidate"].astype(float)-c[m+"_ref"].astype(float)
    if "met_budget" in metrics:
        c["fail_to_pass"]=c.met_budget_candidate & ~c.met_budget_ref
        c["pass_to_fail"]=~c.met_budget_candidate & c.met_budget_ref
    return c


def pooled_contrasts(pair,groups,stem):
    save(pair,stem+"_paired")
    cols=[c for c in pair if c.startswith("delta_")]
    out=pair.groupby(groups,dropna=False,as_index=False).agg(
        replays=("n_candidate","size"),**{m:(m,"mean") for m in cols},
        fail_to_pass=("fail_to_pass","sum"),pass_to_fail=("pass_to_fail","sum"))
    save(out,stem+"_paired_summary")
    return out


def t1():
    f,_=verified("T1")
    arm=["mode","guard","trigger","reduction"]
    day=grouped(f,["day",*arm])
    save(day,"t1_by_day")
    sums=grouped(f,arm)
    ref=f[f["mode"].eq("estimated")&f.guard&f.trigger.eq(.005)&f.reduction.eq(.005)]
    join=["key","day","seed","trace_sha256"]
    pair=compare(f,ref,join,arm)
    pooled_contrasts(pair,["day",*arm],"t1_vs_estimated_guarded_default")
    pooled=pooled_contrasts(pair,arm,"t1_vs_estimated_guarded_default_all")
    ref_p95=ref.p95_ms.mean()
    pooled["p95_pct_vs_reference"]=100*pooled.delta_p95_ms/ref_p95
    pooled["within_1_pct"]=pooled.p95_pct_vs_reference.abs().le(1.)
    pooled["within_2_pct"]=pooled.p95_pct_vs_reference.abs().le(2.)
    pooled["does_not_exceed_reference"]=pooled.delta_p95_ms.le(1e-9)
    save(pooled,"t1_latency_tolerances")
    fig,ax=plt.subplots(figsize=(5.7,3.55))
    ax.axvspan(ref_p95*.99,ref_p95*1.01,facecolor=".92")
    ax.axvline(ref_p95,color=".55",lw=.85)
    for mode,group in sums.groupby("mode"):
        k=group.guard
        ax.scatter(group.loc[k,"p95_ms"],100*group.loc[k,"violation_rate"],color=PALETTE[mode],label=mode.title()+" FT · guard grid")
        ax.scatter(group.loc[~k,"p95_ms"],100*group.loc[~k,"violation_rate"],
                   color=PALETTE[mode],marker="x",label=mode.title()+" FT · guard off")
    baseline=sums[sums["mode"].eq("estimated")&sums.guard&sums.trigger.eq(.005)&sums.reduction.eq(.005)]
    ax.scatter(baseline.p95_ms,100*baseline.violation_rate,s=105,facecolors="none",edgecolors="black",
               label="Estimated FT · default guard")
    ax.set(xlabel="Mean workload P95 (ms)",ylabel="Mean workload miss (%)")
    ax.legend(fontsize=7)
    fig_save(fig,"t1_paired_guard_tradeoff")


def t3_t7(phase):
    f,_=verified(phase)
    flag="corrected" if phase=="T3" else "recalibrated"
    variant=phase.lower()
    cols=["day","mode","guard",flag]
    save(grouped(f,cols),variant+"_by_day")
    save(grouped(f,cols[1:]),variant+"_all_arms")
    k=["day","key","seed","mode","guard","trace_sha256"]
    pair=compare(f[f[flag]],f[~f[flag]],k,[])
    assert pair.model_sha256_candidate.eq(pair.model_sha256_ref).all()
    pooled_contrasts(pair,["day","mode","guard"],variant+"_vs_original")
    pooled_contrasts(pair,["mode","guard"],variant+"_vs_original_all")
    fig,axes=plt.subplots(1,len(f.day.unique()),figsize=(9,3),squeeze=False)
    for ax,(day,group) in zip(axes.flat,f.groupby("day")):
        sums=grouped(group,["mode","guard",flag])
        scale=1000 if phase=="T7" else 1
        for (mode,guard),m in sums.groupby(["mode","guard"]):
            old=m.loc[~m[flag]].iloc[0]
            new=m.loc[m[flag]].iloc[0]
            xy=(new.p95_ms/scale,100*new.violation_rate)
            prior=(old.p95_ms/scale,100*old.violation_rate)
            ax.annotate("",xy=xy,xytext=prior,arrowprops={"arrowstyle":"->","color":PALETTE[mode],
                         "linestyle":"-" if guard else "--","lw":1})
            ax.scatter(*prior,facecolors="none",edgecolors=PALETTE[mode],s=17)
            ax.scatter(*xy,color=PALETTE[mode],s=17)
        ax.set_title(str(day) if phase=="T3" else f"Elapsed day {day}")
        ax.set_xlabel("Mean workload P95 ("+("s)" if phase=="T7" else "ms)"))
    axes.flat[0].set_ylabel("Mean workload miss (%)")
    from matplotlib.lines import Line2D
    handles=[Line2D([],[],color=c,label=f"{m.title()} FT") for m,c in PALETTE.items()]
    handles.extend([Line2D([],[],color=".3",ls="--",label="Guard off"),
                    Line2D([],[],color=".3",label="Guard on")])
    fig.legend(handles=handles,ncol=4,fontsize=7,loc="upper center")
    fig.subplots_adjust(top=.78)
    fig_save(fig,variant+"_calibration_tradeoff")


def t5():
    f,_=verified("T5")
    cols=["channel","mode","kind","bias_ms","sigma_ms"]
    save(grouped(f,["day",*cols]),"t5_dose_by_day")
    sums=grouped(f,cols)
    save(sums,"t5_dose_averages")
    k=["day","key","seed","mode","channel","trace_sha256"]
    for label in ("zero","estimated"):
        pair=compare(f[~f.kind.eq(label)],f[f.kind.eq(label)],k,["kind","bias_ms","sigma_ms"])
        assert pair.model_sha256_candidate.eq(pair.model_sha256_ref).all()
        pooled_contrasts(pair,["day",*cols],f"t5_vs_{label}")
    # Independently verify the overlapping unperturbed oracle-checkpoint D arms.
    d=pd.read_csv(study.TEMPORAL/"D/replays.csv")
    sample=f[(f.kind=="zero")&(f["mode"]=="oracle")].copy()
    sample["arm"]=sample.channel.map({"guard":"EO","policy":"O-"})
    ref=sample.merge(d,on=["key","seed","arm","trace_sha256","model_sha256"],validate="one_to_one",
                     suffixes=("_sample","_ref"))
    assert len(ref)==24
    for col in ("violations","p95_ms","p99_ms"):
        np.testing.assert_allclose(ref[col+"_sample"],ref[col+"_ref"],rtol=0,atol=1e-6)
    save(ref[["key","seed","arm","trace_sha256","model_sha256","violations_sample","p95_ms_sample"]],
         "t5_zero_identity_check")
    fig,axes=plt.subplots(2,2,figsize=(8.8,6.0),sharey="row")
    for row_idx,(metric,label,factor) in enumerate([("violation_rate","Mean workload miss (%)",100),
                                                  ("p95_ms","Mean workload P95 (ms)",1)]):
        for col_idx,channel in enumerate(("policy","guard")):
            ax=axes[row_idx,col_idx]
            subset=sums[sums.channel.eq(channel)]
            for mode,data in subset.groupby("mode"):
                true=data[data.kind.eq("estimated")].iloc[0]
                for kind,linestyle,marker in (("bias","-","o"),("jitter","--","s")):
                    part=data[data.kind.isin(["zero",kind])].sort_values("worker_mae_ms")
                    if kind=="bias":
                        negative=part[part.bias_ms.lt(0)]
                        ax.scatter(negative.worker_mae_ms,negative[metric]*factor,marker="v",
                                   facecolors="none",edgecolors=PALETTE[mode],s=24)
                        part=part[part.bias_ms.ge(0)]
                    ax.plot(part.worker_mae_ms,part[metric]*factor,color=PALETTE[mode],linestyle=linestyle,
                            marker=marker,ms=3,label=f"{mode.title()} FT · {kind}")
                ax.axhline(true[metric]*factor,color=PALETTE[mode],linestyle=":",lw=.85)
            ax.set_title(f"{channel.title()} observation")
            ax.set_xlabel("Realized worker residual MAE (ms)")
            if col_idx==0:
                ax.set_ylabel(label)
    from matplotlib.lines import Line2D
    handles,labels=axes[0,0].get_legend_handles_labels()
    handles.append(Line2D([],[],marker="v",linestyle="none",markerfacecolor="none",
                          markeredgecolor=".25",markersize=5))
    labels.append("Negative bias")
    axes[0,0].legend(handles,labels,fontsize=6.7)
    fig_save(fig,"t5_observation_error_dose_response")


def read_event(row):
    path=study.BASE/"T6/events"/row.event_file
    assert study.digest(path)==row.event_sha256
    with np.load(path,allow_pickle=False) as data:
        return {k:data[k] for k in data.files}


def t6():
    f,_=verified("T6")
    assert len(f)==40 and f.archive_match.all()
    transitions,overall,horizons,request_horizons=[],[],[],[]
    for (key,day,mode,seed),m in f.groupby(["key","day","mode","seed"]):
        on=read_event(m[m.guard].iloc[0])
        off=read_event(m[~m.guard].iloc[0])
        for field in ("arrival_s","request_id","deadline_s","demand_s"):
            np.testing.assert_array_equal(on[field],off[field])
        ids=np.flatnonzero(on["override"])
        arrival=on["arrival_s"]
        assert len(ids)==int(m.loc[m.guard,"overrides"].iloc[0])
        unit=dict(key=key,day=day,mode=mode,seed=seed)
        overall.append(unit|dict(requests=len(arrival),override_count=len(ids),
            guard_trigger_count=int(on["guard_activated"].sum()),
            direct_saves=int(np.count_nonzero(on["proposed_miss_same_state"][ids]&~on["miss"][ids])),
            direct_losses=int(np.count_nonzero(~on["proposed_miss_same_state"][ids]&on["miss"][ids])),
            total_miss_delta=int(on["miss"].sum())-int(off["miss"].sum()),
            on_backlog_p95=float(np.quantile(on["queued"].sum(1),.95)),
            off_backlog_p95=float(np.quantile(off["queued"].sum(1),.95))))
        for frm,to in [(i,j) for i in range(3) for j in range(3) if i!=j]:
            sel=ids[(on["proposal"][ids]==frm)&(on["selected"][ids]==to)]
            transitions.append(unit|dict(proposed=frm+1,selected=to+1,overrides=len(sel),
                direct_saves=int(np.sum(on["proposed_miss_same_state"][sel]&~on["miss"][sel])),
                direct_losses=int(np.sum(~on["proposed_miss_same_state"][sel]&on["miss"][sel]))))
        queue_on=on["queued"]
        queue_off=off["queued"]
        miss_on=np.r_[0,np.cumsum(on["miss"])]
        miss_off=np.r_[0,np.cumsum(off["miss"])]
        routed_on=np.vstack([np.r_[0,np.cumsum(on["demand_s"]*(on["selected"]==j))] for j in range(3)])
        routed_off=np.vstack([np.r_[0,np.cumsum(off["demand_s"]*(off["selected"]==j))] for j in range(3)])
        cap=np.array([6.,3.,1.4])
        dest=on["selected"]
        for h in (1,10,60):
            keep=ids[(arrival[ids]+h<=arrival[-1])&(arrival[ids]-h>=arrival[0])]
            lo=np.searchsorted(arrival,arrival[keep]-h,side="left")
            hi=np.searchsorted(arrival,arrival[keep]+h,side="right")
            j=dest[keep]
            for a,b,c,d in zip(keep,lo,hi,j):
                horizons.append(unit|dict(request_id=int(a),horizon_s=h,proposed_server=int(on["proposal"][a])+1,
                destination=int(d)+1,future_requests=int(c-a-1),
                prior_dest_queue_on=int(queue_on[b,d]),prior_dest_queue_off=int(queue_off[b,d]),
                dest_queue_on=int(queue_on[a,d]),dest_queue_off=int(queue_off[a,d]),
                after_dest_queue_on=int(queue_on[c-1,d]),after_dest_queue_off=int(queue_off[c-1,d]),
                future_misses_on=int(miss_on[c]-miss_on[a+1]),future_misses_off=int(miss_off[c]-miss_off[a+1]),
                future_dest_load_on=float((routed_on[d,c]-routed_on[d,a+1])/(cap[d]*h)),
                future_dest_load_off=float((routed_off[d,c]-routed_off[d,a+1])/(cap[d]*h))))
        for h in (10,100):
            keep=ids[ids+h<len(arrival)]
            for a in keep:
                b=a+h+1
                request_horizons.append(unit|dict(request_id=int(a),future_n=h,
                    misses_on=int(miss_on[b]-miss_on[a+1]),misses_off=int(miss_off[b]-miss_off[a+1])))
    sums=pd.DataFrame(overall)
    save(sums,"t6_whole_trajectory_and_snapshot_by_seed")
    tr=pd.DataFrame(transitions)
    save(tr,"t6_server_transitions_by_seed")
    save(tr.groupby(["day","mode","proposed","selected"],as_index=False)[["overrides","direct_saves","direct_losses"]].sum(),
         "t6_server_transitions")
    events=pd.DataFrame(horizons)
    save(events,"t6_override_event_horizons")
    save(pd.DataFrame(request_horizons),"t6_override_request_horizons")
    events["delta_future_miss_count"]=events.future_misses_on-events.future_misses_off
    events["delta_dest_load"]=events.future_dest_load_on-events.future_dest_load_off
    events["delta_followup_backlog"]=events.after_dest_queue_on-events.after_dest_queue_off
    cols=["delta_future_miss_count","delta_dest_load","delta_followup_backlog"]
    grouped_by_seed=events.groupby(["day","mode","seed","horizon_s"],as_index=False)[cols].mean()
    save(grouped_by_seed,"t6_event_comparison_by_seed")
    group=grouped_by_seed.groupby(["day","mode","horizon_s"],as_index=False)[cols].mean()
    save(group,"t6_event_comparison")
    fig,ax=plt.subplots(1,2,figsize=(8.2,3.2),sharey=True)
    for axis,(d,m) in zip(ax,group.groupby("day")):
        for mode,sub in m.groupby("mode"):
            axis.plot(sub.horizon_s,sub.delta_followup_backlog,marker="o",ms=4,color=PALETTE[mode],
                      label=mode.title()+" FT")
        axis.axhline(0,color=".6",lw=.75)
        axis.set(xscale="log",xlabel="Seconds after override",title=f"Elapsed day {d}")
        axis.set_xticks([1,10,60],["1","10","60"])
    ax[0].set_ylabel("Target-server queued jobs, on − off")
    ax[0].legend()
    fig_save(fig,"t6_guard_event_backlog")


def main():
    OUT.mkdir(exist_ok=True)
    t1()
    t3_t7("T3")
    t5()
    t6()
    t3_t7("T7")
    study.write(OUT/"checks.json", {"validated_completed_phases":["T1","T3","T5","T6","T7"],
                "script_sha256":study.digest(__file__),"outputs":{p.name:study.digest(p) for p in OUT.iterdir()
                if p.suffix in (".csv",".png",".pdf")}})


if __name__=="__main__":
    main()
