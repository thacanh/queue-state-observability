"""Summarize the archived transfer and penalty comparisons."""

import os

import replay_study as study

os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
import pandas as pd

from analyze_new import OUT, compare, fig_save, grouped, plt, pooled_contrasts, save, verified


def archive_extensions():
    paths={
        "a3_original_transfer_full_telemetry":"q2_revision/results/transfer_paired.csv",
        "a4_original_burst_tail_rows":"verified_final/runs/arrival_tail.csv",
        "a4_original_burst_tail_summary":"verified_final/results/E4_tail_summary.csv",
        "a4_original_feasibility_scope":"verified_final/results/E3_feasibility.csv",
    }
    src={name:study.ROOT/"outputs"/f for name,f in paths.items()}
    key=study.lock(OUT/"archived_extensions_protocol_v2.json",{
        "code_sha256":study.digest(__file__),"sources":{n:study.digest(p) for n,p in src.items()},"new_replays":0})
    for tag,p in src.items():
        f=pd.read_csv(p)
        save(f,tag)
        if tag=="a3_original_transfer_full_telemetry":
            assert len(f)==100 and set(f.day)==set(range(9,14))
            keys=["day","seed","training","trace_sha256"]
            paired=compare(f[f.guard],f[~f.guard],keys,[])
            assert paired.model_sha256_candidate.eq(paired.model_sha256_ref).all()
            pooled_contrasts(paired,["day","training"],"a3_original_transfer_guard")
            extra=[c for c in f.columns if c.startswith(("server_","saturation_","upper_","lower_")) or
                   c in ("overrides","direct_saves","direct_losses","backlog_p95","backlog_max","max_routed_server_load")]
            idx=f.set_index(keys+["guard"])
            change=(idx.xs(True,level="guard")[extra]-idx.xs(False,level="guard")[extra]).reset_index()
            save(change,"a3_original_transfer_telemetry_delta")
    study.write(OUT/"archived_extensions_execution.json",{"protocol":key,"new_replays":0,"status":"COMPLETE",
                "output":{k:study.digest(OUT/f"{k}.csv") for k in src}})


def t2():
    f,_=verified("T2")
    f["penalty"]=f["mode"].str.extract(r"lambda(\d+)$")[0].fillna(2).astype(int)
    f["family"]=f["mode"].str.split("_").str[0]
    cols=["family","guard","penalty"]
    summary=grouped(f,cols)
    save(summary,"t2_penalty_all")
    save(grouped(f,["day",*cols]),"t2_penalty_by_day")
    join=["key","day","seed","trace_sha256","guard"]
    pair=compare(f[f.family.eq("oracle")],f[f.family.eq("oracle")&f.penalty.eq(2)],join,["family","penalty"])
    pooled_contrasts(pair,["day","family","penalty","guard"],"t2_oracle_vs_lambda2")
    ref=f[f.family.eq("estimated")&f.penalty.eq(2)&f.guard]
    matched=compare(f[f.family.eq("oracle")],ref,join[:-1],["family","penalty","guard"])
    pooled_contrasts(matched,["day","family","penalty","guard"],"t2_vs_estimated_guarded")
    levels=pooled_contrasts(matched,["family","penalty","guard"],"t2_vs_estimated_guarded_all")
    reference=ref.p95_ms.mean()
    levels["p95_pct_vs_reference"]=100*levels.delta_p95_ms/reference
    levels["within_1_pct"]=levels.p95_pct_vs_reference.abs().le(1)
    levels["within_2_pct"]=levels.p95_pct_vs_reference.abs().le(2)
    levels["does_not_exceed_reference"]=levels.delta_p95_ms.le(1e-9)
    save(levels,"t2_latency_tolerances")

    fig,ax=plt.subplots(figsize=(5.65,3.6))
    oracle=summary[summary.family.eq("oracle")]
    for flag,m in oracle.groupby("guard"):
        ax.scatter(m.p95_ms,100*m.violation_rate,s=24,marker="o" if flag else "x",
                   color="#0c739b" if flag else "#617181",label="Oracle FT · guard "+("on" if flag else "off"))
        for r in m.itertuples():
            ax.annotate(f"λ={r.penalty}",(r.p95_ms,100*r.violation_rate),fontsize=7,
                        xytext=(3,3 if flag else -8),textcoords="offset points")
    for flag,m in summary[summary.family.eq("estimated")].groupby("guard"):
        ax.scatter(m.p95_ms,100*m.violation_rate,s=58,marker="D" if flag else "+",color="#ad5135",
                   label="Estimated FT λ=2 · guard "+("on" if flag else "off"))
    ax.axvspan(reference*.99,reference*1.01,facecolor=".92",zorder=0)
    ax.set(xlabel="Mean workload P95 (ms)",ylabel="Mean workload miss (%)")
    ax.legend(fontsize=7)
    fig_save(fig,"t2_paired_penalty_tradeoff")


if __name__=="__main__":
    archive_extensions()
    t2()
