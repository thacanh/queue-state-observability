"""Verify the validation bank and analyze four paired observation arms."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


KEYS = ["day", "reconstruction_seed", "seed"]
METRICS = ["violation_rate", "met_budget", "mean_ms", "p95_ms", "p99_ms"]
LABELS = [("oracle", False), ("oracle", True),
          ("estimated_fine_tuning", False), ("estimated_fine_tuning", True)]
BUDGET = .005


def digest(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read(p):
    return json.loads(p.read_text(encoding="utf8"))


def check(path, expected):
    if digest(path) != expected:
        raise ValueError(f"SHA-256 mismatch: {path.name}")


def contrast(left, right):
    assert left.index.equals(right.index)
    return left[METRICS].astype(float) - right[METRICS].astype(float)


def ecdf(arms, target):
    styles = {"oracle":"#a5492a", "estimated_fine_tuning":"#126587"}
    plt.rcParams.update({"font.size":10, "axes.labelsize":11, "pdf.fonttype":42})
    fig, ax = plt.subplots(figsize=(6.45,3.25), layout="constrained")
    for train, guard in LABELS:
        a = np.sort(arms[train, guard].violation_rate.values)
        name = "Oracle fine-tuning" if train=="oracle" else "Estimated fine-tuning"
        ax.step(np.r_[a[0],a], np.r_[0,np.arange(1,len(a)+1)/len(a)], where="post",
                lw=2, color=styles[train], linestyle="-" if guard else "--",
                label=f"{name}, guard {'on' if guard else 'off'}")
    ax.axvline(BUDGET, color="#454545", linestyle=":", lw=1.3)
    ax.annotate("0.5% budget", (BUDGET,.985), xytext=(5,0), textcoords="offset points",
                va="top", color="#454545", fontsize=9)
    ax.set(xlabel="Workload deadline miss rate", ylabel="Share of paired replays", ylim=(0,1.035))
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1,decimals=1))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.legend(loc="lower right", fontsize=8, framealpha=.92, borderpad=.7)
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="y",color="#d8dde2",lw=.6)
    fig.savefig(target, bbox_inches="tight", metadata={"CreationDate":None,"ModDate":None})
    plt.close(fig)


def run(artifact, reference, output, csv_only=False):
    sys.path.insert(0,str((reference/"code").resolve()))
    from guard_statistics import grouped_differences

    frozen = artifact/"outputs"
    shared_manifest = read(reference/"results/manifest.json")
    for n,h in shared_manifest["outputs"].items():
        check(reference/"results"/n,h)
    for seed, h in shared_manifest["replay_manifests"].items():
        folder = reference/f"replays/validation_seed{seed}"
        check(folder/"manifest.json",h)
        run_manifest = read(folder/"manifest.json")
        for filename, expected in run_manifest["outputs"].items():
            check(folder/filename,expected)
    first = read(reference/"replays/validation_seed42/manifest.json")
    bank = frozen/"shield_verified/source_data/runs/deadline_guard_v1/inputs"
    oldpath = frozen/"guard_evaluation/paired.csv"
    check(oldpath,first["baseline_sha256"])
    old = pd.read_csv(oldpath)
    if csv_only:
        index = old[["day","reconstruction_seed","trace_sha256"]].drop_duplicates()
        assert len(index)==59 and not index.duplicated(KEYS[:2]).any()
        fingerprints = {f"{r.day}_{r.reconstruction_seed}":{"frame_sha256":r.trace_sha256}
                        for r in index.itertuples()}
    else:
        check(bank/"manifest.json",first["validation_bank_sha256"])
        fingerprints = read(bank/"manifest.json")
        for item in fingerprints.values():
            check(bank/item["filename"],item["sha256"])
    data = pd.read_csv(reference/"results/paired.csv")
    assert len(data)==1180 and set(data.seed)==set(range(42,47))
    days = data[KEYS[:2]].drop_duplicates().groupby("day").size().to_dict()
    assert days=={"d08":28,"d09":31}
    assert not data.duplicated(KEYS+["policy_training","guard"]).any()
    assert data.groupby(KEYS).trace_sha256.nunique().eq(1).all()
    assert all(row.trace_sha256 == fingerprints[f"{row.day}_{row.reconstruction_seed}"]["frame_sha256"]
               for row in data.itertuples())
    assert data.groupby(KEYS+["policy_training"]).model_sha256.nunique().eq(1).all()
    np.testing.assert_allclose(data.violation_rate,data.violations/data.n,rtol=0,atol=1e-15)
    assert (data.met_budget == (data.violations*200<=data.n)).all()
    arms = {label: data[(data.policy_training==label[0]) & (data.guard==label[1])]
            .set_index(KEYS,verify_integrity=True).sort_index() for label in LABELS}
    for seed in range(42,47):
        manifest, = sorted((reference/"checkpoints").glob(f"seed-{seed}-*/status.json"))
        model = read(manifest)
        weights = manifest.with_name("final.pt")
        if not csv_only or weights.exists():
            check(weights, model["model_sha256"])
        for label in LABELS:
            sha = model["model_sha256"] if label[0]!="oracle" else model["oracle_model_sha256"]
            assert set(arms[label].xs(seed,level="seed").model_sha256)=={sha}
    assert all(len(a)==295 and a.index.equals(arms[LABELS[0]].index) for a in arms.values())
    oo, og, eo, eg = (arms[label] for label in LABELS)
    def estimate(left,right):
        rows = left[METRICS].add_suffix("_candidate").join(right[METRICS].add_suffix("_comparator"))
        return grouped_differences(rows.reset_index())

    impacts = {
        "fine_tuning_guard_off":(eo,oo),"fine_tuning_guard_on":(eg,og),
        "guard_oracle_training":(og,oo),"guard_estimated_training":(eg,eo),
        "training_by_guard":(contrast(eg,eo),contrast(og,oo)),
    }
    exact = old.query("observation=='oracle' and method=='Fixed-DDQN+Guard'").set_index(KEYS,verify_integrity=True).sort_index()
    assert len(exact)==295 and exact.index.equals(eg.index)
    assert exact.trace_sha256.equals(eg.trace_sha256) and exact.model_sha256.equals(og.model_sha256)
    np.testing.assert_allclose(exact.violation_rate,exact.violations/exact.n,rtol=0,atol=1e-15)
    impacts["estimated_guard_vs_oracle_guard"] = (eg,exact)
    output.mkdir(parents=True,exist_ok=True)
    diffs = pd.DataFrame(
        [{"contrast":key,**item} for key,(a,b) in impacts.items() for item in estimate(a,b)])
    diffs.to_csv(output/"training_guard_interaction.csv",index=False)
    records, proximity, criteria, near, subgroup = [],[],[],[],[]
    for label, frame in arms.items():
        records.append({"training":label[0],"guard":label[1],"pairs":len(frame),"passes":int(frame.met_budget.sum()),
                        "attainment":frame.met_budget.mean(),**frame[["violation_rate","mean_ms","p95_ms","p99_ms",
                         "override_rate","wait_bias_ms","wait_MAE_ms","brier"]].mean().to_dict()})
        if not label[1]:
            proximity.append({"training":label[0],"median_miss_rate":frame.violation_rate.median(),
                              "above_050_through_060":int(frame.violation_rate.between(.005,.006,inclusive="right").sum()),
                              "above_050_through_055":int(frame.violation_rate.between(.005,.0055,inclusive="right").sum()),
                              "pairs":len(frame)})
        for group, a in [*frame.groupby(level="day"),("pilot",frame[frame.index.get_level_values("seed")<=43]),
                         ("additional",frame[frame.index.get_level_values("seed")>=44]),
                         *[(f"seed_{seed}",s) for seed,s in frame.groupby(level="seed")]]:
            subgroup.append({"training":label[0],"guard":label[1],"group":group,"pairs":len(a),
                             "passes":int(a.met_budget.sum()),**a[["violation_rate","mean_ms","p95_ms","p99_ms"]].mean().to_dict()})
    for train, a,b in [("oracle",oo,og),("estimated_fine_tuning",eo,eg)]:
        for seed in range(42,47):
            x,y = a.xs(seed,level="seed"),b.xs(seed,level="seed")
            criteria.append({"training":train,"seed":seed,
                             "pass_off":int(x.met_budget.sum()),"pass_on":int(y.met_budget.sum()),
                             "gained":int((~x.met_budget&y.met_budget).sum()),
                             "lost":int((x.met_budget&~y.met_budget).sum()),
                             "miss_reduction_replays":int((y.violation_rate<x.violation_rate).sum()),
                             "miss_ties":int((y.violation_rate==x.violation_rate).sum()),
                             "attainment_gain":float(y.met_budget.mean()-x.met_budget.mean()),
                             "p95_change_percent":100*(y.p95_ms.mean()/x.p95_ms.mean()-1)})
    for label, a in (("oracle_observations",exact),("estimated_training_and_observations",eg)):
        near.append({"arm":label,"passes":int(a.met_budget.sum()),"pairs":len(a),
                     **a[["violation_rate","mean_ms","p95_ms","p99_ms"]].mean().to_dict(),
                     "estimated_only_passes":int((eg.met_budget&~exact.met_budget).sum()),
                     "oracle_only_passes":int((exact.met_budget&~eg.met_budget).sum())})
    for name, dataset in (("training_guard_summary",records),("threshold_proximity",proximity),
                          ("guard_success_criteria",criteria),("near_oracle_comparison",near),
                          ("subgroups",subgroup)):
        pd.DataFrame(dataset).to_csv(output/f"{name}.csv",index=False)
    ecdf(arms,output/"miss_rate_ecdf.pdf")
    (output/"validation_analysis_manifest.json").write_text(json.dumps({
        "source_paired_sha256":digest(reference/"results/paired.csv"),"historical_pairs_sha256":digest(oldpath),
        "reconstruction_manifest_sha256":first["validation_bank_sha256"],"analysis_sha256":digest(Path(__file__)),
        "raw_reconstruction_hashes_verified":not csv_only,
        "outputs":{name:digest(output/name) for name in (
            "training_guard_interaction.csv","training_guard_summary.csv","threshold_proximity.csv",
            "guard_success_criteria.csv","near_oracle_comparison.csv","subgroups.csv","miss_rate_ecdf.pdf")},
        "inference":"Post-hoc. 4,000 common crossed seed/day bootstrap draws, within-day reconstruction pairing."
    },indent=2)+"\n",encoding="utf8")
    print(pd.DataFrame(records).to_string(index=False))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("artifact","reference","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--csv-only",action="store_true",help="Validate published replay rows without derived trace files.")
    args=parser.parse_args()
    run(args.artifact,args.reference,args.output,args.csv_only)
