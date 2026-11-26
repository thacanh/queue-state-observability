"""Render the four transfer arms from the verified per-window aggregates."""

import argparse
import json
from pathlib import Path
import hashlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd


def render(results, destination):
    path = results/"transfer_by_day.csv"
    expected = json.loads((results/"transfer_manifest.json").read_text())["outputs"][path.name]
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError("Transfer rows changed")
    data = pd.read_csv(path)
    assert len(data)==20 and set(data.day)==set(range(9,14))
    colors = {"oracle":"#a5492a", "estimated":"#126587"}
    plt.rcParams.update({"font.size":9,"axes.labelsize":10,"pdf.fonttype":42})
    fig, axes = plt.subplots(1,2,figsize=(6.45,2.75),layout="constrained")
    for (train,guard),g in data.groupby(["training","guard"]):
        label = f"{train.title()} FT, guard {'on' if guard else 'off'}"
        style = dict(color=colors[train], marker="o" if guard else "s", markersize=4,
                     linestyle="-" if guard else "--", lw=1.45, label=label)
        axes[0].plot(g.day,g.violation_rate,**style)
        axes[1].plot(g.day,g.p95_ms/1000,**style)
    axes[0].set(ylabel="Mean workload miss rate", ylim=(0,.43))
    axes[0].yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
    axes[1].set(ylabel="Mean workload P95 (s)",yscale="log")
    for ax in axes:
        ax.set(xlabel="Azure 2021 elapsed day",xticks=list(range(9,14)))
        ax.grid(axis="y",color="#e3e6e9",lw=.55)
        ax.spines[["top","right"]].set_visible(False)
    handles,labels=axes[0].get_legend_handles_labels()
    order=(2,3,0,1)
    axes[0].legend([handles[i] for i in order],[labels[i] for i in order],loc="upper left",
                   ncol=1,fontsize=7.2,framealpha=.9,handlelength=2.0)
    fig.savefig(destination,bbox_inches="tight",metadata={"CreationDate":None,"ModDate":None})
    plt.close(fig)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("results",type=Path)
    p.add_argument("destination",type=Path)
    a=p.parse_args()
    render(a.results,a.destination)
