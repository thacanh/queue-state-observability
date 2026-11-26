"""Generate compact vector figures from the verified guard_evaluation tables."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_submission_figures import save

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs/guard_evaluation"


def attainment():
    f=pd.read_csv(OUT/"by_seed.csv")
    f=f[f.method.isin(["Fixed-DDQN","Fixed-DDQN+Guard"])]
    p=f.pivot(index=["observation","seed"],columns="method",values="passes")
    added=(p["Fixed-DDQN+Guard"]-p["Fixed-DDQN"]).unstack(0)
    fig,ax=plt.subplots(figsize=(3.35,2.38))
    fig.subplots_adjust(left=.18,right=.98,top=.89,bottom=.19)
    x=np.arange(len(added))
    for values, shift, name, color in ((added.oracle,-.155,"Oracle","#265d83"),
                                       (added.estimated,.155,"Estimated","#ae6846")):
        bars=ax.bar(x+shift,values,width=.29,color=color,label=name,zorder=3)
        ax.bar_label(bars,fmt="%d",fontsize=6.6,padding=2)
    ax.legend(loc="upper left",frameon=False,ncol=2,borderaxespad=0)
    ax.set(xlim=(-.48,len(x)-.52),ylim=(0,max(added.max())*1.24),
           xlabel="Training seed",ylabel="Additional workloads passing")
    ax.set_xticks(x,added.index.astype(str))
    ax.grid(axis="y",color="#e5eaf0",linewidth=.55)
    save(fig,"guard_attainment_by_seed")


def calibration():
    f=pd.read_csv(OUT/"reliability.csv")
    f=f[f.method.eq("Fixed-DDQN+Guard")]
    fig,ax=plt.subplots(figsize=(3.35,2.5))
    fig.subplots_adjust(left=.18,right=.98,top=.94,bottom=.22)
    for mode,color in (("oracle","#265d83"),("estimated","#ae6846")):
        g=f[f.observation.eq(mode)&f.requests.gt(0)]
        ax.plot(100*g.predicted_mean,100*g.observed_rate,color=color,marker="o",ms=3,
                label=mode.capitalize(),linewidth=1.25)
    ax.plot([0,100],[0,100],ls="--",color="#9aa6ae",lw=.8,label="Perfect calibration")
    ax.set(xlim=(0,102),ylim=(0,102),xlabel="Mean predicted miss probability (%)",
           ylabel="Observed miss rate (%)")
    ax.legend(loc="upper left",frameon=False)
    ax.grid(color="#e5eaf0",linewidth=.55)
    save(fig,"guard_risk_calibration")


if __name__=="__main__":
    attainment()
    calibration()
