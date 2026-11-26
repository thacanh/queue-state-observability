"""Necessary deadline-miss bounds."""

import numpy as np
import pandas as pd


def risk_mark_bins(trace, bins=180):
    out=trace.copy(); a=out.arrival.to_numpy(float); span=a[-1]-a[0]
    if len(out)<2 or span<=0 or np.any(np.diff(a)<0): raise ValueError('invalid risk trace')
    out['_risk_bin']=np.minimum(((a-a[0])/span*bins).astype(int),bins-1)
    out['_risk_request']=np.arange(len(out))
    return out


def risk_deadline_necessary_bound(trace, speeds=(1.5,1.,.7), workers=(4,3,2)):
    """A conservative NECESSARY bound on missed deadlines for this finite trace.

    For jobs released no earlier than u with deadlines <=v, any on-time subset
    consumes <= C*(v-u) service-work. Removing the largest demands first gives a
    necessary miss count. Ignore other jobs, pool all server capacity and allow
    preemption/migration: this relaxation can only help scheduling. Keep one
    maximum count per disjoint job-bin; never add overlapping window counts.
    Intrinsically impossible jobs are counted separately and excluded from the
    capacity sets, so the final union has no double counting.
    """
    a=trace.arrival.to_numpy(float); d=trace.service_demand.to_numpy(float)
    slo=trace.slo.to_numpy(float); labels=trace['_risk_bin'].to_numpy(int)
    if not np.isfinite(np.c_[a,d,slo]).all() or np.any(d<0) or np.any(slo<=0):
        raise ValueError('invalid deadline input')
    C=float(np.dot(speeds,workers)); fastest=float(max(speeds))
    intrinsic=d/fastest>slo
    rows=[]; total=int(intrinsic.sum())
    for label in np.unique(labels):
        ix=np.flatnonzero((labels==label)&~intrinsic)
        if len(ix)==0: continue
        u=float(a[ix].min()); deadlines=a[ix]+slo[ix]

        candidates=np.unique(deadlines) if len(ix)<=32 else np.unique(np.quantile(deadlines,[.01,.05,.1,.25,.5,.75,.9,.95,.99,1.]))
        best=0; best_v=u; chosen_n=0; chosen_work=0.
        for v in candidates:
            demands=d[ix[deadlines<=v]]; work=float(demands.sum()); capacity=C*(float(v)-u)
            excess=work-capacity
            if excess<=1e-9*max(work,capacity,1.): continue
            cumulative=np.cumsum(np.sort(demands)[::-1])
            count=int(np.searchsorted(cumulative,excess-1e-9*max(work,capacity,1.),side='left'))+1
            if count>best: best=count;best_v=float(v);chosen_n=len(demands);chosen_work=work
        total+=best
        rows.append({'bin':int(label),'necessary_queue_misses':best,'u':u,'v':best_v,
            'window_jobs':chosen_n,'window_demand':chosen_work,'pooled_window_capacity':C*(best_v-u)})
    return {'necessary_misses':total,'necessary_violation_floor':total/len(trace),
            'fastest_intrinsic_misses':int(intrinsic.sum()),
            'interpretation':'necessary finite-trace capacity/deadline relaxation; never a sufficient feasibility test'}, pd.DataFrame(rows)
