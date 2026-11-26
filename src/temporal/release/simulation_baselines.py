import time, json, math
import numpy as np
import pandas as pd
from numba import njit

KEYS=['HashOwner','HashApp','HashFunction']
QCOLS=['percentile_Average_1','percentile_Average_25','percentile_Average_50','percentile_Average_75','percentile_Average_99']
QGRID=np.array([.01,.25,.5,.75,.99])
SPEED=np.array([1.5,1.,.7]); WORKERS=np.array([4,3,2])

def training_profiles(root):
    selection=pd.read_csv(root/'processed'/'selected_functions_d01_v2.csv').drop_duplicates(KEYS).reset_index(drop=True)
    if 'func_id' not in selection: selection['func_id']=[f'F{i:02d}' for i in range(len(selection))]
    selection=selection.iloc[:30][['func_id']+KEYS]
    if len(selection)!=30: raise ValueError('Expected 30 unique selected functions')
    all_days=[]
    for day in range(1,8):
        paths=list((root/'data').rglob(f'function_durations_percentiles.anon.d{day:02d}.csv'))
        if len(paths)!=1: raise FileNotFoundError(f'Expected one duration file for d{day:02d}, found {len(paths)}')
        d=pd.read_csv(paths[0],usecols=KEYS+QCOLS,low_memory=False)
        for col in QCOLS: d[col]=pd.to_numeric(d[col],errors='coerce')
        d=selection.merge(d.groupby(KEYS,as_index=False)[QCOLS].median(),on=KEYS,how='left')
        q=d[QCOLS].to_numpy(float)
        valid=np.isfinite(q).all(axis=1)&(q>0).all(axis=1)&(np.diff(q,axis=1)>=0).all(axis=1)
        all_days.append(d.loc[valid,['func_id']+QCOLS])
    prof=pd.concat(all_days).groupby('func_id')[QCOLS].median().reindex(selection.func_id)
    if prof.isna().any().any(): raise ValueError('Missing training profile')
    prof.index=prof.index.astype(str)
    return prof/1000

@njit
def _eval_heuristic(a,demand,slo,pred,kind,speeds,workers):
    n=a.size; m=speeds.size; width=np.max(workers)
    avail=np.full((m,width),np.inf); cur=np.zeros(m); weights=speeds*workers
    for j in range(m):
        for k in range(workers[j]): avail[j,k]=a[0]
    lat=np.empty(n); que=np.empty(n); action=np.empty(n,np.int64); finish=np.empty(n)
    for i in range(n):
        if kind==0:
            cur+=weights; j=np.argmax(cur); cur[j]-=weights.sum()
        else:
            best=1e300; j=0
            for h in range(m):
                if kind==1:
                    score=0.
                    for k in range(workers[h]): score+=max(avail[h,k]-a[i],0.)
                else:
                    free=np.min(avail[h,:workers[h]])
                    score=max(free-a[i],0.)+pred[i]/speeds[h]
                if score<best: best=score; j=h
        k=np.argmin(avail[j,:workers[j]]); start=max(a[i],avail[j,k]); end=start+demand[i]/speeds[j]
        avail[j,k]=end; lat[i]=end-a[i]; que[i]=start-a[i]; action[i]=j; finish[i]=end
    return lat,que,action,finish

def evaluate_heuristic(trace,pred,kind,name,speeds=SPEED,workers=WORKERS):
    needed=['arrival','service_demand','slo','func_id']
    if any(c not in trace for c in needed): raise ValueError('trace missing required column')
    x=trace[needed[:3]].to_numpy(float)
    if not np.isfinite(x).all() or np.any(np.diff(x[:,0])<0) or (x[:,1]<0).any() or (x[:,2]<=0).any(): raise ValueError('invalid trace')
    pred=np.asarray(pred,float)
    if pred.shape!=(len(trace),) or not np.isfinite(pred).all() or (pred<=0).any(): raise ValueError('invalid profile prediction')
    speeds=np.asarray(speeds,float); workers=np.asarray(workers,int)
    if len(speeds)!=len(workers) or np.any(speeds<=0) or np.any(workers<=0): raise ValueError('invalid topology')
    lat,queue,actions,finish=_eval_heuristic(x[:,0],x[:,1],x[:,2],pred,kind,speeds,workers)
    v=lat>x[:,2]
    result={'policy':name,'n':len(trace),'violations':int(v.sum()),'violation_rate':float(v.mean()),
            'mean_ms':float(lat.mean()*1000),'p50_ms':float(np.quantile(lat,.5)*1000),
            'p95_ms':float(np.quantile(lat,.95)*1000),'p99_ms':float(np.quantile(lat,.99)*1000),
            'queue_mean_ms':float(queue.mean()*1000),'completed_rps':float(len(trace)/(finish.max()-x[0,0])),
            'drain_s':float(finish.max()-x[-1,0])}
    return result, (lat,queue,actions)

def heuristic_grid(trace,profile,speeds=SPEED,workers=WORKERS):
    ids=trace.func_id.astype(str)
    if not ids.isin(profile.index.astype(str)).all(): raise ValueError('trace contains unknown function')
    p=profile.loc[ids,QCOLS].to_numpy(float)

    for name,index in [('est_p50',2),('est_p99',4)]:
        if name in trace and not np.allclose(trace[name],p[:,index],rtol=1e-5,atol=1e-8):
            raise ValueError(f'{name} does not match current training-only profile')
    configs=[('WeightedRR',0,p[:,2]),('LeastWorkload',1,p[:,2])]
    for q in [.5,.75,.90,.95,.99]:
        logp=np.log(p); k=np.searchsorted(QGRID,q,side='right'); k=min(max(k,1),len(QGRID)-1)
        z=(q-QGRID[k-1])/(QGRID[k]-QGRID[k-1]); pred=np.exp(logp[:,k-1]*(1-z)+logp[:,k]*z)
        configs.append((f'SED_q{q:.2f}',2,pred))
    for alpha in np.linspace(0,1,11): configs.append((f'SED_alpha{alpha:.1f}',2,(1-alpha)*p[:,2]+alpha*p[:,4]))
    rows=[]
    for name,kind,pred in configs:
        t=time.monotonic(); row,_=evaluate_heuristic(trace,pred,kind,name,speeds,workers); row['elapsed_s']=time.monotonic()-t; rows.append(row)
    result=pd.DataFrame(rows)
    result['pareto']=~np.array([any((x.p95_ms<=r.p95_ms and x.violation_rate<=r.violation_rate and (x.p95_ms<r.p95_ms or x.violation_rate<r.violation_rate)) for _,x in result.iterrows()) for _,r in result.iterrows()])
    return result

def test_simulation_baselines():
    x=pd.DataFrame({'arrival':[0.,0.,1.], 'service_demand':[1.,1.,1.], 'slo':[1.,1.,1.], 'func_id':['a']*3})
    a,q,act,f=_eval_heuristic(x.arrival.to_numpy(),x.service_demand.to_numpy(),x.slo.to_numpy(),np.ones(3),2,np.array([1.]),np.array([1]))
    np.testing.assert_allclose(a,[1,2,2]); np.testing.assert_allclose(q,[0,1,1]); assert (a>x.slo).sum()==2
    for mode in (0,1,2):
        row,_=evaluate_heuristic(x,np.ones(3),mode,'test'); assert row['n']==3
    print('PASS simulation: FIFO, strict threshold, heuristic routes')
