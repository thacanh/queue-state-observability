def find_file(name):
    p=RAW/name
    if p.exists(): return p
    hits=list(RAW.rglob(name))
    if not hits: raise FileNotFoundError(name)
    return hits[0]

def inv_path(day): return find_file(f"invocations_per_function_md.anon.{day}.csv")

def dur_path(day): return find_file(f"function_durations_percentiles.anon.{day}.csv")

def load_duration_selected(day):
    d=pd.read_csv(dur_path(day),usecols=KEYS+QCOLS,low_memory=False)
    for c in QCOLS: d[c]=pd.to_numeric(d[c],errors="coerce")
    d=d.groupby(KEYS,as_index=False)[QCOLS].median()
    d=sel[["func_id"]+KEYS].merge(d,on=KEYS,how="left")
    q=d[QCOLS].to_numpy(float)
    d["valid"]=np.isfinite(q).all(1)&(q>0).all(1)&(np.diff(q,axis=1)>=0).all(1)
    return d[["func_id","valid"]+QCOLS]

def load_inv_selected(day):
    p=inv_path(day)
    header=pd.read_csv(p,nrows=0).columns.tolist()
    mins=sorted([c for c in header if str(c).isdigit() and 1<=int(c)<=1440],key=lambda x:int(x))
    pieces=[]
    for ch in pd.read_csv(p,usecols=KEYS+mins,chunksize=5000,low_memory=False):
        z=ch.merge(sel[["func_id"]+KEYS],on=KEYS,how="inner")
        if len(z): pieces.append(z[["func_id"]+mins])
    if not pieces: raise RuntimeError(f"No selected functions on {day}")
    x=pd.concat(pieces,ignore_index=True)
    vals=x[mins].apply(pd.to_numeric,errors="coerce").fillna(0)
    x=pd.concat([x[["func_id"]].reset_index(drop=True),vals.reset_index(drop=True)],axis=1)
    x=x.groupby("func_id",as_index=False)[mins].sum()
    x=sel[["func_id"]].merge(x,on="func_id",how="left").fillna(0)
    return x,mins

def get_duration(day):
    if day not in duration_cache: duration_cache[day]=load_duration_selected(day)
    return duration_cache[day]

def get_inv(day):
    if day not in inv_cache: inv_cache[day]=load_inv_selected(day)
    return inv_cache[day]

def sample_service(func_idx,q,rng):
    out=np.empty(len(func_idx))
    for f in np.unique(func_idx):
        m=func_idx==f
        out[m]=np.exp(np.interp(rng.random(m.sum()),QGRID,np.log(np.maximum(q[f],1e-6))))
    return out

def build_trace(day,obs_profile,seed):
    inv,mins=get_inv(day)
    dur=get_duration(day)
    counts=inv[mins].to_numpy(np.float64)
    valid=dur["valid"].to_numpy()
    counts[~valid,:]=0
    q=dur[QCOLS].to_numpy(np.float64)/1000

    per_min=counts.sum(0)
    roll=np.convolve(per_min,np.ones(WINDOW_MINUTES),mode="valid")
    start=int(np.argmax(roll))
    end=start+WINDOW_MINUTES
    weights=counts[:,start:end].T.reshape(-1)
    if weights.sum()<=0: raise RuntimeError(f"No traffic on {day}")
    weights/=weights.sum()

    rng=np.random.default_rng(seed)
    pilot_slots=rng.choice(len(weights),40000,replace=True,p=weights)
    pilot_func=pilot_slots%len(sel)
    mean_service=float(sample_service(pilot_func,q,rng).mean())

    target_n=int(round(TARGET_UTIL*CAPACITY*(WINDOW_MINUTES*60)/mean_service))
    target_n=max(30000,min(target_n,180000))

    slots=rng.choice(len(weights),target_n,replace=True,p=weights)
    minute_idx=slots//len(sel)
    func_idx=slots%len(sel)
    arrival=(minute_idx+rng.random(target_n))*60
    service=sample_service(func_idx,q,rng)

    order=np.argsort(arrival)
    arrival,func_idx,service=arrival[order],func_idx[order],service[order]

    raw_util=service.sum()/(max(arrival[-1]-arrival[0],1e-9)*CAPACITY)
    arrival=(arrival-arrival[0])*(raw_util/TARGET_UTIL)
    span=max(arrival[-1]-arrival[0],1e-9)

    p50=dict(zip(obs_profile["func_id"],obs_profile["percentile_Average_50"]/1000))
    p99=dict(zip(obs_profile["func_id"],obs_profile["percentile_Average_99"]/1000))
    funcs=sel["func_id"].to_numpy()[func_idx]
    comp=sel["complexity"].to_numpy()[func_idx]

    trace=pd.DataFrame({
        "arrival":arrival,"func_id":funcs,"complexity":comp,
        "est_p50":[p50[f] for f in funcs],"est_p99":[p99[f] for f in funcs],
        "service_demand":service,"slo":[train_slo[f] for f in funcs]
    })
    meta={
        "day":day,"window_start":start+1,"window_end":end,"requests":len(trace),
        "input_rps":len(trace)/span,"util":service.sum()/(span*CAPACITY),
        "mean_service":trace["service_demand"].mean(),"p95_service":trace["service_demand"].quantile(.95)
    }
    return trace,meta