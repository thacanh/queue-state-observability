"""Matched-snapshot CPU timing and empirical CDF lookup scaling."""

import time

import replay_io as base
import numpy as np
import pandas as pd
from numba import njit
from threadpoolctl import threadpool_limits

from benchmark_routing import timed_selector
from routing_simulator import survival


@njit
def batch_lookups(bank, values, servers):
    total=0.
    for x in values:
        for j in range(servers):
            total+=survival(bank,x+j*.01)
    return total


def main():
    output_dir = base.ROOT / "regenerated/timing"
    output_dir.mkdir(parents=True, exist_ok=True)
    base.setup()
    _,bank,offsets,_,_,_,dummy,models,_=base.CONTEXT
    day,seed=base.load_json(base.BANK_PATH/"protocol.json")["plan"][0]
    frame,source_hash=base.read_workload(day,seed,base.load_json(base.BANK_PATH/"inputs/manifest.json"))
    x,ids,_=base.workload_arrays(frame.iloc[:2000])
    avail=np.zeros((3,4))
    states=np.empty((len(x),3,4))
    for i,row in enumerate(x):
        states[i]=avail
        a,p,_,work,_=row
        pred=[max(0.,avail[j,:w].min()-a)+p/base.SPEED[j] for j,w in enumerate(base.WORKERS)]
        j=np.argmin(pred)
        slot=np.argmin(avail[j,:base.WORKERS[j]])
        avail[j,slot]=max(a,avail[j,slot])+work/base.SPEED[j]
    selector=timed_selector()
    matched=[]
    with threadpool_limits(limits=1):
        for method,nickname,model in [(0,"Fixed-DDQN",models["fixed",42]),(1,"SED-P99",dummy)]:
            scale,weights,sha=model
            for i,row in enumerate(x):
                a,p,q,_,deadline=row
                last=x[max(0,i-1),0]
                g=ids[i]
                samples=bank[offsets[g]:offsets[g+1]]
                args=(a,last,p,q,q,samples,states[i],base.SPEED,base.WORKERS,scale,weights,method)
                args=args[:5]+(deadline,)+args[5:]
                order=(False,True) if i%2==0 else (True,False)
                elapsed={}
                override=False
                for guard in order:
                    start=time.perf_counter_ns()
                    action,proposal,*_=selector(*args,guard,.005,.005)
                    elapsed[guard]=(time.perf_counter_ns()-start)/1000
                    override|=action!=proposal
                if i>=100:
                    matched.append(dict(method=nickname,request_index=i,route_us=elapsed[False],
                                        guard_route_us=elapsed[True],guard_increment_us=elapsed[True]-elapsed[False],
                                        override=override,model_sha256=sha,trace_sha256=source_hash))
        rng=np.random.default_rng(24601)
        queries=rng.uniform(.01,4.,size=30_000)
        _=batch_lookups(np.linspace(.01,4.,256),queries[:8],3)
        scales=[]
        for n in (256,4096,65536):
            values=np.linspace(.01,4.,n)
            for servers in (3,5,8):
                for repeat in range(5):
                    start=time.perf_counter_ns()
                    total=batch_lookups(values,queries,servers)
                    elapsed=time.perf_counter_ns()-start
                    scales.append(dict(samples_per_function=n,servers=servers,repeat=repeat,
                                       decisions=len(queries),amortized_lookup_us=elapsed/(1000*len(queries)),
                                       checksum=total))
    pd.DataFrame(matched).to_csv(output_dir / "decision_timing_matched.csv",index=False)
    pd.DataFrame(scales).to_csv(output_dir / "cdf_scaling.csv",index=False)


if __name__=="__main__":
    main()
