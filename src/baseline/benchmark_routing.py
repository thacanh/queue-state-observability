"""Time actual decisions inside a single-host TCP/asyncio emulator."""

import ast
import asyncio
import inspect
import os
import platform
import time

import replay_io as base
import numpy as np
import pandas as pd
from numba import njit
from threadpoolctl import threadpool_limits
from tqdm import tqdm

from routing_simulator import decide, estimated_calendar, survival


def timed_selector():
    fn = ast.parse(inspect.getsource(decide.py_func)).body[0]
    fn.decorator_list = []
    index = next(i for i, item in enumerate(fn.body)
                 if isinstance(item, ast.Assign) and ast.unparse(item.targets[0]) == "original_risk")
    fn.body[index:index] = ast.parse(
        "if not guard and method != 2:\n    return proposal, proposal, 0., 0., wait[proposal]"
    ).body
    namespace = {"np": np, "survival": survival}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])),
                 "routing without calibration instrumentation", "exec"), namespace)
    return njit(namespace["decide"])


ARM = [("Fixed-DDQN", 0, False, "p50"), ("Fixed-DDQN+Guard", 0, True, "p50"),
       ("RiskOnly", 2, False, "p50"), ("SED-P99", 1, False, "p99"),
       ("SED-P99+Guard", 1, True, "p99")]


async def trial(x, ids, preds, tag, model, selector, bank_data, repeat, scale=.01):
    name, method, guard, key = tag
    bank, offsets, prefix, means = bank_data
    scales, weights, model_hash = model
    speed, workers = base.SPEED, base.WORKERS
    heads = np.full((len(workers), np.max(workers)), -1)
    starts = np.zeros(len(x))
    pending = np.zeros(len(workers))
    availability = np.full(heads.shape, np.inf)
    queues = [asyncio.Queue() for _ in workers]
    servers, worker_tasks = [], []

    async def accept(reader, writer, j):
        try:
            tag, demand = (await asyncio.wait_for(reader.readline(), 10.)).decode().split()
            finished = asyncio.get_running_loop().create_future()
            await queues[j].put((int(tag), float(demand), finished))
            await asyncio.wait_for(finished, 45.)
            writer.write(b"OK\n")
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    async def worker(j, k):
        while True:
            index, demand, finished = await queues[j].get()
            pending[j] -= means[ids[index]]
            starts[index] = (time.perf_counter()-epoch) / scale
            heads[j,k] = index
            await asyncio.sleep(scale*demand/speed[j])
            heads[j,k] = -1
            if not finished.done():
                finished.set_result(None)

    for j in range(len(workers)):
        async def wrapped(reader, writer, j=j):
            await accept(reader, writer, j)
        servers.append(await asyncio.start_server(wrapped, "127.0.0.1", 0))
        worker_tasks.extend(asyncio.create_task(worker(j,k)) for k in range(workers[j]))
    ports = [s.sockets[0].getsockname()[1] for s in servers]
    epoch = time.perf_counter()
    last = 0.
    timing = np.empty((len(x), 4))
    overrides = 0

    async def request(i):
        nonlocal last, overrides
        await asyncio.sleep(max(0., scale*(x[i, 0]-x[0,0])-(time.perf_counter()-epoch)))
        received = time.perf_counter_ns()
        now = (time.perf_counter()-epoch)/scale
        estimated_calendar(now, speed, workers, heads, starts, ids, pending,
                           bank, offsets, prefix, means, 1., availability)
        after_state = time.perf_counter_ns()
        group = ids[i]
        action, proposal, _, _, _ = selector(
            now, last, x[i, 1], x[i, 2], preds[key][i], x[i, 4],
            bank[offsets[group]:offsets[group+1]], availability, speed, workers,
            scales, weights, method, guard, .005, .005)
        pending[action] += means[group]
        last = now
        overrides += (action != proposal) and i >= 100
        after_route = time.perf_counter_ns()
        reader, writer = await asyncio.open_connection("127.0.0.1", ports[action])
        try:
            writer.write(f"{i} {x[i, 3]}\n".encode())
            await writer.drain()
            if await asyncio.wait_for(reader.readline(), 50.) != b"OK\n":
                raise RuntimeError("Backend response missing")
        finally:
            writer.close()
            await writer.wait_closed()
        timing[i] = [after_state-received, after_route-after_state, after_route-received,
                     time.perf_counter_ns()-received]

    try:
        await asyncio.gather(*(request(i) for i in range(len(x))))
    finally:
        for s in servers:
            s.close()
        await asyncio.gather(*(s.wait_closed() for s in servers))
        for task in worker_tasks:
            task.cancel()
        await asyncio.gather(*worker_tasks, return_exceptions=True)
    q = np.quantile(timing[100:] / 1000, [.5,.95,.99], axis=0)
    row = {"method": name, "repeat": repeat, "n": len(x)-100, "warmup":100, "time_scale":scale,
           "overrides":overrides, "model_sha256":model_hash, "throughput_req_per_s":len(x)/(time.perf_counter()-epoch)}
    for i,label in enumerate(("state_us","routing_after_state_us","per_decision_us","loopback_e2e_us")):
        row.update({f"{label}_p{p}": float(q[j,i]) for j,p in enumerate((50,95,99))})
    return row


def main():
    output_dir = base.ROOT / "regenerated/timing"
    output_dir.mkdir(parents=True, exist_ok=True)
    base.setup()
    names, bank, offsets, prefix, means, _, dummy, models, _ = base.CONTEXT
    day, seed = base.load_json(base.BANK_PATH/"protocol.json")["plan"][0]
    frame, trace_sha = base.read_workload(day, seed, base.load_json(base.BANK_PATH/"inputs/manifest.json"))
    x, ids, pred = base.workload_arrays(frame.iloc[:600])
    selector = timed_selector()
    start = time.perf_counter()
    with threadpool_limits(limits=1):

        cap = np.zeros((3,4))
        estimated_calendar(0., base.SPEED, base.WORKERS, np.full((3,4),-1),
                           np.zeros(len(x)), ids, np.zeros(3), bank, offsets, prefix, means, 1., cap)
        model = models["fixed",42]
        for tag in ARM:
            _, method, guard, key = tag
            sc,w,_ = model if method == 0 else dummy
            s = bank[offsets[ids[0]]:offsets[ids[0]+1]]
            args = (0.,0.,x[0,1],x[0,2],pred[key][0],x[0,4],s,cap,base.SPEED,base.WORKERS,sc,w,method,guard,.005,.005)
            assert selector(*args)[:2] == decide(*args)[:2]
        jit_time = time.perf_counter()-start
        rows = []
        for repeat in tqdm(range(3), desc="TCP loopback repetitions"):
            for tag in ARM:
                chosen = model if tag[1] == 0 else dummy
                row = asyncio.run(trial(x,ids,pred,tag,chosen,selector,(bank,offsets,prefix,means),repeat))
                rows.append(row)
    data = pd.DataFrame(rows)
    assert len(data) == 15 and data.n.eq(500).all()
    data.to_csv(output_dir / "loopback.csv",index=False)
    base.publish_json(output_dir / "overhead_manifest.json",
                      {"source_hash":base.digest(__file__), "trace_sha256":trace_sha,
                       "train_sha256":base.digest(base.TRAIN), "model_sha256":model[2],
                       "scope":"Python/Numba, single-host TCP, wall-clock worker starts and completions, scaled service; not cluster deployment",
                       "jit_initial_calls_seconds":jit_time,"num_functions":len(names),
                       "samples":len(bank),"guard_sorted_bank_bytes":bank.nbytes+offsets.nbytes,
                       "state_extra_bank_bytes":prefix.nbytes+means.nbytes,
                       "cpu":platform.processor(),"logical_cpus":os.cpu_count(),
                       "rows":len(data), "output_sha256":base.digest(output_dir / "loopback.csv")})


if __name__ == "__main__":
    main()
