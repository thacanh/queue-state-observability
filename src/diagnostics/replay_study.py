"""Run predeclared frozen-checkpoint comparisons in an isolated results tree."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from hashlib import file_digest, sha256
from itertools import product
from pathlib import Path
import argparse
import csv
import json
import os
import sys
import time


BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[1]
TEMPORAL = ROOT / "outputs/temporal_evaluation"
MANUSCRIPT = ROOT / "outputs/ieee_access_temporal"
ARTIFACT = MANUSCRIPT / "queue-replay-v1.2-temporal-review"
HISTORICAL = ROOT / "outputs/estimated_ddqn/validation/paired.csv"
MODES = ("oracle", "estimated")
GRID = (.0025, .005, .01)
KEYS = ("p50", "p99", "unc", "ia", "res")
LIMITS = {"T1": 600, "T2": 600, "T3": 900, "T4": 900, "T5": 900,
          "T6": 600, "T7": 900, "T8": 600}
RAM = {"T6": 4_000_000_000, "T7": 4_000_000_000}


def digest(path):
    with Path(path).open("rb") as f:
        return file_digest(f, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fingerprint(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def lock(path, protocol):
    full = protocol | {"protocol_sha256": fingerprint(protocol)}
    if path.exists():
        if read(path) != full:
            raise ValueError(f"Incompatible protocol in {path}")
    else:
        write(path, full)
    return full["protocol_sha256"]


def csv_rows(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def utc():
    return datetime.now(timezone.utc).isoformat()


def dependencies():
    cache = BASE / "numba_cache"
    cache.mkdir(exist_ok=True, parents=True)
    sys.dont_write_bytecode = True
    os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                      MKL_THREADING_LAYER="SEQUENTIAL", NUMBA_NUM_THREADS="1", NUMBA_CACHE_DIR=str(cache),
                      CUDA_VISIBLE_DEVICES="-1")
    sys.path.insert(0, str(ARTIFACT / "code/release"))
    global np, pd, torch, psutil, replay, results, XCOLS, SPEED, WORKERS, typeof, threadpool_limits, threadpool_info
    import numpy as np
    import pandas as pd
    import torch
    import psutil
    from numba import typeof
    from threadpoolctl import threadpool_limits, threadpool_info
    from simulation_baselines import SPEED, WORKERS
    from routing_simulator import replay
    from replay_io import results, XCOLS
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


class Frozen:
    def __init__(self, transfer=False):
        self.transfer = transfer
        self.protocol = read(TEMPORAL / "protocol.json")
        self.index = read(ARTIFACT / "package_manifest.json")["files"]
        self.location = ARTIFACT / "evidence/inputs" / ("2021_reserved" if transfer else "2019_R0")
        registry = read(self.location / "manifest.json")
        assert registry["protocol_sha256"] == self.protocol["protocol_sha256"]
        self.traces = registry["traces"]
        self.window_keys = ["elapsed07", "elapsed08"]
        self.later_keys = [f"d{d}_r{i:02d}" for d, i in product((12, 13, 14), (0, 1))]
        self.phase = "transfer" if transfer else "B"
        self.archived = {(r["key"], int(r["seed"]), r["arm"]): r
                         for r in csv_rows(TEMPORAL / self.phase / "replays.csv")}
        self.bank_path = self.location / "prefix_bank.npz" if transfer else ARTIFACT / "inputs/guard_demand_bank_2019.npz"
        self.bank_sha = digest(self.bank_path)
        assert self.bank_sha == self.index[self.bank_path.relative_to(ARTIFACT).as_posix()]["sha256"]
        with np.load(self.bank_path, allow_pickle=False) as npz:
            self.bank = tuple(npz[k] for k in ("bank", "offsets", "prefix", "means"))
            self.names = {f: j for j, f in enumerate(npz["functions"].tolist())} if "functions" in npz else None

    def workload(self, key):
        record = self.traces[key]
        path = self.location / record["filename"]
        assert digest(path) == record["file_sha256"]
        if self.transfer:
            with np.load(path, allow_pickle=False) as f:
                x, ids = f["values"], f["ids"]
            sha = sha256(x.tobytes() + ids.tobytes()).hexdigest()
        else:
            frame = pd.read_csv(path, dtype=record["dtypes"], float_precision="round_trip")
            sha = sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest()
            x = np.ascontiguousarray(frame[XCOLS].to_numpy(np.float64))
            ids = frame["func_id"].astype(str).map(self.names).to_numpy(np.int64)
        assert sha == record["trace_sha256"] and len(x) == record["meta"]["requests"]
        return x, ids, x[:, 1].copy()

    def checkpoint(self, mode, seed):
        path = ARTIFACT / f"inputs/checkpoints/{mode}/seed{seed}.pt"
        expected = self.protocol["models"][f"{mode}/{seed}"]
        return model(path, expected["sha256"], expected.get("identity"))


def model(path, expected_sha=None, expected_identity=None):
    sha = digest(path)
    if expected_sha is not None and expected_sha != sha:
        raise ValueError(f"Checkpoint SHA mismatch: {path}")
    record = torch.load(path, map_location="cpu", weights_only=True)
    if expected_identity is not None and record.get("identity") != expected_identity:
        raise ValueError(f"Checkpoint identity mismatch: {path}")
    scales = np.asarray([record["scales"][feature] for feature in KEYS], dtype=np.float64)
    weights = tuple(np.ascontiguousarray(record["model"][f"net.{i}.{p}"].numpy(), dtype=np.float32)
                    for i in (0, 2, 4) for p in ("weight", "bias"))
    return scales, weights, sha


def archived_outcome(frozen, key, seed, mode, guard, sha):
    arm = f"{mode}_{'on' if guard else 'off'}"
    record = frozen.archived[key, seed, arm]
    assert record["trace_sha256"] == frozen.traces[key]["trace_sha256"] and record["model_sha256"] == sha
    omit = {"key", "day", "reconstruction_seed", "window_start_minute", "seed", "arm",
            "trace_sha256", "model_sha256", "elapsed_s"}
    integers = {"n", "violations", "overrides", "direct_saves", "direct_losses", "backlog_max"}
    return {k: (v == "True" if k == "met_budget" else int(float(v)) if k in integers else float(v))
            for k, v in record.items() if k not in omit and v != ""}


def check_outcome(actual, reference):
    for k, v in actual.items():
        if k not in reference:
            continue
        ref = reference[k]
        if isinstance(v, (int, bool)):
            same = v == ref
        else:
            same = np.isclose(v, ref, atol=1e-6, rtol=0)
        if not same:
            raise ValueError(f"Replay mismatch in {k}: {v} vs {ref}")


def arguments(inputs, checkpoint, guard, bank, threshold=(.005, .005), scale=1., method=0):
    return (*inputs, SPEED, WORKERS, *checkpoint[:2], method, guard, True, scale, *bank, *threshold)


def base_protocol(phase, frozen, keys, seeds, models, analysis, extra):
    return {
        "phase": phase, "parent": frozen.protocol["protocol_sha256"],
        "runtime_budget_s": LIMITS[phase], "workers": 4, "native_threads": 1,
        "software": {"numpy": np.__version__, "pandas": pd.__version__, "torch": torch.__version__},
        "bank_sha256": frozen.bank_sha,
        "checkpoints": {f"{mode}/{seed}": item[2] for (mode, seed), item in models.items()},
        "workloads": {key: {k: frozen.traces[key][k] for k in ("day", "trace_sha256", "file_sha256")}
                      for key in keys},
        "seeds": list(seeds), "budget": .005, "load": .6,
        "provenance": "Later Azure 2019 days have unverified prior use; Azure 2021 elapsed 7–8 are fixed diagnostic windows.",
        "runner": digest(Path(__file__)),
        "simulator": digest(ARTIFACT / "code/release/routing_simulator.py"),
        "analysis": analysis, "plan": extra,
    }


def run_shards(phase, protocol, keys, tasks, evaluate, warm, expected_new, expected_cached=0):
    start, cpu = time.perf_counter(), time.process_time()
    folder = BASE / phase
    key = lock(folder / "protocol.json", protocol)
    limit = LIMITS[phase]
    memory_limit = RAM.get(phase, 2_000_000_000)
    shards = folder / "shards"
    shards.mkdir(parents=True, exist_ok=True)
    all_rows = []
    status = "FAILED"
    usage = psutil.Process()
    times = {}
    try:
        with threadpool_limits(limits=1):
            pools = {r["internal_api"]: r["num_threads"] for r in threadpool_info()}
            if any(value != 1 for value in pools.values()):
                raise RuntimeError(f"Unexpected numerical threads {pools}")
            warm()
            times["warmup_s"] = time.perf_counter() - start
            with ThreadPoolExecutor(max_workers=4) as pool:
                for name in keys:
                    if time.perf_counter() - start > limit - 8:
                        raise TimeoutError("Stopped before the next workload at its time limit")
                    if usage.memory_info().rss > memory_limit:
                        raise MemoryError("Stopped before the next workload at its RAM limit")
                    dest = shards / f"{name}.json"
                    if dest.exists():
                        cached = read(dest)
                        if cached["protocol"] != key or cached["rows_sha256"] != fingerprint(cached["rows"]):
                            raise ValueError(f"Shard integrity changed: {dest}")
                        rows = cached["rows"]
                    else:
                        rows = list(pool.map(evaluate, tasks(name)))
                        write(dest, {"protocol": key, "rows_sha256": fingerprint(rows), "rows": rows})
                    all_rows.extend(rows)
                    print(f"{phase} {name}: {len(all_rows)}/{expected_new+expected_cached}", flush=True)
        assert len(all_rows) == expected_new + expected_cached
        assert sum(row.get("reference_reused", False) for row in all_rows) == expected_cached
        output = folder / "replays.csv"
        temp = output.with_suffix(".tmp")
        pd.DataFrame(all_rows).to_csv(temp, index=False)
        temp.replace(output)
        status = "COMPLETE"
    except BaseException as exc:
        times["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        mem = usage.memory_info()
        write(folder / "execution.json", {"phase": phase, "protocol": key, "status": status,
              "wall_s": time.perf_counter() - start, "cpu_s": time.process_time() - cpu,
              "peak_rss_bytes": getattr(mem, "peak_wset", mem.rss), "gpu": None,
              "recorded_rows": len(all_rows), "reused_rows": sum(row.get("reference_reused", False) for row in all_rows),
              "shards": {p.name: digest(p) for p in sorted(shards.glob("*.json"))}, "completed_utc": utc(),
              "replays_sha256": digest(output) if status == "COMPLETE" else None, **times})


def sweep_t1():
    frozen = Frozen()
    keys, seeds = frozen.later_keys, (42, 43)
    models = {(mode, seed): frozen.checkpoint(mode, seed) for mode, seed in product(MODES, seeds)}
    settings = ((False, .005, .005),) + tuple((True, t, d) for t, d in product(GRID, repeat=2))
    protocol = base_protocol(
        "T1", frozen, keys, seeds, models,
        "Paired by checkpoint seed and workload, show all levels and per-day; compare to guarded estimated "
        "default at P95 within ±1%, ±2% or no greater. Descriptive only; no threshold selection.",
        {"thresholds": [list(x) for x in settings], "policy_view": "estimated", "guard_view": "estimated",
         "expected_new": 192, "expected_reused": 48, "reference_csv": digest(TEMPORAL / "B/replays.csv")})

    def tasks(key):
        inputs = frozen.workload(key)
        return [(key, inputs, mode, seed, *setting)
                for (mode, seed), setting in product(models, settings)]

    def evaluate(task):
        key, inputs, mode, seed, guard, trigger, reduction = task
        ck = models[mode, seed]
        ref = not guard or (trigger, reduction) == (.005, .005)
        stats = (archived_outcome(frozen, key, seed, mode, guard, ck[2]) if ref else
                 results(inputs[0], replay(*arguments(inputs, ck, guard, frozen.bank, (trigger, reduction))))[0])
        return dict(key=key, day=frozen.traces[key]["day"], seed=seed, mode=mode, guard=guard,
                    trigger=trigger, reduction=reduction, reference_reused=ref,
                    model_sha256=ck[2], trace_sha256=frozen.traces[key]["trace_sha256"], **stats)

    def warm():
        items = tasks(keys[0])
        key, inputs, mode, seed, *_ = items[0]
        ck = models[mode, seed]
        params = arguments(inputs, ck, True, frozen.bank)
        replay.compile(tuple(typeof(p) for p in params))
        actual = results(inputs[0], replay(*params))[0]
        check_outcome(actual, archived_outcome(frozen, key, seed, mode, True, ck[2]))

    run_shards("T1", protocol, keys, tasks, evaluate, warm, 192, 48)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=["T1"])
    args = cli.parse_args()
    dependencies()
    sweep_t1()
