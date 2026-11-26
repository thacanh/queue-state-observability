"""Optional replay from the author-review artifact's frozen array and checkpoint files."""

from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
import csv
from hashlib import file_digest, sha256
import json
import os
from pathlib import Path
import sys


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def parse():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--phase", choices=("B", "D", "transfer"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse()
    root = args.artifact.resolve()
    output = args.output.resolve()
    if output == root or output.is_relative_to(root):
        raise ValueError("Use a separate output directory outside the frozen artifact")
    output.mkdir(exist_ok=False, parents=True)
    cache = output / "numba_cache"
    cache.mkdir()
    sys.dont_write_bytecode = True
    os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                      MKL_THREADING_LAYER="SEQUENTIAL", NUMBA_NUM_THREADS="1", NUMBA_CACHE_DIR=str(cache),
                      CUDA_VISIBLE_DEVICES="-1")
    sys.path.insert(0, str(root / "code/release"))

    import numpy as np
    import pandas as pd
    import torch
    from numba import typeof
    from threadpoolctl import threadpool_limits
    from crossed_router import crossed_kernel
    from replay_io import XCOLS, results
    from replay_crossed import queue_metrics
    from routing_simulator import replay
    from simulation_baselines import SPEED, WORKERS

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    evidence = root / "evidence"
    protocol = read(evidence / "protocol.json")
    key = protocol.pop("protocol_sha256")
    assert sha256(json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()).hexdigest() == key
    stage = protocol["phases"][args.phase]
    seeds = protocol["training_seeds"]
    source = evidence / "inputs" / ("2021_reserved" if args.phase == "transfer" else "2019_R0")
    index = read(source / "manifest.json")
    assert index["protocol_sha256"] == key
    artifact_index = read(root / "package_manifest.json")["files"]
    model_keys = ("p50", "p99", "unc", "ia", "res")
    thresholds = protocol["fixed"]
    shape = (len(seeds), len(stage["arms"]))
    models = {}
    for mode in (("oracle",) if args.phase == "D" else ("oracle", "estimated")):
        for seed in seeds:
            path = root / f"inputs/checkpoints/{mode}/seed{seed}.pt"
            expected = protocol["models"][f"{mode}/{seed}"]
            if digest(path) != expected["sha256"]:
                raise ValueError(f"Checkpoint differs from locked protocol: {path}")
            record = torch.load(path, map_location="cpu", weights_only=True)
            if mode == "estimated" and record["identity"] != expected["identity"]:
                raise ValueError("Estimated checkpoint identity differs from protocol")
            scales = np.asarray([record["scales"][feature] for feature in model_keys])
            weights = tuple(np.ascontiguousarray(record["model"][f"net.{i}.{p}"].numpy(), dtype=np.float32)
                            for i in (0, 2, 4) for p in ("weight", "bias"))
            models[mode, seed] = scales, weights, expected["sha256"]
    bank_path = source / "prefix_bank.npz" if args.phase == "transfer" else root / "inputs/guard_demand_bank_2019.npz"
    relative = bank_path.relative_to(root).as_posix()
    assert digest(bank_path) == artifact_index[relative]["sha256"]
    with np.load(bank_path, allow_pickle=False) as saved:
        bank = tuple(saved[name] for name in ("bank", "offsets", "prefix", "means"))
        names = {f: j for j, f in enumerate(saved["functions"].tolist())} if "functions" in saved else None

    def workload(rec):
        data = source / rec["filename"]
        if digest(data) != rec["file_sha256"]:
            raise ValueError(f"Frozen workload bytes differ: {data}")
        if args.phase == "transfer":
            with np.load(data, allow_pickle=False) as inp:
                x, ids = inp["values"], inp["ids"]
            fingerprint = sha256(x.tobytes()+ids.tobytes()).hexdigest()
        else:
            frame = pd.read_csv(data, dtype=rec["dtypes"], float_precision="round_trip")
            fingerprint = sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest()
            x = np.ascontiguousarray(frame[XCOLS].to_numpy(np.float64))
            ids = frame["func_id"].astype(str).map(names).to_numpy(np.int64)
        assert len(x) == rec["meta"]["requests"] and fingerprint == rec["trace_sha256"]
        return x, ids, x[:, 1].copy()

    def params(inputs, arm, seed):
        if args.phase == "D":
            mode = "oracle"
            policy, guard_view = arm
        else:
            mode, flag = arm.split("_")
            policy, guard_view = "E", ("E" if flag == "on" else "-")
        values = models[mode, seed]
        tail = (thresholds["guard_trigger"], thresholds["guard_improvement"])
        if args.phase != "B":
            tail = ((guard_view if guard_view != "-" else policy) == "E", *tail)
        return (*inputs, SPEED, WORKERS, *values[:2], 0, guard_view != "-", policy == "E",
                thresholds["residual_scale"], *bank, *tail), values[2]

    kernel = replay if args.phase == "B" else crossed_kernel()

    def run(task):
        inputs, rec, seed, arm = task
        call, checkpoint = params(inputs, arm, seed)
        response = kernel(*call)
        outcome = results(inputs[0], response[:, :10])[0]
        if args.phase == "transfer":
            outcome.update(queue_metrics(inputs[0], response))
        return dict(key=rec["key"], day=rec["day"], seed=seed, arm=arm,
                    trace_sha256=rec["trace_sha256"], model_sha256=checkpoint, **outcome)

    indexed = {}
    with (evidence / args.phase / "replays.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            lookup = row["key"], int(row["seed"]), row["arm"]
            assert lookup not in indexed
            indexed[lookup] = row
    first = workload(index["traces"][stage["workloads"][0]["key"]])
    initial, _ = params(first, stage["arms"][0], seeds[0])
    all_rows = []
    with threadpool_limits(limits=1):
        kernel.compile(tuple(typeof(value) for value in initial))
        with ThreadPoolExecutor(max_workers=4) as pool:
            for requested in stage["workloads"]:
                rec = index["traces"][requested["key"]]
                inputs = workload(rec)
                items = [(inputs, rec, seed, arm) for seed in seeds for arm in stage["arms"]]
                rows = list(pool.map(run, items))
                assert len(rows) == shape[0] * shape[1]
                pending = output / (rec["key"] + ".tmp")
                pending.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
                pending.replace(output / (rec["key"] + ".json"))
                all_rows.extend(rows)
    assert len(all_rows) == len(indexed) == stage["runs"]
    mismatches = []
    for row in all_rows:
        expected = indexed[row["key"], row["seed"], row["arm"]]
        for field, value in row.items():
            if field in {"key", "day", "seed", "arm", "trace_sha256", "model_sha256",
                         "n", "violations", "met_budget", "overrides", "backlog_max"}:
                same = str(value) == expected[field]
            else:
                same = np.isclose(value, float(expected[field]), rtol=0, atol=1e-6)
            if not same:
                mismatches.append(dict(workload=row["key"], seed=row["seed"], arm=row["arm"], metric=field,
                                       value=value, archived=expected[field]))
    report = dict(protocol_sha256=key, phase=args.phase, replay_count=len(all_rows), mismatches=mismatches)
    (output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if mismatches:
        raise RuntimeError(f"{len(mismatches)} differences, see comparison.json")
    print(f"Verified {len(all_rows)} reproduced rows against {args.phase} evidence.")


if __name__ == "__main__":
    main()
