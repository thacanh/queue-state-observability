"""Train the fixed-penalty control with estimated queue observations."""

import argparse
import ast
import hashlib
import json
from functools import partial
from pathlib import Path
from time import monotonic

import numpy as np
import numba
import pandas as pd
import torch

import estimated_queue
import rl_core_execution
import rl_pilot
import routing_simulator
from deadline_guard import _guard_service_bank
from estimated_queue import EstimatedQueue
from rl_core_execution import atomic_torch_save, train_dynamic
from rl_pilot import prepare_stage2_config


def sha(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def save_metadata(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def training_source():
    modules = (estimated_queue, rl_core_execution, rl_pilot, routing_simulator)
    source = [ast.dump(ast.parse(Path(m.__file__).read_text(encoding="utf-8")))
              for m in modules]
    source.extend(ast.dump(ast.parse(Path(path).read_text(encoding="utf-8")))
                  for path in (__file__, Path(routing_simulator.__file__).with_name("deadline_guard.py")))
    return hashlib.sha256(json.dumps(source).encode()).hexdigest()


def require_digest(path, expected):
    if not path.is_file():
        raise FileNotFoundError(path)
    if sha(path) != expected:
        raise ValueError(f"Input differs from the fixed-penalty reference: {path.name}")


def source_inputs(root, registry_path, seeds):
    records = json.loads(registry_path.read_text(encoding="utf-8"))["jobs"]
    base = root / "multiday_final_5seeds"
    train, validation = (base / "traces" / name
                         for name in ("train_d01_d07.csv", "val_d08_d09.csv"))
    inputs = {}
    for seed in seeds:
        record = records[f"fixed_seed{seed}_lambda2"]["record"]
        expected = record["input_sha256"]
        stage1 = base / "models" / f"latency_ddqn_seed{seed}.pt"
        require_digest(stage1, expected["stage1"])
        inputs[seed] = (stage1, record)
    for name, path in (("train", train), ("validation", validation)):
        digests = {record["input_sha256"][name] for _, record in inputs.values()}
        if len(digests) != 1:
            raise ValueError(f"Inconsistent historical digest: {name}")
        require_digest(path, digests.pop())
    return train, validation, inputs


def train_control(root, registry, output, seeds, device="cuda",
                  observation="estimated", stop_after_batches=None):
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Colab did not allocate a CUDA runtime")
    train_path, val_path, jobs = source_inputs(root, registry, seeds)
    train, val = pd.read_csv(train_path), pd.read_csv(val_path)
    if (len(train), len(val)) != (868593, 242585):
        raise ValueError("Unexpected historical training or validation row count")
    names, bank, offsets = _guard_service_bank(train_path)
    factory = partial(EstimatedQueue, names=names, bank=bank, offsets=offsets)
    runtime = {
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "numpy": np.__version__,
        "numba": numba.__version__,
        "pandas": pd.__version__,
    }
    code = training_source()
    for seed in seeds:
        stage1, oracle = jobs[seed]
        config = prepare_stage2_config(seed, "fixed", 2, device, "pending")
        config.update(checkpoint_every=1000, amp_init_scale=1024., amp_growth_interval=1_000_000_000)
        identity = {
            "source": code, "runtime": runtime, "config": config.copy(),
            "inputs": oracle["input_sha256"], "observation": observation,
            "residual_scale": 1., "guard_in_training": False,
        }
        config["identity"] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        directory = output / f"seed-{seed}-{config['identity'][:16]}"
        directory.mkdir(parents=True, exist_ok=True)
        latest, manifest, final = (directory / name for name in ("latest.pt", "status.json", "final.pt"))
        if manifest.exists():
            previous = json.loads(manifest.read_text(encoding="utf-8"))
            if (previous["status"] == "COMPLETE" and final.is_file()
                    and previous["model_sha256"] == sha(final)):
                print(f"Seed {seed}: verified cached training", flush=True)
                continue
        metadata = {**identity, "id": config["identity"], "seed": seed, "status": "RUNNING",
                    "oracle_model_sha256": oracle["model_sha256"]}
        save_metadata(manifest, metadata)
        begin = monotonic()
        try:
            source = torch.load(stage1, map_location="cpu", weights_only=False)
            if source.get("seed") != seed:
                raise ValueError("Stage-1 checkpoint seed mismatch")
            model, meta = train_dynamic(train, val, source["scales"], config, latest,
                                        initial=source["model"], resume=True,
                                        stop_after_batches=stop_after_batches,
                                        queue_factory=factory if observation == "estimated" else rl_core_execution.BatchQueue)
            if model is None:
                save_metadata(manifest, {**metadata, "status": "INTERRUPTED",
                                         "saved_batches": meta["steps"], "session_seconds": monotonic() - begin})
                print(f"Seed {seed}: checkpoint saved at batch {meta['steps']}", flush=True)
                continue
            transitions = meta["steps"] * config["nenv"]
            if (transitions, meta["updates"]) != (2_000_128, 7695):
                raise RuntimeError("Realized training budget differs from the historical comparison")
            atomic_torch_save({"model": model, "scales": source["scales"],
                               "identity": config["identity"], "meta": meta}, final)
            save_metadata(manifest, {**metadata, "status": "COMPLETE", "actual_transitions": transitions,
                                     "optimizer_updates": meta["updates"], "model_sha256": sha(final),
                                     "session_seconds": monotonic() - begin})
            print(f"Seed {seed}: {transitions:,} transitions, {meta['updates']:,} updates", flush=True)
        except (Exception, KeyboardInterrupt) as error:
            save_metadata(manifest, {**metadata, "status": "INTERRUPTED" if isinstance(error, KeyboardInterrupt)
                                     else "FAILED", "error": f"{type(error).__name__}: {error}"})
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--observation", choices=("oracle", "estimated"), default="estimated")
    parser.add_argument("--stop-after-batches", type=int)
    args = parser.parse_args()
    train_control(args.root, args.registry, args.output, args.seeds, args.device,
                  args.observation, args.stop_after_batches)
