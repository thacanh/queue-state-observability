"""Compare matched fixed-penalty policies on the 59 frozen reconstructions."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from deadline_guard import _guard_service_bank
from estimated_ddqn import save_metadata, sha
from replay_io import load_json, results, XCOLS
from routing_simulator import replay
from simulation_baselines import SPEED, WORKERS


def estimated_models(runs, seeds, device):
    import torch

    found = {}
    for path in sorted(runs.glob("seed-*/status.json")):
        source = load_json(path)
        seed = source["seed"]
        if (seed not in seeds or source["status"] != "COMPLETE" or
                source["runtime"]["device"] != device or source.get("observation") != "estimated"):
            continue
        model = path.with_name("final.pt")
        if sha(model) != source["model_sha256"] or seed in found:
            raise ValueError(f"Invalid or ambiguous model for seed {seed}")
        state = torch.load(model, map_location="cpu", weights_only=True)
        if (source["actual_transitions"], source["optimizer_updates"], source["id"]) != (
                2_000_128, 7695, state["identity"]):
            raise ValueError(f"Invalid training budget or identity for seed {seed}")
        weights = tuple(np.ascontiguousarray(state["model"][f"net.{key}"].numpy(), dtype=np.float32)
                        for key in ("0.weight", "0.bias", "2.weight", "2.bias", "4.weight", "4.bias"))
        scales = np.array([state["scales"][key] for key in ("p50", "p99", "unc", "ia", "res")])
        found[seed] = (scales, weights, source)
    if set(found) != set(seeds):
        raise ValueError(f"Missing completed training for seeds {sorted(set(seeds) - set(found))}")
    if len({tuple(sorted(meta["runtime"].items())) for _, _, meta in found.values()}) != 1:
        raise ValueError("Training runtimes differ across selected seeds")
    return found


def write_csv(frame, path):
    temporary = path.with_suffix(".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def compare(artifact, runs, output, seeds, device, workers=4):
    shield = artifact / "outputs/shield_verified"
    train = shield / "inputs/train_d01_d07.csv"
    bank_path = shield / "source_data/runs/deadline_guard_v1/inputs"
    source_rows = artifact / "outputs/guard_evaluation/paired.csv"
    registry = load_json(bank_path / "manifest.json")
    if len(registry) != 59:
        raise ValueError("Unexpected validation bank size")
    models = estimated_models(runs, seeds, device)
    train_digest = sha(train)
    if any(md["inputs"]["train"] != train_digest for _, _, md in models.values()):
        raise ValueError("Guard service bank training input mismatch")
    names, bank, offsets = _guard_service_bank(train)
    prefix = np.r_[0., np.cumsum(bank)]
    means = np.diff(prefix[offsets]) / np.diff(offsets)
    old = pd.read_csv(source_rows)
    old = old.query("observation == 'estimated' and seed in @seeds and method in ['Fixed-DDQN', 'Fixed-DDQN+Guard']").copy()
    old["guard"] = old.method.str.endswith("+Guard")
    old["policy_training"] = "oracle"
    keys = ["day", "reconstruction_seed", "seed", "guard"]
    if len(old) != 2*59*len(seeds) or old.duplicated(keys).any():
        raise ValueError("Incomplete baseline replay bank")
    old_index = old.set_index(keys)

    def evaluate(item):
        label, spec = item
        trace_path = bank_path / spec["filename"]
        if sha(trace_path) != spec["sha256"]:
            raise ValueError(f"Reconstruction digest mismatch: {label}")
        day, rec_seed = label.split("_")
        rec_seed = int(rec_seed)
        frame = pd.read_csv(trace_path, dtype=spec["dtypes"], float_precision="round_trip")
        fingerprint = hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest()
        if fingerprint != spec["frame_sha256"]:
            raise ValueError(f"Reconstruction frame digest mismatch: {label}")
        x = np.ascontiguousarray(frame[XCOLS].to_numpy(np.float64))
        ids = frame["func_id"].astype(str).map(names).to_numpy(np.int64)
        output = []
        for seed, (scales, weights, meta) in models.items():
            for guarded in (False, True):
                reference = old_index.loc[(day, rec_seed, seed, guarded)]
                if (reference.trace_sha256, reference.model_sha256, reference.n) != (
                        fingerprint, meta["oracle_model_sha256"], len(frame)):
                    raise ValueError(f"Unpaired baseline run: {label}, seed {seed}")
                response = replay(x, ids, x[:, 1].copy(), SPEED, WORKERS, scales, weights,
                                  0, guarded, True, 1., bank, offsets, prefix, means, .005, .005)
                metrics, _ = results(x, response)
                output.append(dict(day=day, reconstruction_seed=rec_seed, seed=seed,
                                   trace_sha256=fingerprint, model_sha256=meta["model_sha256"],
                                   guard=guarded, policy_training="estimated_fine_tuning", **metrics))
        return output

    items = list(registry.items())
    first = evaluate(items[0])
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = first + [row for block in tqdm(pool.map(evaluate, items[1:]), total=58,
                                              desc="Paired validation replays") for row in block]
    new = pd.DataFrame(rows)
    fields = keys + ["policy_training", "trace_sha256", "model_sha256", "n", "violations",
                     "violation_rate", "met_budget", "mean_ms", "p95_ms", "p99_ms",
                     "override_rate", "wait_bias_ms", "wait_MAE_ms", "brier"]
    all_rows = pd.concat([old[fields], new[fields]], ignore_index=True)
    pair = new.merge(old, on=keys, suffixes=("_estimated", "_oracle"), validate="one_to_one")
    pair["miss_difference_pp"] = 100 * (pair.violation_rate_estimated - pair.violation_rate_oracle)
    pair["p95_difference_ms"] = pair.p95_ms_estimated - pair.p95_ms_oracle
    pair["gained_pass"] = pair.met_budget_estimated & ~pair.met_budget_oracle
    pair["lost_pass"] = ~pair.met_budget_estimated & pair.met_budget_oracle
    effects = pair.groupby(["day", "seed", "guard"], as_index=False).agg(
        pairs=("miss_difference_pp", "size"), mean_miss_difference_pp=("miss_difference_pp", "mean"),
        mean_p95_difference_ms=("p95_difference_ms", "mean"),
        gained_passes=("gained_pass", "sum"), lost_passes=("lost_pass", "sum"))
    summary = all_rows.groupby(["policy_training", "guard"], as_index=False).agg(
        pairs=("met_budget", "size"), passes=("met_budget", "sum"), mean_miss_rate=("violation_rate", "mean"),
        mean_p95_ms=("p95_ms", "mean"), mean_p99_ms=("p99_ms", "mean"), mean_brier=("brier", "mean"))
    output.mkdir(parents=True, exist_ok=True)
    for df, name in ((all_rows, "paired"), (effects, "paired_effects_by_day_seed"), (summary, "summary")):
        write_csv(df, output / f"{name}.csv")
    save_metadata(output / "manifest.json", {
        "scope": "Paired Azure 2019 validation replays conditional on the inspected source days d08 and d09",
        "training_observation": "Oracle Stage 1 for both groups, completion events only during Estimated-DDQN fine-tuning",
        "evaluation_observation": "Completion-event estimates for both groups",
        "guard_trigger_and_reduction": .005,
        "validation_bank_sha256": sha(bank_path / "manifest.json"),
        "baseline_sha256": sha(source_rows),
        "training_manifests": {str(seed): m for seed, (_, _, m) in models.items()},
        "evaluation_source_sha256": sha(Path(__file__)),
        "outputs": {p.name: sha(p) for p in output.glob("*.csv")},
    })
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--training-device", "--device", dest="device", choices=("cpu", "cuda"), default="cuda",
                        help="Filter the checkpoint training device. Replay itself runs on the CPU.")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    compare(args.artifact, args.runs, args.output, args.seeds, args.device, args.workers)
