"""Load trace and checkpoint inputs and summarize routing outcomes."""

import hashlib
import json
import os
from pathlib import Path
import runpy

for setting in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[setting] = "1"

import numpy as np
import pandas as pd
import scipy.linalg
from numba import njit
from threadpoolctl import threadpool_limits

from simulation_baselines import SPEED, WORKERS, QCOLS, training_profiles
from routing_simulator import replay, survival, remaining_demand

ROOT = Path(__file__).resolve().parents[1]
SHIELD = ROOT / "outputs/shield_verified"
DATA = SHIELD / "source_data"
BANK_PATH = DATA / "runs/deadline_guard_v1"
OUT = ROOT / "outputs/guard_evaluation"
TRAIN = SHIELD / "inputs/train_d01_d07.csv"
REGISTRY = ROOT / "outputs/training_evidence/matched_training/registry.json"
BASELINE = ROOT / "outputs/training_evidence/matched_training/confirmation_paired.csv"
SEEDS = range(42, 47)
METHODS = [("RiskOnly", 2, False, "p50")]
METHODS += [(f"SED-P{q}{suffix}", 1, guarded, f"p{q}") for q in (50, 95, 99)
            for suffix, guarded in (("", False), ("+Guard", True))]
BINS = np.array([0., .001, .005, .01, .025, .05, .1, .25, .5, 1.000001])
XCOLS = ["arrival", "est_p50", "est_p99", "service_demand", "slo"]


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def publish_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def model_path(record):
    relative = Path(record["model_path"])
    candidates = (DATA / relative, SHIELD / "inputs" / relative.parent.name / relative.name)
    return next((path for path in candidates if path.is_file() and digest(path) == record["model_sha256"]), None)


def audited_inputs():
    jobs = load_json(REGISTRY)["jobs"]
    train_hash = digest(TRAIN)
    models, missing = {}, []
    for method, penalty in (("fixed", 2), ("adaptive", 10), ("latency", 0)):
        for seed in SEEDS:
            name = f"{method}_seed{seed}_lambda{penalty}"
            record = jobs[name]["record"]
            if record["input_sha256"]["train"] != train_hash:
                raise ValueError("Training digest mismatch")
            path = model_path(record)
            if path is None:
                missing.append({"model": name, "sha256": record["model_sha256"]})
            else:
                models[(method, seed)] = (path, record["model_sha256"])
    if any(("fixed", seed) not in models for seed in SEEDS):
        raise ValueError("All five fixed checkpoints are required")
    return models, missing, train_hash


def setup():
    import torch

    models, missing, train_hash = audited_inputs()
    global CONTEXT
    scope = runpy.run_path(str(ROOT / "work/deadline_guard.py"),
                          init_globals={"njit": njit, "np": np, "pd": pd})
    names, bank, offsets = scope["_guard_service_bank"](TRAIN)
    prefix = np.r_[0., np.cumsum(bank)]
    means = np.diff(prefix[offsets]) / np.diff(offsets)
    profile = training_profiles(DATA)
    fallback = (np.ones(5), (np.zeros((3, 22), np.float32), np.zeros(3, np.float32),
                np.zeros((3, 3), np.float32), np.zeros(3, np.float32),
                np.zeros((3, 3), np.float32), np.zeros(3, np.float32)), "")
    data = {}
    for key, (path, sha) in models.items():
        ck = torch.load(path, map_location="cpu", weights_only=True)
        weights = tuple(np.ascontiguousarray(ck["model"]["net." + k].numpy(), dtype=np.float32)
                        for k in ("0.weight", "0.bias", "2.weight", "2.bias", "4.weight", "4.bias"))
        scales = np.array([ck["scales"][s] for s in ("p50", "p99", "unc", "ia", "res")])
        data[key] = (scales, weights, sha)
    baselines = pd.read_csv(BANK_PATH / "paired.csv").set_index(["day", "reconstruction_seed", "seed", "method"])
    CONTEXT = (names, bank, offsets, prefix, means, profile, fallback, data, baselines)


def read_workload(day, seed, manifest):
    item = manifest[f"{day}_{seed}"]
    path = BANK_PATH / "inputs" / item["filename"]
    if digest(path) != item["sha256"]:
        raise ValueError("Trace file digest mismatch")
    frame = pd.read_csv(path, dtype=item["dtypes"], float_precision="round_trip")
    fingerprint = hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest()
    if fingerprint != item["frame_sha256"]:
        raise ValueError("Trace content digest mismatch")
    return frame, fingerprint


def replay_case(x, group, pred, model, method, guarded, estimated, bias=1., trigger=.005, reduction=.005):
    _, bank, offsets, prefix, means, *_ = CONTEXT
    scales, weights, _ = model
    return replay(x, group, pred, SPEED, WORKERS, scales, weights, method, guarded, estimated,
                  bias, bank, offsets, prefix, means, trigger, reduction)


def workload_arrays(frame):
    names, _, _, _, _, profile, *_ = CONTEXT
    groups = frame.func_id.astype(str)
    x = np.ascontiguousarray(frame[XCOLS].to_numpy(np.float64))
    if not np.isfinite(x).all() or (np.diff(x[:, 0]) < 0).any() or (x[:, 3] < 0).any():
        raise ValueError("Invalid trace")
    ids = groups.map(names).to_numpy(np.int64)
    p = profile.loc[groups, QCOLS].to_numpy()
    np.testing.assert_allclose(p[:, 2], x[:, 1], rtol=1e-5)
    np.testing.assert_allclose(p[:, 4], x[:, 2], rtol=1e-5)
    alpha = (.95 - .75) / (.99 - .75)
    return x, ids, {"p50": x[:, 1].copy(),
                     "p95": np.exp((1-alpha)*np.log(p[:, 3])+alpha*np.log(p[:, 4])),
                     "p99": x[:, 2].copy()}


def results(x, out):
    lat, action, proposal, risk, proposed_risk, error, direct, wait, oracle_risk, _ = out.T
    miss = lat > x[:, 4]
    override = action != proposal
    summary = dict(n=len(x), violations=int(miss.sum()), violation_rate=float(miss.mean()),
                   met_budget=bool(miss.mean() <= .005), mean_ms=float(lat.mean()*1000),
                   p95_ms=float(np.quantile(lat, .95)*1000), p99_ms=float(np.quantile(lat, .99)*1000),
                   overrides=int(override.sum()), override_rate=float(override.mean()),
                   wait_bias_ms=float(error.mean()*1000), wait_MAE_ms=float(np.abs(error).mean()*1000),
                   risk_vs_oracle_MAE=float(np.abs(risk-oracle_risk).mean()),
                   brier=float(np.mean(np.square(risk-miss))), risk_mean=float(risk.mean()),
                   oracle_state_brier=float(np.mean(np.square(oracle_risk-miss))),
                   direct_saves=int(np.count_nonzero(override & (direct > 0) & ~miss)),
                   direct_losses=int(np.count_nonzero(override & (direct == 0) & miss)),
                   risk_drop_mean=float(np.mean(proposed_risk[override]-risk[override])) if override.any() else 0.)
    bins = np.searchsorted(BINS, risk, side="right") - 1
    calibration = np.stack([np.bincount(bins, weights=w, minlength=len(BINS)-1)
                            for w in (np.ones(len(x)), risk, miss, oracle_risk)]).T
    return summary, calibration.tolist()
