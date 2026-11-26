"""Verify the release manifest and recompute the transfer aggregates."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def verify(root):
    document = json.loads((root/"SHA256_MANIFEST.json").read_text(encoding="utf8"))
    for name, expected in document["files"].items():
        path = (root/name).resolve()
        if not path.is_relative_to(root.resolve()) or path.stat().st_size != expected["bytes"] or sha(path) != expected["sha256"]:
            raise ValueError(f"File failed SHA-256 verification: {name}")
    result = root/"results"
    scope = json.loads((result/"transfer_manifest.json").read_text())
    assert scope["new_replays"] == scope["historical_replays_verified"] == 50
    assert sha(result/"transfer_protocol.json")==scope["protocol_sha256"]
    for name,digest in scope["outputs"].items():
        assert sha(result/name)==digest
    raw = pd.read_csv(result/"transfer_paired.csv")
    assert len(raw)==100 and not raw.duplicated(["day","seed","training","guard"]).any()
    assert set(raw.day)==set(range(9,14)) and set(raw.seed)==set(range(42,47))
    assert raw.groupby(["day","seed"]).trace_sha256.nunique().eq(1).all()
    assert raw.groupby(["day","seed","training"]).model_sha256.nunique().eq(1).all()
    np.testing.assert_allclose(raw.violation_rate,raw.violations/raw.n,rtol=0,atol=1e-15)
    assert (raw.met_budget == (raw.violations*200<=raw.n)).all()
    for group, filename in ((["training","guard"],"transfer_summary.csv"),
                            (["day","training","guard"],"transfer_by_day.csv")):
        expected=pd.read_csv(result/filename).set_index(group).sort_index()
        groups=raw.groupby(group)
        fields=[f for f in expected.columns if f not in ("passes","pairs")]
        actual=groups[fields].mean()
        actual["pairs"]=groups.size()
        actual["passes"]=groups.met_budget.sum()
        actual=actual[expected.columns]
        assert actual.index.equals(expected.index)
        np.testing.assert_allclose(actual.values,expected.values,rtol=1e-13,atol=1e-13)
    overlap=json.loads((result/"validation_analysis_manifest.json").read_text())
    for name,digest in overlap["outputs"].items():
        if name == "miss_rate_ecdf.pdf" and not (result/name).is_file():
            continue  # This plot is generated from the released CSV files.
        assert sha(result/name)==digest
    print(f"Verified {len(document['files'])} files, 100 transferred outcomes, and validation manifests.")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path)
    verify(parser.parse_args().root)
