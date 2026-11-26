"""Verify distributed files and regenerated evidence against recorded digests."""
from argparse import ArgumentParser
from hashlib import file_digest
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def verify_output(path, expected):
    with path.open("rb") as source:
        actual = file_digest(source, "sha256").hexdigest()
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch: {path}")


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--results", action="store_true")
    parser.add_argument("--transfer", action="store_true")
    parser.add_argument("--figure-dir", type=Path)
    parser.add_argument("--table-dir", type=Path)
    args = parser.parse_args()
    manifest = json.loads((ROOT / "REPRODUCTION_INVENTORY.json").read_text(encoding="utf-8"))
    hashes = manifest["files"]
    for name, expected in hashes.items():
        path = ROOT / name
        if path.stat().st_size != expected["bytes"]:
            raise ValueError(f"File size mismatch: {name}")
        verify_output(path, expected["sha256"])
    print(f"Verified {len(hashes)} distributed files.")
    generated = ROOT / "regenerated"
    if args.results:
        for name in manifest["crossed_outputs"]:
            filename = Path(name).name
            fresh = generated / "crossed_observations" / filename
            if filename == "status.json":
                reference = json.loads((ROOT / name).read_text(encoding="utf-8"))
                actual = json.loads(fresh.read_text(encoding="utf-8"))
                for key in ("rows", "sha256"):
                    if actual[key] != reference[key]:
                        raise ValueError(f"Crossed status differs in {key}.")
            else:
                verify_output(fresh, hashes[name]["sha256"])
        print(f"Verified {len(manifest['crossed_outputs']) - 1} crossed-experiment files and status hashes.")
    if args.transfer:
        verify_output(generated / "azure2021/paired_2021.csv",
                      hashes["outputs/guard_evaluation/external/paired_2021.csv"]["sha256"])
        reference = json.loads((ROOT / "outputs/guard_evaluation/external/status.json").read_text(encoding="utf-8"))
        actual = json.loads((generated / "azure2021/status.json").read_text(encoding="utf-8"))
        for key in ("rows", "cohort_size", "excluded_source_records", "paired_sha256"):
            if actual[key] != reference[key]:
                raise ValueError(f"Transfer status differs in {key}.")
        print(f"Verified all {actual['rows']} Azure 2021 replay rows.")
    if args.figure_dir:
        for name, expected in manifest["figure_sha256"].items():
            verify_output(args.figure_dir / name, expected)
        print(f"Verified {len(manifest['figure_sha256'])} regenerated figures.")
    if args.table_dir:
        for name in manifest["table_outputs"]:
            verify_output(args.table_dir / Path(name).name, hashes[name]["sha256"])
        print(f"Verified {len(manifest['table_outputs'])} regenerated table fragments.")


if __name__ == "__main__":
    main()
