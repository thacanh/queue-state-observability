"""Stage and verify published replay assets and separately obtained Azure traces."""

import argparse
import json
from pathlib import Path
import sys
from zipfile import ZipFile


REVISION = Path(__file__).resolve().parent
sys.path.insert(0, str(REVISION.parent))
from stage_replay_inputs import attach, destination, verified


def stage_zip(path, records):
    with ZipFile(path) as archive:
        names = [entry.filename for entry in archive.infolist() if not entry.is_dir()]
        if len(names) != len(set(names)) or set(names) != set(records):
            raise ValueError(f"Unexpected files in {path.name}")
        for name, record in records.items():
            with archive.open(name) as source:
                attach(source, destination(name), record)
    print(f"Verified and staged {len(records)} files from {path.name}.")


def stage_azure(folder, records):
    for name, record in records.items():
        target = destination(name)
        if verified(target, record):
            continue
        candidate = next((file for file in sorted(folder.rglob(target.name))
                          if verified(file, record)), None)
        if candidate is None:
            raise ValueError(f"No original file with the expected SHA-256: {target.name}")
        with candidate.open("rb") as source:
            attach(source, target, record)
    print(f"Verified and staged {len(records)} original Azure files.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, help="Path to checkpoints.zip")
    parser.add_argument("--workloads", type=Path, help="Path to reconstructed-workloads.zip")
    parser.add_argument("--azure-2019", type=Path, help="Directory containing extracted original Azure 2019 CSVs")
    parser.add_argument("--azure-2021", type=Path, help="Directory containing the original RAR and extracted TXT")
    args = parser.parse_args()
    if not any(vars(args).values()):
        parser.error("Specify at least one input to stage.")
    manifest = json.loads((REVISION / "ASSET_SHA256_MANIFEST.json").read_text(encoding="utf-8"))
    for path, key in ((args.checkpoints, "checkpoints.zip"), (args.workloads, "reconstructed-workloads.zip")):
        if path:
            asset = manifest["assets"][key]
            if not verified(path, asset):
                raise ValueError(f"Archive hash differs: {path.name}")
            stage_zip(path, asset["files"])
    for path, key in ((args.azure_2019, "2019"), (args.azure_2021, "2021")):
        if path:
            if not path.is_dir():
                raise NotADirectoryError(path)
            stage_azure(path, manifest["upstream_azure"][key])


if __name__ == "__main__":
    main()
