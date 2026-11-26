"""Attach exact local replay inputs by content hash. Never fetch or publish data."""

from argparse import ArgumentParser
from hashlib import file_digest
import json
from pathlib import Path, PurePosixPath
from shutil import copyfileobj
from tempfile import NamedTemporaryFile
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent


def sha256(path):
    with path.open("rb") as source:
        return file_digest(source, "sha256").hexdigest()


def verified(path, expected):
    return (path.is_file() and path.stat().st_size == expected["bytes"]
            and sha256(path) == expected["sha256"])


def destination(name):
    rel = PurePosixPath(name)
    if rel.is_absolute() or ".." in rel.parts or "\\" in name or ":" in name:
        raise ValueError(f"Invalid manifest path: {name}")
    target = (ROOT / rel).resolve()
    if not target.is_relative_to(ROOT):
        raise ValueError(f"Manifest path resolves outside the replay directory: {name}")
    return target


def attach(source, target, expected):
    if target.exists():
        if not verified(target, expected):
            raise ValueError(f"Existing replay input differs: {target.relative_to(ROOT)}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = NamedTemporaryFile(prefix="replay-", dir=target.parent, delete=False)
    temporary_path = Path(temporary.name)
    try:
        with temporary:
            copyfileobj(source, temporary)
        if not verified(temporary_path, expected):
            raise ValueError(f"Replay input hash differs: {target.relative_to(ROOT)}")
        temporary_path.rename(target)
    finally:
        temporary_path.unlink(missing_ok=True)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Replay input archive or directory matching REPLAY_INPUTS.json")
    args = parser.parse_args()
    expected = json.loads((ROOT / "REPLAY_INPUTS.json").read_text(encoding="utf-8"))["files"]
    if args.input.is_dir():
        candidates = None
        for name, record in expected.items():
            rel = destination(name).relative_to(ROOT)
            candidate = args.input / rel
            if not candidate.is_file():
                if candidates is None:
                    candidates = list(args.input.rglob("*"))
                matches = [p for p in candidates if p.name == rel.name and verified(p, record)]
                if not matches:
                    raise FileNotFoundError(f"No matching replay input: {rel}")
                candidate = matches[0]
            with candidate.open("rb") as source:
                attach(source, ROOT / rel, record)
    else:
        with ZipFile(args.input) as archive:
            entries = {entry.filename: entry for entry in archive.infolist() if not entry.is_dir()}
            for name, record in expected.items():
                info = entries.get(name)
                if info is None:
                    for candidate in entries.values():
                        if (PurePosixPath(candidate.filename).name == PurePosixPath(name).name
                                and candidate.file_size == record["bytes"]):
                            with archive.open(candidate) as source:
                                if file_digest(source, "sha256").hexdigest() == record["sha256"]:
                                    info = candidate
                                    break
                    if info is None:
                        raise FileNotFoundError(f"No matching replay input: {name}")
                if info.file_size != record["bytes"]:
                    raise ValueError(f"Invalid replay input size: {name}")
                with archive.open(info) as source:
                    attach(source, destination(name), record)
    print(f"Verified and staged {len(expected)} exact replay inputs.")


if __name__ == "__main__":
    main()
