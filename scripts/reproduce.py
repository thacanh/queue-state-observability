"""Verify or reproduce evidence from a clean checkout."""

from argparse import ArgumentParser
from hashlib import file_digest
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
LAYOUT = json.loads((ROOT / "configs/layout.json").read_text(encoding="utf-8"))
ASSETS = json.loads((ROOT / "configs/release_assets.json").read_text(encoding="utf-8"))
TEMPORAL = {
    "temporal-evaluation": "B",
    "crossed-observations": "D",
    "reserved-window-transfer": "transfer",
}
ENV = os.environ | {
    "MPLBACKEND": "Agg",
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "MKL_THREADING_LAYER": "SEQUENTIAL",
    "NUMBA_NUM_THREADS": "1",
}


def sha256(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def verify():
    count = 0
    for line in (ROOT / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        path = ROOT / relative
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Missing file or SHA-256 mismatch: {relative}")
        count += 1
    print(f"Verified {count} repository files.", flush=True)


def archive(directory, name):
    path = directory / name
    if not path.is_file() or sha256(path) != ASSETS[name]["sha256"]:
        raise ValueError(f"Missing release asset or SHA-256 mismatch: {path}")
    return path


def stage(workspace, *, assets=None, group=None):
    workspace.mkdir(parents=True, exist_ok=True)
    for layout, relative in LAYOUT.items():
        if group and layout.partition("/")[0] not in group:
            continue
        source = ROOT / relative
        if source.is_file():
            target = workspace / layout
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    if not assets:
        return
    inverse = {public: layout for layout, public in LAYOUT.items()}
    for name in ("temporal-replay-inputs.zip", "diagnostic-replay-inputs.zip"):
        members = ASSETS[name]["members"]
        selected = [p for p in members if not group or inverse[p].partition("/")[0] in group]
        if not selected:
            continue
        with ZipFile(archive(assets, name)) as package:
            if sorted(package.namelist()) != sorted(members):
                raise ValueError(f"Unexpected archive members in {name}")
            for public in selected:
                target = workspace / inverse[public]
                target.parent.mkdir(parents=True, exist_ok=True)
                with package.open(public) as stream, target.open("wb") as output:
                    shutil.copyfileobj(stream, output)
                if sha256(target) != members[public]:
                    raise ValueError(f"Incorrect archive member hash: {public}")


def run(base, script, *arguments):
    subprocess.run([sys.executable, "-B", script, *map(str, arguments)],
                   cwd=base, env=ENV, check=True)


def baseline_inventory(baseline):
    """Keep the frozen evidence hashes when staging the runnable baseline."""
    path = baseline / "REPRODUCTION_INVENTORY.json"
    inventory = json.loads(path.read_text(encoding="utf-8"))
    for name in (".gitattributes", ".gitignore", "ENVIRONMENT.md", "README.md"):
        inventory["files"].pop(name, None)
    path.write_text(json.dumps(inventory) + "\n", encoding="utf-8")


def exhibits(out):
    stage(out / "workspace", group={"baseline", "temporal", "work", "diagnostics"})
    baseline = out / "workspace/baseline"
    baseline_inventory(baseline)
    temporal = out / "workspace/temporal"
    run(baseline, "work/reproduce.py", "summaries")
    run(baseline, "work/audit_training_reference.py")
    run(baseline, "figures/generate_figures.py", "--evidence-root", baseline,
        "--output-dir", out / "baseline/figures")
    run(baseline, "generate_tables.py", "--evidence-root", baseline,
        "--output-dir", out / "baseline/tables")
    run(baseline, "verify_reproduction.py", "--table-dir", out / "baseline/tables")
    inventory = json.loads((baseline / "REPRODUCTION_INVENTORY.json").read_text(encoding="utf-8"))
    for name in inventory["figure_sha256"]:
        figure = out / "baseline/figures" / name
        with figure.open("rb") as stream:
            if stream.read(5) != b"%PDF-" or figure.stat().st_size < 1000:
                raise ValueError(f"Invalid PDF exhibit: {name}")
    run(baseline, "revision/code/extended_paired_analysis.py", "--artifact", baseline,
        "--reference", baseline / "revision/reference", "--output", out / "fine_tuning",
        "--csv-only")
    run(baseline, "revision/code/transfer_figure.py", baseline / "revision/results",
        out / "fine_tuning/transfer.pdf")
    run(temporal, "code/prepare_ieee_temporal.py", "--evidence", temporal / "evidence",
        "--historical", temporal / "historical", "--draft", out / "temporal", "--archived-only")
    workspace = out / "workspace"
    (workspace / "outputs").mkdir()
    (workspace / "diagnostics").rename(workspace / "outputs/supervisor_strengthening")
    build = workspace / "outputs/ieee_access_full_review"
    (build / "source/figures").mkdir(parents=True)
    run(workspace, "work/build_full_exhibits.py")
    shutil.copytree(build / "source", out / "diagnostics")
    shutil.copyfile(build / "exhibit_input_checksums.json", out / "diagnostics/source_hashes.json")


def smoke(out):
    stage(out / "workspace", group={"baseline", "training"})
    baseline = out / "workspace/baseline"
    run(baseline, "work/test_observability.py")
    run(baseline, "revision/reference/code/test_estimated_queue.py")
    run(out / "workspace/training", "test_shared_calendar.py")


def prepare_baseline(out, assets, azure_2019, azure_2021):
    stage(out / "workspace", group={"baseline"})
    baseline = out / "workspace/baseline"
    baseline_inventory(baseline)
    run(baseline, "revision/stage_assets.py", "--checkpoints", archive(assets, "checkpoints.zip"),
        "--workloads", archive(assets, "reconstructed-workloads.zip"))
    if azure_2019:
        run(baseline, "revision/stage_assets.py", "--azure-2019", azure_2019)
    if azure_2021:
        run(baseline, "revision/stage_assets.py", "--azure-2021", azure_2021)
    print(f"Verified baseline replay workspace: {baseline}")


def stage_diagnostics(out, assets, azure_2019, azure_2021):
    prepare_baseline(out, assets, azure_2019, azure_2021)
    workspace = out / "workspace"
    stage(workspace, assets=assets, group={"temporal", "diagnostics", "training", "outputs"})
    records = workspace / "outputs"
    records.mkdir(exist_ok=True)
    (workspace / "diagnostics").rename(records / "supervisor_strengthening")
    temporal = records / "ieee_access_temporal/queue-replay-v1.2-temporal-review"
    temporal.parent.mkdir()
    (workspace / "temporal").rename(temporal)
    shutil.copytree(temporal / "evidence", records / "temporal_evaluation",
                    ignore=shutil.ignore_patterns("inputs"))
    original = workspace / "baseline"
    csv_out = records / "estimated_ddqn/validation/paired.csv"
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(original / "revision/reference/results/paired.csv", csv_out)
    for relative in ("shield_verified/source_data", "guard_evaluation/external"):
        source = original / "outputs" / relative
        if source.is_dir():
            shutil.copytree(source, records / relative, dirs_exist_ok=True)
    gpu = workspace / "training"
    shutil.copyfile(records / "verified_final/e2/registry.json", gpu / "registry.json")
    shutil.copyfile(temporal / "evidence/protocol.json", gpu / "temporal_protocol.json")
    shutil.copytree(temporal / "inputs/checkpoints/estimated", gpu / "estimated")
    study = records / "supervisor_strengthening"
    shutil.copytree(study / "prefix_inputs", gpu / "prefix_inputs")
    shutil.copyfile(study / "T7/calibration.json", gpu / "prefix_inputs/calibration.json")
    print(f"Verified diagnostic replay workspace: {workspace}")


def main():
    cli = ArgumentParser(description=__doc__)
    cli.add_argument("task", choices=("verify", "exhibits", "smoke", "stage-baseline",
                                     "stage-diagnostics", *TEMPORAL))
    cli.add_argument("--output", type=Path, default=Path("regenerated"))
    cli.add_argument("--assets", type=Path, help="Directory of downloaded release ZIPs.")
    cli.add_argument("--azure-2019", type=Path)
    cli.add_argument("--azure-2021", type=Path)
    args = cli.parse_args()
    verify()
    if args.task == "verify":
        return
    if args.task not in ("exhibits", "smoke") and not args.assets:
        cli.error("This workflow requires --assets.")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    if args.task == "exhibits":
        exhibits(out)
    elif args.task == "smoke":
        smoke(out)
    elif args.task in ("stage-baseline", "stage-diagnostics"):
        prepare = prepare_baseline if args.task == "stage-baseline" else stage_diagnostics
        prepare(out, args.assets.resolve(), args.azure_2019 and args.azure_2019.resolve(),
                args.azure_2021 and args.azure_2021.resolve())
    else:
        stage(out / "workspace", assets=args.assets.resolve(), group={"temporal"})
        temporal = out / "workspace/temporal"
        run(temporal, "code/replay_frozen_artifact.py", "--artifact", temporal,
            "--phase", TEMPORAL[args.task], "--output", out / "replay")


if __name__ == "__main__":
    main()
