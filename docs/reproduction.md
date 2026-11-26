# Replay and fine-tuning

Read the [setup and exhibit workflow](../README.md) first. Use a new output directory for each workflow. Frozen outcomes stay in the checkout.

The baseline table fragments are checked against the original output hashes. PDF rendering bytes can vary with font and library builds. The exhibit command verifies the source data and checks that the expected PDF files were generated.

## Original Azure 2019 and Azure 2021 replay

Download the source archives linked from [the data guide](../data/README.md). Extract the Azure 2019 duration-percentile CSVs for d01–d07 into `downloads/azure2019`. Place the original Azure 2021 RAR and its extracted TXT file in `downloads/azure2021`. The runner validates the expected source-file digests from the prior release.

```bash
python scripts/reproduce.py stage-baseline --assets downloads --azure-2019 downloads/azure2019 --azure-2021 downloads/azure2021 --output regenerated-original
cd regenerated-original/workspace/baseline
python work/reproduce.py crossed --workers 4
python work/reproduce.py transfer
python verify_reproduction.py --results --transfer
python revision/code/estimated_transfer.py --artifact . --checkpoints revision/reference/checkpoints --output regenerated/estimated-transfer
```

Repeat the command below for seeds 42–46 for the original fine-tuning control. The training-device flag selects the provenance of the checkpoint. Replay itself uses CPU.

```bash
python revision/reference/code/evaluate_estimated_ddqn.py --artifact . --runs revision/reference/checkpoints --output regenerated/validation/validation_seed42 --seeds 42 --training-device cuda
python revision/reference/code/summarize_estimated_ddqn.py --root regenerated/validation --output regenerated/validation/summary
```

## Repeat fine-tuning

Use the staged baseline from the preceding section. Set up a separate CUDA environment with `revision/requirements-gpu.txt`. Record the accelerator, driver, and package versions. The commands start from the specified Stage 1 models. Training results can vary across hardware and software builds.

```bash
python revision/reference/code/estimated_ddqn.py --root . --registry revision/reference/code/reference_registry.json --output regenerated/trained-estimated --seeds 42 43 44 45 46
python revision/reference/code/estimated_ddqn.py --root . --registry revision/reference/code/reference_registry.json --output regenerated/trained-oracle --seeds 42 43 44 45 46 --observation oracle
```

The matched-calendar and in-domain protocols are archived with [their training records](../checkpoints/diagnostics/gpu_results) and [trainer](../src/training/gpu_training.py). The latter uses the preceding Azure 2021 prefix, which is supplied as a separate timestamped training input in the diagnostic data archive. The original Stage 1 models are supplied as frozen initializations. The exhibit workflow and later-day replay commands are independent of GPU training.

## Reuse the diagnostic inputs

Stage all four asset archives into a new workspace. The source arguments are optional for archive checks and are needed for diagnostics that regenerate source-trace profiles.

```bash
python scripts/reproduce.py stage-diagnostics --assets downloads --azure-2019 downloads/azure2019 --azure-2021 downloads/azure2021 --output regenerated-diagnostic-inputs
cd regenerated-diagnostic-inputs/workspace
python outputs/supervisor_strengthening/code/gpu_evidence.py
python outputs/supervisor_strengthening/code/analyze_new.py
```

These last two commands verify checkpoint and replay identities and regenerate the paired comparisons and event analyses from the archived records. Historical phase names in this staged workspace match the original protocol IDs. See the [naming map](protocol.md) for their public names.

To run the two GPU controls using the frozen input workspace and a matching CUDA environment:

```bash
python training/gpu_training.py T4 --workspace training --project baseline --output repeated-training
python training/gpu_training.py T8 --workspace training --project baseline --output repeated-training
```
