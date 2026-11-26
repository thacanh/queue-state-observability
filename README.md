# Queue-State Observability

Research artifact for *The Role of Queue-State Observability in Deadline-Aware Heterogeneous Request Routing: A Controlled Empirical Study* by Thang Tat Nguyen, Anh Thac Nguyen, Yen Hai Nguyen, and Long Hoang Dang.

The study examines how queue information affects a learned request router and its runtime deadline guard. The oracle observer exposes worker release times in simulation. The estimated observer uses events available at decision time and historical service profiles. The artifact includes the original evaluation, replay on later Azure 2019 days, transfer to Azure 2021, and diagnostics for estimator error, fine-tuning, and workload-level miss budgets.

## 1. Reproduce exhibits from archived outcomes

Use Python 3.13. Create an environment from the checkout root.

```bash
python -m venv .venv
```

Activate it with `. .venv/bin/activate` on macOS or Linux, or `.venv\Scripts\Activate.ps1` in PowerShell. Then run:

```bash
python -m pip install -r requirements-level1.txt
python scripts/reproduce.py verify
python scripts/reproduce.py exhibits --output regenerated-exhibits
```

This workflow uses the included replay tables and the two reserved-window arrays for checking the feasibility bounds. It verifies file hashes, regenerates baseline statistics, tables and empirical figures, recalculates the paired fine-tuning comparisons, audits the temporal evaluation, and regenerates the main diagnostic exhibits. Generated files go in the specified fresh output directory. The analysis uses CPU only.

## 2. Replay the frozen evaluation

Install the CPU replay dependencies. Obtain the archive files named in [the release inventory](configs/release_assets.json) and place them in `downloads/`. The runner verifies each archive and the individual members it stages.

```bash
python -m pip install -r requirements-level2.txt
python scripts/reproduce.py smoke --output regenerated-smoke
python scripts/reproduce.py temporal-evaluation --assets downloads --output regenerated-temporal
python scripts/reproduce.py crossed-observations --assets downloads --output regenerated-crossed
python scripts/reproduce.py reserved-window-transfer --assets downloads --output regenerated-transfer
```

The three temporal commands compare every regenerated row with the frozen result for that arm. They use four CPU workers and one thread per numerical library. The archived campaigns counted 1,090 executions and 940 distinct configurations, including 150 configurations shared by the two Azure 2019 evaluations. For the original evaluation, use the separate source-trace setup and commands in the [reproduction guide](docs/reproduction.md).

## 3. Reproduce compute-matched fine-tuning

The original estimated-state comparison, matched-calendar control, and target-domain diagnostic start from hashed initialization checkpoints. See [GPU setup and commands](docs/reproduction.md#repeat-fine-tuning). The archived training configurations record the seeds, initialization hashes, transition counts, and optimizer updates. The artifact provides frozen weights for verification when the matching CUDA environment is unavailable.

## Browse the evidence

- [Original results](results/baseline) cover baseline comparisons, crossed observations, robustness, and retrained state ablations.
- [Observation-aligned fine-tuning](results/finetuning_comparison) contains the original four-arm evaluation and cross-trace transfer.
- [Temporal evaluation](results/temporal) contains the later-day and reserved-window results, locked protocol records, paired analyses and overlap audit.
- [Diagnostics](results/diagnostics) contains budget and penalty sensitivity, estimator calibration, matched-calendar results, observation perturbations, guard dynamics, feature recalibration, and in-domain fine-tuning.
- [Protocol and provenance](docs/protocol.md) describes the scope of inference, original identifiers and checksum mapping.

## Data and license

Use the provider's archives for the original request traces. The release archives contain selected, processed, or reconstructed workload inputs and checkpoint weights. See [data sourcing, attribution and licensing](data/README.md) and [checkpoint provenance](checkpoints/README.md). The [MIT License](LICENSE) applies to original software source. See [CITATION.cff](CITATION.cff) for citation metadata.
