# Protocol and evidence scope

- The original Azure 2019 bank has 59 reconstructions from two inspected source days. The later bank uses three source days, d12–d14, with prior-use provenance unverified. Both bank analyses preserve pairing by workload and training seed.
- The crossed observation runs compare oracle and estimated observation paths for the learned router and guard. Hybrid oracle configurations are simulation diagnostics. The estimated observer uses only information available before dispatch.
- Azure 2021 elapsed windows 7–8 were reserved under the existing transfer protocol. Source-day and workload dependencies matter when interpreting variation, and conditional intervals describe variation under the observed evaluation design.
- The five-seed historical fine-tuning comparisons used different calendar implementations. The matched-calendar control covers two seeds. The feasibility floors for Azure 2021 are necessary bounds.
- The results preserve workload-dependent reversals and failed transfer or attainment improvements. Archived miss rates and latencies are stored in their original units and schemas.

The [provenance map](provenance_map.csv) connects source record paths and SHA-256 hashes to the files in this artifact and its release archives. Historical schemas and embedded identifiers are preserved for independent hash verification. The [layout manifest](../configs/layout.json) maps those public paths to the input layout created on demand by the runner.

Seven frozen temporal records retain strings with historical workstation paths. These strings contribute to the archived protocol or manifest hashes. The artifact runner locates supplied inputs by archive member and verifies their hashes independently of those paths.

| Historical identifier | Study name |
| --- | --- |
| E2 | Compute-matched training and penalty comparisons |
| E3 | Workload and topology robustness |
| E4 | State and reward ablations |
| E5 | Guard evaluation and transfer |
| E6 | Crossed observations and feasibility |
| B | Later-day temporal evaluation |
| D | Later-day crossed observations |
| Transfer | Reserved-window Azure 2021 transfer |
| A1 | Budget sensitivity |
| A2 | Archived miss–latency trade-offs |
| A3 | Transfer telemetry |
| A4 | Workload and topology robustness |
| T1 | Guard threshold sensitivity |
| T2 | Reward penalty sensitivity |
| T3 | Estimator calibration |
| T4 | Matched-calendar fine-tuning |
| T5 | Observation error sensitivity |
| T6 | Guard and queue dynamics |
| T7 | Target-domain feature recalibration |
| T8 | In-domain fine-tuning |

Earlier public artifact: [Queue Replay v1.1-submission](https://github.com/thacanh/queue-replay/releases/tag/v1.1-submission), commit `382f68c5b7a9535f0d9378658ad00129ad9b592a`.
