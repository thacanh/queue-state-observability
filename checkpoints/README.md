# Checkpoints

`checkpoints.zip` contains the five Stage 1 initializations, five oracle-state fine-tuned models, and five estimated-state fine-tuned models from the original comparison. The temporal and diagnostic archives also contain the exact models consumed by their runners. Some archived files intentionally reuse a checkpoint across protocols. The paper's matched-calendar diagnostic identifies equality of tensor values on the two audited seeds.

Use [the release inventory](../configs/release_assets.json) to check archive hashes and [the provenance map](../docs/provenance_map.csv) to locate the frozen source of each model. Training manifests retain seed, initialization digest, transition and optimizer counts, and device information. The software MIT license applies to the code. Checkpoint weights are distributed separately from that license.
