# Phase 1: Config Parsing & Sweep Generation {{SKIP_LABEL}}

## Objective
Parse the master YAML config and generate the full benchmark sweep matrix.

## Steps

### 1. Detect Config File
```bash
CONFIG_KEY="{{CONFIG_KEY}}"
if [[ "$CONFIG_KEY" == *mi3* ]]; then
    CONFIG_FILE=".github/configs/amd-master.yaml"
else
    CONFIG_FILE=".github/configs/nvidia-master.yaml"
fi
echo "Config file: $CONFIG_FILE"
```

### 2. Generate Sweep Configs
Use the InferenceX matrix generation script to produce the full benchmark matrix:
```bash
cd "{{REPO_DIR}}"
python3 utils/matrix_logic/generate_sweep_configs.py \
    test-config --config-files "$CONFIG_FILE" --config-keys "{{CONFIG_KEY}}"
```

Save the output to `{{OUTPUT_DIR}}/results/sweep_configs.json`.

### 3. Apply Filters
If `{{FILTER_CONC}}` is set, filter configs to only matching concurrency values.
If `{{FILTER_SEQ}}` is set, filter configs to only matching sequence lengths.

Sequence length mapping:
- `1k1k` → ISL=1024, OSL=1024
- `1k8k` → ISL=1024, OSL=8192
- `8k1k` → ISL=8192, OSL=1024

Save filtered configs to `{{OUTPUT_DIR}}/results/filtered_configs.json`.

### 4. Report Config Summary
Print summary of benchmark points:
- Total configs in sweep
- Filtered configs count
- For each config: model, framework, precision, TP, concurrency, ISL×OSL

## Completion
Update progress.json:
```json
{
  "phase": "config",
  "phases_completed": ["env", "config"],
  "current_step": "configs generated",
  "details": {
    "total_configs": <N>,
    "filtered_configs": <M>
  }
}
```
