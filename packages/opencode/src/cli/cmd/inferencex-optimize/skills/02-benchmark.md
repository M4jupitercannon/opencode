# Phase 2: Benchmark Execution {{SKIP_LABEL}}

## Objective
Run benchmarks for each config point using Docker containers.

## Steps

### 1. Load Filtered Configs
Read configs from `{{OUTPUT_DIR}}/results/filtered_configs.json` (or `sweep_configs.json` if no filters applied).

### 2. For Each Config Point, Run Docker Benchmark
For each config entry, extract these fields:
- `image`: Docker image to use
- `model`: HuggingFace model name
- `model-prefix`: Model prefix
- `precision`: Quantization precision
- `framework`: Inference framework (vllm, sglang, etc.)
- `runner`: GPU runner type (mi300x, h100, etc.)
- `isl`: Input sequence length
- `osl`: Output sequence length
- `tp`: Tensor parallelism
- `ep`: Expert parallelism (default: 1)
- `conc`: Concurrency level
- `max-model-len`: Maximum model length
- `exp-name`: Experiment name

### 3. Determine Benchmark Script
```bash
BENCHMARK_SCRIPT="benchmarks/single_node/${EXP_NAME%%_*}_${PRECISION}_${RUNNER}.sh"
if [ ! -f "{{REPO_DIR}}/$BENCHMARK_SCRIPT" ]; then
    BENCHMARK_SCRIPT="benchmarks/single_node/${EXP_NAME%%_*}_${PRECISION}_${RUNNER}_${FRAMEWORK}.sh"
fi
```

### 4. Build Docker Command
Detect GPU vendor and set appropriate flags:
```bash
# For AMD GPUs (runner starts with "mi")
GPU_FLAGS="--device=/dev/kfd --device=/dev/dri --group-add video --security-opt seccomp=unconfined"

# For NVIDIA GPUs
GPU_FLAGS="--gpus all"
```

Run the Docker container:
```bash
RESULT_FILENAME="${EXP_NAME}_${PRECISION}_${FRAMEWORK}_tp${TP}-ep${EP}_conc${CONC}"

docker run --rm \
    --entrypoint /bin/bash \
    $GPU_FLAGS \
    --shm-size 64g \
    --ipc=host \
    --network=host \
    -v {{REPO_DIR}}:/workspace \
    -v {{HF_CACHE}}:/root/.cache/huggingface \
    -w /workspace \
    -e MODEL=$MODEL \
    -e TP=$TP \
    -e EP_SIZE=$EP \
    -e CONC=$CONC \
    -e ISL=$ISL \
    -e OSL=$OSL \
    -e MAX_MODEL_LEN=$MAX_MODEL_LEN \
    -e RANDOM_RANGE_RATIO=0.5 \
    -e RESULT_FILENAME=$RESULT_FILENAME \
    -e PRECISION=$PRECISION \
    -e FRAMEWORK=$FRAMEWORK \
    -e EXP_NAME=$EXP_NAME \
    -e HF_HOME=/root/.cache/huggingface \
    -e HF_HUB_CACHE=/root/.cache/huggingface/hub \
    $IMAGE \
    $BENCHMARK_SCRIPT
```

{{DRY_RUN_NOTE}}

### 5. Collect Results
After each run, copy result files from the repo directory to `{{OUTPUT_DIR}}/results/`.
Log the result filename and status.

## Completion
Update progress.json:
```json
{
  "phase": "benchmark",
  "phases_completed": ["env", "config", "benchmark"],
  "current_step": "benchmarks complete",
  "details": {
    "benchmarks_run": <N>,
    "benchmarks_succeeded": <M>,
    "benchmarks_failed": <F>
  }
}
```
