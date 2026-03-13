# Phase 2: Benchmark Execution {{SKIP_LABEL}}

## Objective
Run benchmarks for each config point using a **single persistent Docker container**.
All benchmark configs are executed inside one container via `docker exec`, avoiding repeated container startup/teardown and reducing overhead.

## Steps

### 1. Load Configs
Read configs from `{{OUTPUT_DIR}}/results/sweep_configs.json` (filters were already applied during Phase 1).

### 2. Extract Config Fields
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
Substitute the actual values of EXP_NAME, PRECISION, RUNNER, and FRAMEWORK from the config into the path pattern:
```bash
BENCHMARK_SCRIPT="benchmarks/single_node/${EXP_NAME%%_*}_${PRECISION}_${RUNNER}.sh"
if [ ! -f "{{REPO_DIR}}/$BENCHMARK_SCRIPT" ]; then
    BENCHMARK_SCRIPT="benchmarks/single_node/${EXP_NAME%%_*}_${PRECISION}_${RUNNER}_${FRAMEWORK}.sh"
fi
```
IMPORTANT: When printing the script path, always print the full absolute path including the repo directory with actual values substituted (e.g. `{{REPO_DIR}}/benchmarks/single_node/kimik2.5_int4_mi355x.sh`), NOT the shell variable template or relative path.
Echo: `echo "BENCHMARK_SCRIPT={{REPO_DIR}}/$BENCHMARK_SCRIPT"`

### 4. Group Configs by Docker Image
Group all configs by their `image` field. Configs sharing the same Docker image will run in the same container.
Typically all configs for a given config-key use the same image, so there will be a single group.

### 5. Start One Persistent Container Per Image Group
Detect GPU vendor and start the container with access to **all** GPUs. GPU selection happens later at `docker exec` time (not at container start).

**5a. Detect GPU vendor:**
```bash
if [[ "$RUNNER" == mi* ]]; then
    GPU_VENDOR="amd"
else
    GPU_VENDOR="nvidia"
fi
```

**5b. Set GPU device flags (NO GPU visibility env vars):**
The container gets access to all GPUs. Visibility is restricted per-benchmark at `docker exec` time.
```bash
# For AMD GPUs (runner starts with "mi")
GPU_FLAGS="--device=/dev/kfd --device=/dev/dri --group-add video --security-opt seccomp=unconfined"

# For NVIDIA GPUs
GPU_FLAGS="--gpus all"
```

**5c. Start container:**
Start **one** container per image group in detached mode with `sleep infinity` to keep it alive:
```bash
CONTAINER_NAME="inferencex-benchmark-{{CONFIG_KEY}}"

docker run -d \
    --name "$CONTAINER_NAME" \
    --label inferencex-pipeline=true \
    --entrypoint /bin/bash \
    $GPU_FLAGS \
    --shm-size 64g \
    --ipc=host \
    --network=host \
    -v {{REPO_DIR}}:/workspace \
    -v {{HF_CACHE}}:/root/.cache/huggingface \
    -w /workspace \
    -e HF_HOME=/root/.cache/huggingface \
    -e HF_HUB_CACHE=/root/.cache/huggingface/hub \
    $IMAGE \
    -c "sleep infinity"
```

**5d. Copy GPU selection script into the container:**
```bash
docker cp {{SCRIPTS_DIR}}/select_gpus.py "$CONTAINER_NAME":/tmp/select_gpus.py
```

{{DRY_RUN_NOTE}}

### 6. Run Each Benchmark via `docker exec`
For each config in the group, **select the most free GPUs inside the container**, then run the benchmark script.

CRITICAL: You MUST use **two separate bash tool calls** for each benchmark run — one to select GPUs and print the info, and a second to execute `docker exec`. Do NOT combine them into a single bash call.

**Bash call 1 — Select GPUs and print DOCKER_LOG and RUN_CMD (separate bash call):**
Select the most free GPUs **inside the container** based on real-time VRAM usage. If GPUs were manually specified, use those instead.
```bash
MANUAL_GPUS="{{GPUS}}"
if [ -n "$MANUAL_GPUS" ]; then
    SELECTED_GPUS="$MANUAL_GPUS"
    echo "Using manually specified GPUs: $SELECTED_GPUS"
else
    SELECTED_GPUS=$(docker exec "$CONTAINER_NAME" python3 /tmp/select_gpus.py $TP)
    echo "Auto-selected most free GPUs (inside container): $SELECTED_GPUS"
fi

# Set GPU visibility env var for docker exec
# AMD: ONLY set ROCR_VISIBLE_DEVICES — NEVER set HIP_VISIBLE_DEVICES (it breaks PyTorch/ROCm GPU detection)
# NVIDIA: set CUDA_VISIBLE_DEVICES
if [ "$GPU_VENDOR" = "amd" ]; then
    GPU_ENV="-e ROCR_VISIBLE_DEVICES=$SELECTED_GPUS"
else
    GPU_ENV="-e CUDA_VISIBLE_DEVICES=$SELECTED_GPUS"
fi

RESULT_FILENAME="${EXP_NAME}_${PRECISION}_${FRAMEWORK}_tp${TP}-ep${EP}_conc${CONC}"
DOCKER_LOG="{{OUTPUT_DIR}}/results/${RESULT_FILENAME}_docker.log"
echo "DOCKER_LOG: $DOCKER_LOG"
RUN_CMD="docker exec $GPU_ENV -e MODEL=$MODEL -e TP=$TP -e EP_SIZE=$EP -e CONC=$CONC -e ISL=$ISL -e OSL=$OSL -e MAX_MODEL_LEN=$MAX_MODEL_LEN -e RANDOM_RANGE_RATIO=0.5 -e RESULT_FILENAME=$RESULT_FILENAME -e PRECISION=$PRECISION -e FRAMEWORK=$FRAMEWORK -e EXP_NAME=$EXP_NAME $CONTAINER_NAME /bin/bash /workspace/$BENCHMARK_SCRIPT"
echo "RUN_CMD: $RUN_CMD"
```
All variables must be fully expanded to actual values (not shell variables).

**Bash call 2 — Execute docker exec (separate bash call):**
```bash
docker exec \
    $GPU_ENV \
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
    "$CONTAINER_NAME" \
    /bin/bash /workspace/$BENCHMARK_SCRIPT \
    > "$DOCKER_LOG" 2>&1
EXIT_CODE=$?
echo "Benchmark exit code: $EXIT_CODE"
if [ $EXIT_CODE -ne 0 ]; then
    echo "=== Last 50 lines of docker log ==="
    tail -n 50 "$DOCKER_LOG"
fi
```

IMPORTANT: The docker exec runs in the **foreground** writing stdout/stderr to the log file (no output is printed to the terminal). If the command fails (non-zero exit code), the last 50 lines of the log are printed to help diagnose the issue. On success, only the exit code line is shown.

After each benchmark run, copy result files from the repo directory to `{{OUTPUT_DIR}}/results/`.
Then remove the copied result files from the repo directory to keep it clean:
```bash
cp {{REPO_DIR}}/results/${RESULT_FILENAME}*.json "{{OUTPUT_DIR}}/results/" 2>/dev/null || true
rm -f {{REPO_DIR}}/results/${RESULT_FILENAME}*.json 2>/dev/null || true
```
Log the result filename and status, then proceed to the next config.

### 7. Clean Up Container
After **all** benchmarks in the group are complete, stop and remove the container:
```bash
docker stop "$CONTAINER_NAME"
docker rm "$CONTAINER_NAME"
```

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
