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
Detect GPU vendor and set appropriate flags:
```bash
# For AMD GPUs (runner starts with "mi")
GPU_FLAGS="--device=/dev/kfd --device=/dev/dri --group-add video --security-opt seccomp=unconfined"

# For NVIDIA GPUs
GPU_FLAGS="--gpus all"
```

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

{{DRY_RUN_NOTE}}

### 6. Run Each Benchmark via `docker exec`
For each config in the group, run the benchmark script inside the already-running container using `docker exec`.

CRITICAL: You MUST use **two separate bash tool calls** for each benchmark run — one to print the info, and a second to execute `docker exec`. Do NOT combine them into a single bash call.

**Bash call 1 — Print DOCKER_LOG and RUN_CMD (separate bash call):**
```bash
RESULT_FILENAME="${EXP_NAME}_${PRECISION}_${FRAMEWORK}_tp${TP}-ep${EP}_conc${CONC}"
DOCKER_LOG="{{OUTPUT_DIR}}/results/${RESULT_FILENAME}_docker.log"
echo "DOCKER_LOG: $DOCKER_LOG"
RUN_CMD="docker exec -e MODEL=$MODEL -e TP=$TP -e EP_SIZE=$EP -e CONC=$CONC -e ISL=$ISL -e OSL=$OSL -e MAX_MODEL_LEN=$MAX_MODEL_LEN -e RANDOM_RANGE_RATIO=0.5 -e RESULT_FILENAME=$RESULT_FILENAME -e PRECISION=$PRECISION -e FRAMEWORK=$FRAMEWORK -e EXP_NAME=$EXP_NAME $CONTAINER_NAME /bin/bash /workspace/$BENCHMARK_SCRIPT"
echo "RUN_CMD: $RUN_CMD"
```
All variables must be fully expanded to actual values (not shell variables).

**Bash call 2 — Execute docker exec (separate bash call):**
```bash
docker exec \
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
echo "Benchmark exit code: $?"
```

Do NOT print or display the contents of the docker log file. The log is saved for debugging purposes only.

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
