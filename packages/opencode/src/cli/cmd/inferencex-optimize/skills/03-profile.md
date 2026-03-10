# Phase 3: Profiling {{SKIP_LABEL}}

## Objective
Re-run selected benchmarks with profiling enabled to capture detailed performance traces.

{{PROFILE_SKIP_NOTE}}

## Steps

### 1. Select Profiling Configs
Choose a representative subset of configs to profile (typically one concurrency level, one sequence length).
If `{{FILTER_TP}}`, `{{FILTER_CONC_START}}`/`{{FILTER_CONC_END}}`, and `{{FILTER_SEQ}}` are set, use those to narrow configs. Otherwise, pick a low-concurrency config (e.g., conc=4) with the default sequence length.

### 2. Create Profiles Directory
```bash
mkdir -p "{{PROFILE_DIR}}"
```

### 3. Start Persistent Profiling Container
Start **one** persistent container for all profiling runs. Add profiling environment variables.
Detect GPU vendor and set appropriate flags:
```bash
# For AMD GPUs (runner starts with "mi")
GPU_FLAGS="--device=/dev/kfd --device=/dev/dri --group-add video --security-opt seccomp=unconfined"

# For NVIDIA GPUs
GPU_FLAGS="--gpus all"
```

```bash
CONTAINER_NAME="inferencex-profile-{{CONFIG_KEY}}"

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
    -e PROFILE=1 \
    -e SGLANG_TORCH_PROFILER_DIR=/workspace/profiles \
    -e VLLM_TORCH_PROFILER_DIR=/workspace/profiles \
    -e VLLM_RPC_TIMEOUT=1800000 \
    $IMAGE \
    -c "sleep infinity"
```

{{DRY_RUN_NOTE}}

### 3a. Inject vLLM Profiler Config
vLLM v0.16+ requires `--profiler-config` on the `vllm serve` command to register the `/start_profile` and `/stop_profile` API endpoints. The `VLLM_TORCH_PROFILER_DIR` env var alone is not enough; without `--profiler-config`, the profiling routes are never attached and calls to `/start_profile` silently fail, producing no torch traces.

After starting the container, patch the resolved benchmark script **inside the container** so that any `vllm serve` invocation includes the profiler config.

Use Python to avoid nested bash/sed quoting issues — `json.dumps` guarantees valid JSON and `chr(39)` inserts shell single-quotes around the value so bash treats it as one argument:
```bash
docker exec "$CONTAINER_NAME" python3 -c "
import glob, json, os
prof_dir = os.environ.get('VLLM_TORCH_PROFILER_DIR', '/workspace/profiles')
cfg = json.dumps({'profiler': 'torch', 'torch_profiler_dir': prof_dir, 'torch_profiler_use_gzip': True})
q = chr(39)
for f in glob.glob('/workspace/benchmarks/**/*.sh', recursive=True):
    with open(f) as fh:
        content = fh.read()
    content = content.replace('vllm serve ', 'vllm serve --profiler-config ' + q + cfg + q + ' ', 1)
    with open(f, 'w') as fh:
        fh.write(content)
print('Patched benchmark scripts with --profiler-config')
"
```

This only modifies the copy inside the container, not the host repo.

### 3b. Disable Relay Trace Staging
The `move_profile_trace_for_relay()` function in `benchmark_lib.sh` copies the rank trace to the repo root as a relay file. This is for CI/CD workflows and not needed here — we collect rank traces directly from the profiles directory. Neutralize it inside the container:
```bash
docker exec "$CONTAINER_NAME" python3 -c "
with open('/workspace/benchmarks/benchmark_lib.sh') as f:
    content = f.read()
content = content.replace('move_profile_trace_for_relay', '# move_profile_trace_for_relay')
with open('/workspace/benchmarks/benchmark_lib.sh', 'w') as f:
    f.write(content)
print('Disabled move_profile_trace_for_relay')
"
```

### 4. Run Each Profile via `docker exec`
For each selected config, run the benchmark script with profiling env vars inside the persistent container.

CRITICAL: You MUST use **two separate bash tool calls** for each profiling run — one to print the info, and a second to execute `docker exec`. Do NOT combine them into a single bash call.

**Bash call 1 — Print DOCKER_LOG and RUN_CMD (separate bash call):**
```bash
PROFILE_RESULT="${EXP_NAME}_${PRECISION}_${FRAMEWORK}_tp${TP}-ep${EP}_conc${CONC}_profile"
DOCKER_LOG="{{PROFILE_DIR}}/${PROFILE_RESULT}_docker.log"
echo "DOCKER_LOG: $DOCKER_LOG"
RUN_CMD="docker exec -e MODEL=$MODEL -e TP=$TP -e EP_SIZE=$EP -e CONC=$CONC -e ISL=$ISL -e OSL=$OSL -e MAX_MODEL_LEN=$MAX_MODEL_LEN -e RANDOM_RANGE_RATIO=0.5 -e RESULT_FILENAME=$PROFILE_RESULT -e PRECISION=$PRECISION -e FRAMEWORK=$FRAMEWORK -e EXP_NAME=$EXP_NAME $CONTAINER_NAME /bin/bash /workspace/$BENCHMARK_SCRIPT"
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
    -e RESULT_FILENAME=$PROFILE_RESULT \
    -e PRECISION=$PRECISION \
    -e FRAMEWORK=$FRAMEWORK \
    -e EXP_NAME=$EXP_NAME \
    "$CONTAINER_NAME" \
    /bin/bash /workspace/$BENCHMARK_SCRIPT \
    > "$DOCKER_LOG" 2>&1
echo "Profile exit code: $?"
```

Do NOT print or display the contents of the docker log file. The log is saved for debugging purposes only.

### 5. Clean Up Container
After **all** profiling runs are complete, stop and remove the container:
```bash
docker stop "$CONTAINER_NAME"
docker rm "$CONTAINER_NAME"
```

### 6. Collect Profile Traces and Benchmark Results
Copy the **actual torch profiler traces** (produced by vLLM to `VLLM_TORCH_PROFILER_DIR`) and benchmark result JSONs to the output directory, then clean up all generated files from the repo:
```bash
# Torch profiler traces written by vLLM to the profiles subdirectory
# Skip async_llm traces (CPU-side scheduling only, not needed for GPU analysis)
for f in {{REPO_DIR}}/profiles/*.json*; do
    [ -f "$f" ] || continue
    case "$(basename "$f")" in
        *async_llm*) rm -f "$f" ;;
        *)           cp "$f" "{{PROFILE_DIR}}/" && rm -f "$f" ;;
    esac
done

# Copy benchmark result JSONs from the repo to the output results directory
mkdir -p "{{OUTPUT_DIR}}/results"
cp {{REPO_DIR}}/results/*.json "{{OUTPUT_DIR}}/results/" 2>/dev/null || true
rm -f {{REPO_DIR}}/results/*.json 2>/dev/null || true

echo "Collected trace files:"
ls -lh "{{PROFILE_DIR}}/"
echo "Collected benchmark results:"
ls -lh "{{OUTPUT_DIR}}/results/" 2>/dev/null || echo "(none)"
```

### 6a. Validate Trace Files
Verify that at least one collected trace file is a genuine torch profiler trace (contains `traceEvents` key), not just a benchmark result JSON:
```bash
python3 -c "
import json, gzip, glob, sys
trace_dir = '{{PROFILE_DIR}}'
valid = []
for f in sorted(glob.glob(trace_dir + '/*.json*')):
    if '_docker.log' in f:
        continue
    try:
        opener = gzip.open if f.endswith('.gz') else open
        with opener(f, 'rt') as fh:
            data = json.load(fh)
        if isinstance(data, dict) and 'traceEvents' in data:
            valid.append(f)
            print(f'VALID torch trace: {f}')
        else:
            keys = list(data.keys())[:5] if isinstance(data, dict) else type(data).__name__
            print(f'NOT a torch trace (keys: {keys}): {f}')
    except Exception as e:
        print(f'ERROR reading {f}: {e}')
if not valid:
    print('WARNING: No valid torch profiler traces found. TraceLens analysis will be skipped.')
    print('This usually means vLLM profiling endpoints were not activated.')
else:
    print(f'Found {len(valid)} valid torch trace(s)')
"
```

### 7. Profile Summary
List captured trace files and their sizes.
Note: traces can be viewed at https://ui.perfetto.dev/

## Completion
Update progress.json:
```json
{
  "phase": "profile",
  "phases_completed": ["env", "config", "benchmark", "profile"],
  "current_step": "profiling complete",
  "details": {
    "profile_runs": <N>,
    "trace_files": [<list of trace files>]
  }
}
```
