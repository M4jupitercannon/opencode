# Phase 4: Profiling {{SKIP_LABEL}}

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

### 2a. Adaptive Layer Estimation
Estimate total trace file size from the model's HuggingFace config. If estimated traces exceed 1 GB, profile a reduced model (8 layers) to keep traces manageable while still capturing representative kernel behavior.

```bash
python3 << 'PYEOF'
import json, os, sys, glob

model = os.environ["MODEL"]
hf_cache = os.environ.get("HF_CACHE", os.environ.get("HF_HUB_CACHE", "~/.cache/huggingface"))
tp = int(os.environ.get("TP", "1"))
max_iters = 256

config_path = None
model_slug = model.replace("/", "--")
for path in sorted(glob.glob(os.path.expanduser(hf_cache) + f"/hub/models--{model_slug}/snapshots/*/config.json")):
    config_path = path
    break

if not config_path:
    for root, dirs, files in os.walk(os.path.expanduser(hf_cache)):
        if "config.json" in files and model_slug in root:
            config_path = os.path.join(root, "config.json")
            break

if not config_path:
    print("PROFILE_NUM_LAYERS=full")
    sys.exit(0)

with open(config_path) as f:
    config = json.load(f)
num_layers = config.get("num_hidden_layers", 32)
num_experts = config.get("n_routed_experts", config.get("num_local_experts", 0))

base_mb = 0.12 if num_experts > 0 else 0.08
estimated_mb = num_layers * tp * max_iters * base_mb * 1.35

if estimated_mb > 1024:
    print(f"PROFILE_NUM_LAYERS=8")
    print(f"NOTE: Estimated {estimated_mb:.0f} MB > 1 GB for {num_layers} layers. Profiling 8/{num_layers} layers.")
else:
    print(f"PROFILE_NUM_LAYERS=full")
    print(f"NOTE: Estimated {estimated_mb:.0f} MB for {num_layers} layers. Profiling all layers.")
PYEOF
```

If `PROFILE_NUM_LAYERS != full`, inject `--hf-overrides '{"num_hidden_layers": 8}'` into the vLLM serve command in step 3a. The report should note: `**Model**: <name> (<N> layers profiled out of <total>)`.

### 3. Start Persistent Profiling Container
Detect GPU vendor, compute the number of required GPUs, select the best GPUs on the **host**, and start a container with **only those GPUs** mounted.

**3a. Detect GPU vendor:**
```bash
if [[ "$RUNNER" == mi* ]]; then
    GPU_VENDOR="amd"
else
    GPU_VENDOR="nvidia"
fi
```

**3b. Select GPUs and set device flags:**
Same host-side GPU isolation as Phase 2 (Benchmark). EP is a subdivision within TP and does not add extra GPUs. Use `select_gpus.py --docker-flags` to select GPUs and generate Docker device isolation flags in one step.
```bash
NUM_GPUS=$((TP * ${DP:-1}))

MANUAL_GPUS="{{GPUS}}"
if [ -n "$MANUAL_GPUS" ]; then
    echo "Using manually specified GPUs: $MANUAL_GPUS"
    if [ "$GPU_VENDOR" = "amd" ]; then
        GPU_FLAGS="--device=/dev/kfd --device=/dev/dri --group-add video --security-opt seccomp=unconfined"
    else
        GPU_FLAGS="--gpus device=$MANUAL_GPUS"
    fi
else
    GPU_FLAGS=$(python3 "{{SCRIPTS_DIR}}/select_gpus.py" $NUM_GPUS --docker-flags)
    echo "Auto-selected GPUs on host for TP=$TP DP=${DP:-1}"
    echo "GPU_FLAGS: $GPU_FLAGS"
fi
```

**3c. Start container:**
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

**3d. Verify GPU isolation:**
```bash
docker exec "$CONTAINER_NAME" python3 -c \
    "import torch; n=torch.cuda.device_count(); print(f'GPUs visible: {n}'); assert n==$NUM_GPUS, f'Expected $NUM_GPUS but got {n}'"
```

{{DRY_RUN_NOTE}}

### 3e. Inject vLLM Profiler Config
vLLM >= 0.15 requires `--profiler-config.*` CLI args on the `vllm serve` command to register the `/start_profile` and `/stop_profile` API endpoints. The `VLLM_TORCH_PROFILER_DIR` env var is deprecated; without `--profiler-config.*` args, the profiling routes are never attached and calls to `/start_profile` silently fail, producing no torch traces. 

First restore the benchmark script to its original state (previous runs may have patched the host copy via bind mount), then inject the profiler args:
```bash
cd {{REPO_DIR}} && git checkout -- "$BENCHMARK_SCRIPT" benchmarks/benchmark_lib.sh 2>/dev/null || true
```

Now patch the target benchmark script inside the container to inject `--profiler-config.*` args into the `vllm serve` command:
```bash
docker exec \
    -e OSL="${OSL}" -e CONC="${CONC}" -e RANDOM_RANGE_RATIO="${RANDOM_RANGE_RATIO:-0.5}" \
    "$CONTAINER_NAME" python3 - "/workspace/$BENCHMARK_SCRIPT" <<'PYEOF'
import sys, os, re, math

target = sys.argv[1]
prof_dir = os.environ.get('VLLM_TORCH_PROFILER_DIR', '/workspace/profiles')

osl = int(os.environ.get('OSL', '512'))
conc = int(os.environ.get('CONC', '32'))
rrr = float(os.environ.get('RANDOM_RANGE_RATIO', '0.5'))

# Phase-split profiling: position the window to capture both
# prefill-decode (mixed) and decode-only phases so the
# split_vllm_trace_annotation.py (--find-steady-state --num-steps 32)
# can extract both phase traces for roofline analysis.
#
# With num_prompts = conc * 10 (step 3g disables capping), the
# workload runs in ~10 "waves" of conc concurrent requests.
# Each iteration produces one token per active sequence.
#   total_iters ≈ 10 * avg_osl
# The mixed→decode-only transition occurs when the last wave
# finishes prefilling, at ~90% of total iterations.
num_prompts = conc * 10
avg_osl = osl * (1 + rrr) / 2 if rrr < 1 else osl
total_iters = int(num_prompts * avg_osl / conc)
transition = int(0.9 * total_iters)

# Profile 256 iterations centered on the estimated transition:
# ~128 mixed-phase steps + ~128 decode-only steps.
max_iters = 256
delay_iters = max(0, transition - max_iters // 2)

print(f'Computed profiler iterations: delay={delay_iters}, max={max_iters}  '
      f'(OSL={osl}, CONC={conc}, RANDOM_RANGE_RATIO={rrr}, '
      f'avg_osl={avg_osl:.0f}, total_est={total_iters}, transition_est={transition})')

profiler_args = (
    '--enforce-eager '
    '--profiler-config.profiler torch '
    '--profiler-config.torch_profiler_dir ' + prof_dir + ' '
    '--profiler-config.torch_profiler_record_shapes True '
    '--profiler-config.torch_profiler_with_memory False '
    '--profiler-config.torch_profiler_with_flops False '
    '--profiler-config.torch_profiler_use_gzip True '
    '--profiler-config.ignore_frontend True '
    '--profiler-config.delay_iterations ' + str(delay_iters) + ' '
    '--profiler-config.max_iterations ' + str(max_iters)
)
with open(target) as fh:
    content = fh.read()
content = re.sub(r'--enforce-eager\s+', '', content)
content = re.sub(r'--profiler-config\.\S+\s+\S+\s*', '', content)
content = re.sub(r'--ignore_frontend\s+\S+\s*', '', content)
new_content = content.replace('vllm serve ', 'vllm serve ' + profiler_args + ' ', 1)
if new_content != content:
    with open(target, 'w') as fh:
        fh.write(new_content)
    print(f'Patched {target} with --profiler-config.* args')
else:
    print(f'No "vllm serve" found in {target}, nothing to patch')
PYEOF
```

NOTE: The container bind-mounts `{{REPO_DIR}}:/workspace`, so these changes affect the host repo. Step 6 cleans up generated files, and the `git checkout` above ensures a clean starting state.

### 3f. Disable Relay Trace Staging
The `move_profile_trace_for_relay()` function in `benchmark_lib.sh` copies the rank trace to the repo root as a relay file. This is for CI/CD workflows and not needed here — we collect rank traces directly from the profiles directory. Neutralize the function **call** inside the container by replacing it with a bash no-op (`:`) so the enclosing `if` block remains syntactically valid:
```bash
docker exec "$CONTAINER_NAME" python3 -c "
import re
with open('/workspace/benchmarks/benchmark_lib.sh') as f:
    content = f.read()
# Only replace the bare function call (not the definition).
# Use word-boundary matching to avoid clobbering the 'function_name() {' definition line.
content = re.sub(
    r'^(\s*)move_profile_trace_for_relay\s*$',
    r'\1: # move_profile_trace_for_relay (disabled)',
    content,
    flags=re.MULTILINE,
)
with open('/workspace/benchmarks/benchmark_lib.sh', 'w') as f:
    f.write(content)
print('Disabled move_profile_trace_for_relay')
"
```

### 3g. Keep Full Prompt Count for Steady-State Profiling
By default `benchmark_lib.sh` caps `num_prompts` to `max_concurrency` when `PROFILE=1`, producing a single-batch run with no mixed prefill+decode steady state. Disable this cap so the benchmark sends `conc * 10` prompts, giving the profiler a continuous-flow workload with both prefill-decode and decode-only phases for phase-split roofline analysis:
```bash
docker exec "$CONTAINER_NAME" python3 -c "
import re
with open('/workspace/benchmarks/benchmark_lib.sh') as f:
    content = f.read()
content = re.sub(
    r'^(\s*)num_prompts=\"\\\$max_concurrency\"',
    r'\1: # num_prompts=\"\$max_concurrency\" (disabled for steady-state profiling)',
    content,
    flags=re.MULTILINE,
)
with open('/workspace/benchmarks/benchmark_lib.sh', 'w') as f:
    f.write(content)
print('Disabled num_prompts capping — benchmark will use original num_prompts (conc * 10)')
"
```

### 4. Run Dual-Mode Profiling (Eager + Graph)
Run profiling twice: first in **eager mode** (`--enforce-eager`), then in **graph mode** (CUDA graphs enabled). GPU isolation was applied at container start (step 3b–3d), so all visible GPUs inside the container are the selected ones — no per-exec GPU selection needed.

Each mode produces multi-rank traces that are collected into separate directories for independent analysis.

**4a. Eager-mode profiling run:**
The benchmark script was already patched in step 3e with `--enforce-eager` and `--profiler-config.*` args.
```bash
PROFILE_RESULT="${EXP_NAME}_${PRECISION}_${FRAMEWORK}_tp${TP}-ep${EP}_conc${CONC}_profile"
DOCKER_LOG="{{PROFILE_DIR}}/${PROFILE_RESULT}_eager_docker.log"

echo "=== EAGER MODE PROFILING ==="
echo "DOCKER_LOG: $DOCKER_LOG"
echo "RUN_CMD: docker exec -e MODEL=$MODEL -e TP=$TP -e EP_SIZE=$EP -e CONC=$CONC -e ISL=$ISL -e OSL=$OSL -e MAX_MODEL_LEN=$MAX_MODEL_LEN -e RANDOM_RANGE_RATIO=0.5 -e RESULT_FILENAME=${PROFILE_RESULT}_eager -e PRECISION=$PRECISION -e FRAMEWORK=$FRAMEWORK -e EXP_NAME=$EXP_NAME $CONTAINER_NAME /bin/bash /workspace/$BENCHMARK_SCRIPT"

docker exec \
    -e MODEL=$MODEL \
    -e TP=$TP \
    -e EP_SIZE=$EP \
    -e CONC=$CONC \
    -e ISL=$ISL \
    -e OSL=$OSL \
    -e MAX_MODEL_LEN=$MAX_MODEL_LEN \
    -e RANDOM_RANGE_RATIO=0.5 \
    -e RESULT_FILENAME=${PROFILE_RESULT}_eager \
    -e PRECISION=$PRECISION \
    -e FRAMEWORK=$FRAMEWORK \
    -e EXP_NAME=$EXP_NAME \
    "$CONTAINER_NAME" \
    /bin/bash /workspace/$BENCHMARK_SCRIPT \
    > "$DOCKER_LOG" 2>&1
EXIT_CODE=$?
echo "Eager profile exit code: $EXIT_CODE"
if [ $EXIT_CODE -ne 0 ]; then
    echo "=== Last 50 lines of docker log ==="
    tail -n 50 "$DOCKER_LOG"
fi
```

Collect eager-mode traces immediately (before graph-mode run overwrites the profiles directory):
```bash
mkdir -p "{{PROFILE_DIR}}/profiles_eager"
for f in {{REPO_DIR}}/profiles/*.json*; do
    [ -f "$f" ] || continue
    case "$(basename "$f")" in
        *async_llm*) rm -f "$f" ;;
        *)           cp "$f" "{{PROFILE_DIR}}/profiles_eager/" && rm -f "$f" ;;
    esac
done
for f in {{REPO_DIR}}/profiles/profiler_out_*.txt; do
    [ -f "$f" ] && cp "$f" "{{PROFILE_DIR}}/profiles_eager/" && rm -f "$f"
done
echo "Eager traces collected:"
ls -lh "{{PROFILE_DIR}}/profiles_eager/"
```

**4b. Re-patch for graph mode and run:**
Remove `--enforce-eager` from the benchmark script so vLLM uses CUDA graphs:
```bash
cd {{REPO_DIR}} && git checkout -- "$BENCHMARK_SCRIPT" benchmarks/benchmark_lib.sh 2>/dev/null || true
```
Re-apply all patches from steps 3e, 3f, 3g **without** `--enforce-eager`:
```bash
docker exec \
    -e OSL="${OSL}" -e CONC="${CONC}" -e RANDOM_RANGE_RATIO="${RANDOM_RANGE_RATIO:-0.5}" \
    "$CONTAINER_NAME" python3 - "/workspace/$BENCHMARK_SCRIPT" <<'PYEOF'
import sys, os, re, math

target = sys.argv[1]
prof_dir = os.environ.get('VLLM_TORCH_PROFILER_DIR', '/workspace/profiles')

osl = int(os.environ.get('OSL', '512'))
conc = int(os.environ.get('CONC', '32'))
rrr = float(os.environ.get('RANDOM_RANGE_RATIO', '0.5'))

num_prompts = conc * 10
avg_osl = osl * (1 + rrr) / 2 if rrr < 1 else osl
total_iters = int(num_prompts * avg_osl / conc)
transition = int(0.9 * total_iters)
max_iters = 256
delay_iters = max(0, transition - max_iters // 2)

print(f'Graph mode: delay={delay_iters}, max={max_iters}')

# No --enforce-eager for graph mode
profiler_args = (
    '--profiler-config.profiler torch '
    '--profiler-config.torch_profiler_dir ' + prof_dir + ' '
    '--profiler-config.torch_profiler_record_shapes True '
    '--profiler-config.torch_profiler_with_memory False '
    '--profiler-config.torch_profiler_with_flops False '
    '--profiler-config.torch_profiler_use_gzip True '
    '--profiler-config.ignore_frontend True '
    '--profiler-config.delay_iterations ' + str(delay_iters) + ' '
    '--profiler-config.max_iterations ' + str(max_iters)
)
with open(target) as fh:
    content = fh.read()
content = re.sub(r'--enforce-eager\s+', '', content)
content = re.sub(r'--profiler-config\.\S+\s+\S+\s*', '', content)
content = re.sub(r'--ignore_frontend\s+\S+\s*', '', content)
new_content = content.replace('vllm serve ', 'vllm serve ' + profiler_args + ' ', 1)
if new_content != content:
    with open(target, 'w') as fh:
        fh.write(new_content)
    print(f'Patched {target} for graph mode (no --enforce-eager)')
PYEOF
```

**IMPORTANT**: After `git checkout`, `benchmark_lib.sh` is also restored. You **must** re-apply the patches from steps 3f (disable relay trace staging) and 3g (disable num_prompts capping) by re-running both `docker exec` commands from those steps. Also add `--hf-overrides '{"num_hidden_layers": 8}'` if reduced layers are needed (step 2a). Then run graph-mode profiling:
```bash
DOCKER_LOG="{{PROFILE_DIR}}/${PROFILE_RESULT}_graph_docker.log"

echo "=== GRAPH MODE PROFILING ==="
echo "DOCKER_LOG: $DOCKER_LOG"

docker exec \
    -e MODEL=$MODEL \
    -e TP=$TP \
    -e EP_SIZE=$EP \
    -e CONC=$CONC \
    -e ISL=$ISL \
    -e OSL=$OSL \
    -e MAX_MODEL_LEN=$MAX_MODEL_LEN \
    -e RANDOM_RANGE_RATIO=0.5 \
    -e RESULT_FILENAME=${PROFILE_RESULT}_graph \
    -e PRECISION=$PRECISION \
    -e FRAMEWORK=$FRAMEWORK \
    -e EXP_NAME=$EXP_NAME \
    "$CONTAINER_NAME" \
    /bin/bash /workspace/$BENCHMARK_SCRIPT \
    > "$DOCKER_LOG" 2>&1
EXIT_CODE=$?
echo "Graph profile exit code: $EXIT_CODE"
if [ $EXIT_CODE -ne 0 ]; then
    echo "=== Last 50 lines of docker log ==="
    tail -n 50 "$DOCKER_LOG"
fi
```

Collect graph-mode traces:
```bash
mkdir -p "{{PROFILE_DIR}}/profiles_graph"
for f in {{REPO_DIR}}/profiles/*.json*; do
    [ -f "$f" ] || continue
    case "$(basename "$f")" in
        *async_llm*) rm -f "$f" ;;
        *)           cp "$f" "{{PROFILE_DIR}}/profiles_graph/" && rm -f "$f" ;;
    esac
done
for f in {{REPO_DIR}}/profiles/profiler_out_*.txt; do
    [ -f "$f" ] && cp "$f" "{{PROFILE_DIR}}/profiles_graph/" && rm -f "$f"
done
echo "Graph traces collected:"
ls -lh "{{PROFILE_DIR}}/profiles_graph/"
```

IMPORTANT: Each docker exec runs in the **foreground** writing stdout/stderr to the log file. If the command fails (non-zero exit code), the last 50 lines of the log are printed to help diagnose the issue.

### 5. Clean Up Container
After **all** profiling runs are complete, stop and remove the container:
```bash
docker stop "$CONTAINER_NAME"
docker rm "$CONTAINER_NAME"
```

### 6. Collect Benchmark Results and Clean Up Repo
Traces were already collected per-mode in step 4. Copy any remaining benchmark result JSONs and clean up:
```bash
mkdir -p "{{OUTPUT_DIR}}/results"
cp {{REPO_DIR}}/results/*.json "{{OUTPUT_DIR}}/results/" 2>/dev/null || true
rm -f {{REPO_DIR}}/results/*.json 2>/dev/null || true

cd {{REPO_DIR}} && git checkout -- "$BENCHMARK_SCRIPT" benchmarks/benchmark_lib.sh 2>/dev/null || true

echo "=== Collected traces ==="
echo "Eager mode:" && ls -lh "{{PROFILE_DIR}}/profiles_eager/" 2>/dev/null || echo "(none)"
echo "Graph mode:" && ls -lh "{{PROFILE_DIR}}/profiles_graph/" 2>/dev/null || echo "(none)"
echo "Benchmark results:" && ls -lh "{{OUTPUT_DIR}}/results/" 2>/dev/null || echo "(none)"
```

### 7. Profile Summary
List captured trace files and their sizes.
Note: traces can be viewed at https://ui.perfetto.dev/

## Completion
Update progress.json:
```json
{
  "phase": "profile",
  "phases_completed": ["env", "config", "benchmark", "benchmark-analyze", "profile"],
  "current_step": "profiling complete",
  "details": {
    "profile_runs": <N>,
    "trace_files": [<list of trace files>]
  }
}
```
