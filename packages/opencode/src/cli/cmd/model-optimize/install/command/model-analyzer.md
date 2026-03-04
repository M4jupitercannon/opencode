---
description: "Model analysis pipeline: environment setup, serving, profiling, and bottleneck analysis. Usage: /model-analyze <model_name> [output_dir]"
agent: model-opt
---

# Model Analysis Pipeline (Phases 0–4)

## Target
- **HuggingFace Model**: $1
- **Output Directory**: $2 (if not specified, use `/tmp/model_opt_<model_short_name>`)

## First Steps
1. Parse model name from `$1`
2. Determine output directory: `$2` if provided, else `/tmp/model_opt_<short_name>` (**MUST be outside the working directory**)
3. Create directory structure + `.gitignore` (exclude venv/, model/, *.safetensors, etc.)
4. Copy helper scripts from `~/.config/opencode/scripts/` to `<output_dir>/scripts/`

## ⚠️ CRITICAL RULES
- **Docker mode** (preferred): If `env_info.json` has `env_type: "docker"`, search available docker images on dockerhub(rocm/vllm-dev:nightly preferred). prefix commands with `docker exec $CONTAINER_NAME bash -c "..."`. Set `HIP_VISIBLE_DEVICES=$BEST_GPU`.
- **venv mode** (fallback): If `env_type: "venv"`, activate venv: `source <output_dir>/venv/bin/activate`
- **ALL vLLM commands MUST redirect output to log files** (`&> logfile`) — NEVER dump vLLM logs into bash output
- **ALL decisions MUST be data-driven** — read shapes from TraceLens `analysis_summary.json` / `unified_perf_summary.csv`, not hardcoded
- **Serving benchmarks MUST use `vllm bench serve --save-result`**

---

# Phase 0: Environment Setup 

## Goal
Search for latest docker images such as rocm/vllm-dev in dockerhub is compatible to vllm and platform, and create a container as isolated environment and install all required dependencies.

If there is no docker images available, then Create an isolated Python virtual environment with vLLM-rocm and all required dependencies.

## Steps

### 1. Detect host platform

```bash
ROCM_VERSION=$(cat /opt/rocm/.info/version 2>/dev/null | head -1 | cut -d'-' -f1 || echo "unknown")

# Try rocminfo first, fall back to kfd sysfs (works without /dev/kfd permissions)
GPU_ARCH=$(rocminfo 2>/dev/null | grep -oP 'gfx\w+' | head -1 || true)
if [ -z "$GPU_ARCH" ]; then
  # gfx_target_version is packed decimal: major*10000 + minor*100 + stepping
  # gfx string format: gfx{major}{minor:hex}{stepping:hex} e.g. 120001→gfx1201, 90010→gfx90a
  GPU_ARCH=$(cat /sys/class/kfd/kfd/topology/nodes/*/properties 2>/dev/null \
    | grep gfx_target_version | awk '$2 > 0 {v=$2; maj=int(v/10000); min=int((v%10000)/100); step=v%100; printf "gfx%d%x%x\n", maj, min, step}' \
    | head -1 || echo "unknown")
fi

DOCKER_OK=$(docker info >/dev/null 2>&1 && echo "yes" || echo "no")

echo "ROCm: $ROCM_VERSION  GPU: $GPU_ARCH  Docker: $DOCKER_OK"
```

### 2. Search for a compatible Docker image (preferred path)

Look for a `rocm/vllm-dev` nightly image whose ROCm version and GPU architecture match the host.
Use [Docker Hub tags](https://hub.docker.com/r/rocm/vllm-dev/tags) or the Docker CLI to find candidates.

```bash
# Determine target tag pattern based on GPU arch
# CDNA (gfx90a, gfx942, …) → mainline tags
# RDNA (gfx1100, gfx1201, …) → navi-specific tags
if [[ "$GPU_ARCH" == gfx9* ]]; then
  TAG_PATTERN="nightly_main"
elif [[ "$GPU_ARCH" == gfx1* ]]; then
  TAG_PATTERN="navi"
else
  TAG_PATTERN=""
fi

echo "Searching rocm/vllm-dev tags matching: $TAG_PATTERN"

# List recent tags from Docker Hub (requires internet)
TAGS=$(curl -sL "https://hub.docker.com/v2/repositories/rocm/vllm-dev/tags?page_size=20&ordering=last_updated" \
  | python3 -c "
import json, sys
data = json.load(sys.stdin)
for t in data.get('results', []):
    print(t['name'])
" 2>/dev/null || echo "")

if [ -n "$TAGS" ]; then
  echo "Recent tags:"
  echo "$TAGS" | head -10
  # Pick the latest tag matching our pattern
  IMAGE_TAG=$(echo "$TAGS" | grep -i "$TAG_PATTERN" | head -1)
fi

if [ -z "$IMAGE_TAG" ]; then
  echo "No matching Docker image found — will fall back to venv setup"
fi
```

### 3. Create container (if image found)

```bash
CONTAINER_NAME="vllm_model_opt"

if [ -n "$IMAGE_TAG" ]; then
  IMAGE="rocm/vllm-dev:$IMAGE_TAG"
  echo "Using image: $IMAGE"

  # Pull if not already local
  docker pull "$IMAGE" 2>/dev/null

  # Check if container already exists
  if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Container $CONTAINER_NAME already exists — starting it"
    docker start "$CONTAINER_NAME"
  else
    docker run -d \
      --name "$CONTAINER_NAME" \
      --device=/dev/kfd --device=/dev/dri \
      --group-add video --group-add render \
      --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
      --shm-size 16G \
      -v <output_dir>:/workspace/output \
      -p 8192:8192 -p 8193:8193 \
      "$IMAGE" sleep infinity
    echo "Created container: $CONTAINER_NAME"
  fi

  # Verify inside container
  docker exec "$CONTAINER_NAME" bash -c "
    python3 -c \"
import torch, vllm
print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')
print(f'vLLM {vllm.__version__}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    free, total = torch.cuda.mem_get_info(i)
    print(f'  cuda:{i} — {torch.cuda.get_device_name(i)}, free={free/1e9:.1f}GB, total={total/1e9:.1f}GB')
\"
  "

  # Record which GPU to use (pick the one with most free memory)
  BEST_GPU=$(docker exec "$CONTAINER_NAME" python3 -c "
import torch
best, best_free = 0, 0
for i in range(torch.cuda.device_count()):
    free, _ = torch.cuda.mem_get_info(i)
    if free > best_free:
        best, best_free = i, free
print(best)
")
  echo "Best GPU: cuda:$BEST_GPU"

  # Save environment info
  docker exec "$CONTAINER_NAME" bash -c "
    mkdir -p /workspace/output
    python3 -c \"
import json, torch, vllm
info = {
    'env_type': 'docker',
    'container': '$CONTAINER_NAME',
    'image': '$IMAGE',
    'pytorch': torch.__version__,
    'vllm': vllm.__version__,
    'gpu_count': torch.cuda.device_count(),
    'best_gpu': $BEST_GPU,
}
with open('/workspace/output/env_info.json', 'w') as f:
    json.dump(info, f, indent=2)
print(json.dumps(info, indent=2))
\"
  "
fi
```

### 4. Fallback: venv setup (if no Docker image)

Only execute this section if Step 2/3 did not find a suitable image.

```bash
if [ -z "$IMAGE_TAG" ]; then
  echo "Setting up venv environment..."
  cd <output_dir>

  if [ ! -d "venv" ]; then
    python3 -m venv venv --system-site-packages
    echo "Created venv with system site-packages access"
  fi

  source venv/bin/activate

  # Install vLLM
  python3 -c "import vllm; print(f'vLLM {vllm.__version__}')" 2>/dev/null || \
    pip install vllm --extra-index-url https://wheels.vllm.ai/rocm/

  # Install other dependencies
  python3 -c "import transformers" 2>/dev/null || pip install transformers
  python3 -c "import accelerate" 2>/dev/null || pip install accelerate

  # Verify
  python3 -c "
import torch, vllm
print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')
print(f'vLLM {vllm.__version__}')
print(f'GPU: {torch.cuda.get_device_name()}')
"

  # Save environment info
  python3 -c "
import json, torch, vllm
info = {
    'env_type': 'venv',
    'pytorch': torch.__version__,
    'vllm': vllm.__version__,
    'gpu_count': torch.cuda.device_count(),
}
with open('<output_dir>/env_info.json', 'w') as f:
    json.dump(info, f, indent=2)
print(json.dumps(info, indent=2))
"
fi
```

### 5. Copy helper scripts into the output directory

```bash
mkdir -p <output_dir>/scripts
cp ~/.config/opencode/scripts/*.py <output_dir>/scripts/ 2>/dev/null
ls <output_dir>/scripts/
```

### 6. Update progress.json
Update progress.json: phase="env", phases_completed.append("env")

⚠️ **CRITICAL for all subsequent phases**: If `env_type` is `docker` in `env_info.json`, prefix all commands with `docker exec $CONTAINER_NAME bash -c "..."` and use `/workspace/output` as the output directory inside the container. Set `HIP_VISIBLE_DEVICES=$BEST_GPU` to target the GPU with the most free memory.

---

# Phase 1: Model Serving with vLLM 

## Goal
Start the model using `vllm serve` and verify it works. vLLM handles model download automatically.

## ⚠️ vLLM Mode
In vLLM mode, there is NO need to:
- Manually download the model (vLLM auto-downloads from HuggingFace)
- Write a demo inference script
- Fix compatibility issues manually

## ⚠️ CRITICAL: Never dump vLLM logs into bash output
**ALL vLLM commands MUST redirect output to log files.** vLLM logs are thousands of lines and will break the session context.

## Steps

### 1. Test vLLM serve
```bash
source <output_dir>/venv/bin/activate

# Start vLLM — ALL output to log file, NEVER to stdout
vllm serve $1 \
  --dtype auto \
  --max-model-len 2048 \
  --port 8192 \
  --disable-log-requests &> <output_dir>/vllm_serve.log &
VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"

# Wait for server (silent polling)
for i in $(seq 1 60); do
  curl -s http://localhost:8192/health > /dev/null 2>&1 && break
  sleep 5
done
curl -s http://localhost:8192/health > /dev/null 2>&1 && echo "✓ Server ready" || echo "✗ Server failed — check <output_dir>/vllm_serve.log"

# Quick inference test (only show the result, not vllm internals)
curl -s http://localhost:8192/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "$1", "prompt": "Hello, I am", "max_tokens": 20}' \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print('✓ Inference OK' if 'choices' in d else f'✗ Error: {d}')"

# Kill the test server
kill $VLLM_PID 2>/dev/null; wait $VLLM_PID 2>/dev/null
```

### 2. Record model config
```bash
source <output_dir>/venv/bin/activate
python3 -c "
from transformers import AutoConfig
config = AutoConfig.from_pretrained('$1', trust_remote_code=True)
import json
info = {
    'model_type': getattr(config, 'model_type', 'unknown'),
    'num_hidden_layers': getattr(config, 'num_hidden_layers', None),
    'hidden_size': getattr(config, 'hidden_size', None),
    'num_attention_heads': getattr(config, 'num_attention_heads', None),
    'num_key_value_heads': getattr(config, 'num_key_value_heads', None),
    'intermediate_size': getattr(config, 'intermediate_size', None),
    'vocab_size': getattr(config, 'vocab_size', None),
}
print(json.dumps(info, indent=2))
with open('<output_dir>/model_config.json', 'w') as f:
    json.dump(info, f, indent=2)
"
```

### 3. Update progress.json
Update progress.json: phases_completed.append("download"), phases_completed.append("demo"), phases_completed.append("compatibility")

> **Note**: In vLLM mode, Phase 1 covers download + demo + compatibility in one step.


---

# Phase 2: (Covered by Phase 1 in vLLM mode) 

> In vLLM mode, demo generation is handled by Phase 1 (`vllm serve`). Skip this phase.

Update progress.json if not already done.


---

# Phase 3: (Covered by Phase 1 in vLLM mode) 

> In vLLM mode, compatibility fixes are handled by vLLM itself. Skip this phase.

If vLLM serve failed in Phase 1, debug using vLLM logs (check `--dtype`, `--tensor-parallel-size`, `--max-model-len`).

Update progress.json if not already done.


---

# Phase 4: Performance Profiling 

## Goal
Benchmark vLLM serving throughput AND collect GPU kernel trace for bottleneck analysis.

## ⚠️ CRITICAL: ALL vLLM output MUST go to log files
**NEVER let vLLM stdout/stderr appear in bash output.** Always use `&> logfile`.
**For `vllm bench serve`, redirect to file and only extract key metrics.**

## Step 1: Baseline Throughput Benchmark

```bash
source <output_dir>/venv/bin/activate

# Start vLLM — ALL output to log file
vllm serve $1 \
  --dtype auto \
  --max-model-len 4096 \
  --port 8192 \
  --disable-log-requests &> <output_dir>/vllm_baseline.log &
VLLM_PID=$!
echo "Baseline vLLM PID: $VLLM_PID (log: <output_dir>/vllm_baseline.log)"

# Wait silently
for i in $(seq 1 60); do
  curl -s http://localhost:8192/health > /dev/null 2>&1 && break
  sleep 5
done
curl -s http://localhost:8192/health > /dev/null 2>&1 && echo "✓ Server ready" || { echo "✗ Failed — check vllm_baseline.log"; tail -5 <output_dir>/vllm_baseline.log; }

# Run benchmark — output to file, then extract only key metrics
vllm bench serve \
  --model $1 --port 8192 \
  --dataset-name random \
  --input-len 1024 --output-len 1024 \
  --num-prompts 100 --max-concurrency 16 \
  --request-rate inf --save-result \
  --result-dir <output_dir>/profile --result-filename baseline_benchmark.json \
  --label baseline &> <output_dir>/profile/bench_baseline.log

kill $VLLM_PID 2>/dev/null; wait $VLLM_PID 2>/dev/null

# Show ONLY key metrics (not the full benchmark output)
python3 -c "
import json
with open('<output_dir>/profile/baseline_benchmark.json') as f:
    d = json.load(f)
print('=== Baseline Metrics ===')
for k in ['output_throughput','request_throughput','mean_tpot_ms','mean_ttft_ms','mean_itl_ms','completed']:
    print(f'  {k}: {d.get(k,\"N/A\")}')
"
```

## Step 2: Collect Kernel Trace

### ⛔ HARD REQUIREMENTS — all three are mandatory, do NOT skip any:

1. **`--enforce-eager`** on the `vllm serve` command line — disables CUDA Graphs so every GPU kernel retains its `External id` linkage back to the CPU op that launched it. Without this, the trace has GPU kernels but no way to correlate them to operator shapes.

2. **`--profiler-config`** with a JSON object containing **`torch_profiler_record_shapes: true`** — tells the PyTorch profiler to record `Input Dims` on every CPU operator event. Without this, the trace has CPU ops but their shapes are empty.

3. **`/start_profile` → send requests → `/stop_profile`** API sequence — vLLM does NOT profile from startup. You must explicitly start and stop profiling via the HTTP API. Without this sequence, no trace file is written at all.

**If any of the three is missing, `analyze_kernels.py` WILL produce 0% attributed shapes. You MUST re-collect the trace — do NOT proceed with bad data.**

### Docker path note
When running inside Docker, the profiler writes to the **container-side path**. Use `/workspace/output/profile/traces` (not the host path) inside `--profiler-config`.
Do NOT set the `VLLM_TORCH_PROFILER_DIR` environment variable — it is deprecated (removed in v0.15+). Use `torch_profiler_dir` inside `--profiler-config` instead.

### ⚠️ Two trace files are written
vLLM writes **two** separate trace files per profiling session:
- **`*async_llm*`** — frontend-only trace (CPU activity only, NO GPU kernels, NO shapes). **This file is USELESS for shape analysis.**
- **`*rank-0*`** — worker trace (CPU + CUDA activities, has `cpu_op` events with `Input Dims`, has `kernel` events with `External id`). **This is the file you need.**

Setting `ignore_frontend: true` in the profiler config suppresses the frontend trace entirely, which also reduces profiling overhead.

**When selecting the trace file in subsequent steps, you MUST pick the `rank-0` worker trace, NOT the `async_llm` trace.**

```bash
source <output_dir>/venv/bin/activate
mkdir -p <output_dir>/profile/traces

# Determine the correct trace directory path
# Docker mode: use container-side path; venv mode: use host path
ENV_TYPE=$(python3 -c "import json; print(json.load(open('<output_dir>/env_info.json')).get('env_type','venv'))" 2>/dev/null || echo "venv")
if [ "$ENV_TYPE" = "docker" ]; then
  TRACE_DIR="/workspace/output/profile/traces"
else
  TRACE_DIR=$(realpath <output_dir>/profile/traces)
fi

# Build profiler config JSON
# ⚠️ torch_profiler_record_shapes: true is the critical flag for shape analysis
# ⚠️ ignore_frontend: true suppresses the useless async_llm trace (CPU-only, no GPU data)
PROFILER_CFG=$(python3 -c "
import json; print(json.dumps({
  'profiler': 'torch',
  'torch_profiler_dir': '$TRACE_DIR',
  'torch_profiler_record_shapes': True,
  'torch_profiler_with_stack': True,
  'torch_profiler_with_flops': True,
  'torch_profiler_with_memory': False,
  'torch_profiler_use_gzip': True,
  'ignore_frontend': True,
}))
")
echo "Profiler config: $PROFILER_CFG"

# VLLM_RPC_TIMEOUT: trace flush after /stop_profile can take minutes for large models.
# vLLM docs recommend 30min for 100 reqs on 70B. Default 10s causes timeouts.
export VLLM_RPC_TIMEOUT=1800000

# Start vLLM WITH profiler + enforce-eager — output to log file
# ⚠️ BOTH --enforce-eager AND --profiler-config are MANDATORY
# ⚠️ Do NOT set VLLM_TORCH_PROFILER_DIR env var — it is deprecated.
#    torch_profiler_dir in --profiler-config is the correct way.
vllm serve $1 \
  --dtype auto \
  --max-model-len 4096 \
  --port 8193 \
  --disable-log-requests \
  --enforce-eager \
  --profiler-config "$PROFILER_CFG" &> <output_dir>/vllm_trace.log &
VLLM_PID=$!
echo "Trace vLLM PID: $VLLM_PID (log: <output_dir>/vllm_trace.log)"

# Wait for server ready
for i in $(seq 1 60); do
  curl -s http://localhost:8193/health > /dev/null 2>&1 && break
  sleep 5
done
curl -s http://localhost:8193/health > /dev/null 2>&1 && echo "Trace server ready" || { echo "FAILED"; tail -5 <output_dir>/vllm_trace.log; kill $VLLM_PID 2>/dev/null; exit 1; }

# Verify --enforce-eager is active (check log for confirmation)
grep -qi "enforce.eager\|CUDA graphs.*disabled\|eager mode" <output_dir>/vllm_trace.log && echo "enforce-eager: confirmed" || echo "WARNING: enforce-eager not confirmed in log"

# ⚠️ CRITICAL: Start profiling via API — without this, NO trace is written
PROFILE_RESP=$(curl -s -X POST http://localhost:8193/start_profile)
echo "start_profile response: $PROFILE_RESP"
echo "$PROFILE_RESP" | grep -qi "error" && { echo "FAILED to start profiler"; kill $VLLM_PID 2>/dev/null; exit 1; }
sleep 2

# Send requests for trace — output to file
vllm bench serve \
  --model $1 --port 8193 \
  --dataset-name random \
  --input-len 1024 --output-len 1024 \
  --num-prompts 30 --max-concurrency 16 \
  --request-rate inf --save-result \
  --result-dir <output_dir>/profile --result-filename trace_benchmark.json \
  --label trace &> <output_dir>/profile/bench_trace.log

# ⚠️ CRITICAL: Stop profiling via API — this flushes the trace to disk
STOP_RESP=$(curl -s -X POST http://localhost:8193/stop_profile)
echo "stop_profile response: $STOP_RESP"
sleep 15
kill $VLLM_PID 2>/dev/null; wait $VLLM_PID 2>/dev/null

echo "Trace files:"
ls -lh <output_dir>/profile/traces/ 2>/dev/null | head -5
```

### ⛔ Step 2b: Verify Trace Has Shape Data (MANDATORY before proceeding)

**Do NOT skip this step.** If this verification fails, you MUST go back and re-collect the trace with the correct flags.

The `analyze_kernels.py` script accepts a **directory** as `-i` input and will auto-select the worker trace (`*rank-0*`), rejecting `async_llm` frontend traces. It validates the trace contents and exits non-zero if the trace is unsuitable.

```bash
cp <output_dir>/scripts/analyze_kernels.py <output_dir>/profile/
# Run with --validate-only for validation (no TraceLens required)
python3 <output_dir>/profile/analyze_kernels.py -i <output_dir>/profile/traces/ --validate-only 2>&1 | head -20

# If the above exits non-zero, the trace is invalid for analysis.
# Check the output for the specific failure reason and re-collect with:
#   1. --enforce-eager on vllm serve
#   2. --profiler-config with torch_profiler_record_shapes: true AND ignore_frontend: true
#   3. /start_profile API call BEFORE requests, /stop_profile AFTER
```

## Step 3: Split Trace & Run TraceLens Performance Analysis

This step uses TraceLens to:
1. **Split** the trace into phase-specific sub-traces (prefill-decode, decode-only) using steady-state detection
2. **Analyze** each phase trace with TraceLens standalone analysis (roofline model, op breakdown, GPU timeline)

This replaces the old manual kernel extraction and correlation steps. TraceLens provides:
- Accurate per-op performance models (GFLOPS, data movement, arithmetic intensity)
- Roofline analysis (memory-bound vs compute-bound classification)
- GPU timeline breakdown (busy/idle/communication)
- Per-phase comparison (prefill vs decode have fundamentally different characteristics)

### TraceLens availability

The script auto-discovers TraceLens by searching common paths (`/TraceLens-internal`, `/TraceLens`, `~/TraceLens`, etc.)
and the `TRACELENS_DIR` environment variable. **If TraceLens is not found, it is automatically cloned from GitHub:**

```
https://github.com/AMD-AGI/TraceLens.git
```

You can also provide the path explicitly via `--tracelens-dir`, or pre-install TraceLens:
```bash
# Option A: Clone into container (Docker mode)
docker exec $CONTAINER_NAME git clone --depth 1 https://github.com/AMD-AGI/TraceLens.git /TraceLens

# Option B: Copy a local checkout into container
docker cp /path/to/TraceLens-internal $CONTAINER_NAME:/TraceLens-internal

# Option C: Set environment variable
export TRACELENS_DIR=/path/to/TraceLens
```

### Run the full analysis pipeline

```bash
cd <output_dir>/profile
cp <output_dir>/scripts/analyze_kernels.py .

# Full analysis: validate → split trace → run TraceLens on each phase
# TraceLens is auto-detected or auto-cloned if not found
python3 analyze_kernels.py \
  -i traces/ \
  -o .
```

This produces:
- `phase_traces/` — split trace files (combined steady-state, prefill-decode, decode-only)
- `prefilldecode_report/` — TraceLens CSVs for prefill-decode phase
- `decode_report/` — TraceLens CSVs for decode-only phase
- `analysis_summary.json` — machine-readable summary of all phases

Key output files per phase:
- `unified_perf_summary.csv` — per-op roofline analysis (GFLOPS, TB/s, arithmetic intensity)
- `ops_summary_by_category.csv` — time breakdown by op category (GEMM, Attention, Norm, etc.)
- `ops_summary.csv` — time breakdown by individual op
- `gpu_timeline.csv` — GPU utilization (busy/idle/communication split)
- `kernel_summary.csv` — raw kernel-level statistics

## Step 4: Generate bottlenecks.json from TraceLens Results

Read TraceLens standalone analysis CSVs and generate the bottlenecks manifest for downstream phases.
Uses `ops_summary.csv` for the op-level time breakdown and enriches each entry with roofline
metrics from `unified_perf_summary.csv` (GFLOPS, TB/s, arithmetic intensity, bound type).

```bash
cd <output_dir>/profile
python3 -c "
import csv, json, os, glob, ast

# ══════════════════════════════════════════════════════════════════════════
# Platform peak specs — used for roofline efficiency calculation.
# These are MAX ACHIEVABLE (measured), not theoretical peak.
# Source: TraceLens platform_specs.py / AMD ROCm benchmarks.
# ══════════════════════════════════════════════════════════════════════════
PLATFORM_PEAKS = {
    'MI300X': {'mem_bw_tbps': 5.3, 'matrix_bf16_tflops': 708, 'matrix_fp16_tflops': 654, 'matrix_fp8_tflops': 1273},
    'MI325X': {'mem_bw_tbps': 6.0, 'matrix_bf16_tflops': 843, 'matrix_fp16_tflops': 794, 'matrix_fp8_tflops': 1519},
    'MI355X': {'mem_bw_tbps': 8.0, 'matrix_bf16_tflops': 1686, 'matrix_fp16_tflops': 1686, 'matrix_fp8_tflops': 3567},
}

# Auto-detect GPU or default to MI300X
gpu_name = 'MI300X'
try:
    import subprocess
    out = subprocess.check_output(['rocm-smi', '--showproductname'], text=True, timeout=5)
    for name in PLATFORM_PEAKS:
        if name in out:
            gpu_name = name; break
except Exception:
    pass

peak = PLATFORM_PEAKS[gpu_name]
PEAK_MEM_BW_TBPS = peak['mem_bw_tbps']
PEAK_BF16_TFLOPS = peak['matrix_bf16_tflops']
RIDGE_POINT = PEAK_BF16_TFLOPS / PEAK_MEM_BW_TBPS  # FLOPS/Byte where memory→compute transition
print(f'Platform: {gpu_name}  Peak BW: {PEAK_MEM_BW_TBPS} TB/s  Peak BF16: {PEAK_BF16_TFLOPS} TFLOPS  Ridge: {RIDGE_POINT:.1f} FLOPS/Byte')

def roofline_eff(flops_byte, tflops_s, tb_s):
    if flops_byte < RIDGE_POINT:
        eff = (tb_s / PEAK_MEM_BW_TBPS) * 100 if PEAK_MEM_BW_TBPS else 0
        return eff, 'memory'
    else:
        eff = (tflops_s / PEAK_BF16_TFLOPS) * 100 if PEAK_BF16_TFLOPS else 0
        return eff, 'compute'

# ══════════════════════════════════════════════════════════════════════════
# Locate report directories
# ══════════════════════════════════════════════════════════════════════════
report_dirs = {}
for d in sorted(glob.glob('*_report')):
    if os.path.isdir(d) and os.path.isfile(os.path.join(d, 'ops_summary.csv')):
        report_dirs[d.replace('_report', '')] = d

phase_key = next((k for k in ('decode', 'prefilldecode', 'combined', 'full') if k in report_dirs), None)
if not phase_key:
    phase_key = list(report_dirs.keys())[0] if report_dirs else None
if not phase_key:
    print('ERROR: No TraceLens report directories found.'); exit(1)
report_dir = report_dirs[phase_key]
print(f'Using phase: {phase_key}  (report dir: {report_dir})')

# ── 1. Read ops_summary.csv ──
ops = []
with open(os.path.join(report_dir, 'ops_summary.csv')) as f:
    for row in csv.DictReader(f): ops.append(row)

# ── 2. Read unified_perf_summary.csv (GEMM roofline data) ──
unified = []
unified_path = os.path.join(report_dir, 'unified_perf_summary.csv')
if os.path.isfile(unified_path):
    with open(unified_path) as f:
        for row in csv.DictReader(f): unified.append(row)

op_roofline = {}
for row in unified:
    name = row.get('name', '')
    op_roofline.setdefault(name, [])
    entry = {'input_dims': row.get('Input Dims', ''), 'count': int(float(row.get('operation_count', 0) or 0))}
    for col in ('GFLOPS', 'Data Moved (MB)', 'FLOPS/Byte', 'TB/s_mean', 'TFLOPS/s_mean', 'Percentage (%)'):
        val = row.get(col, '')
        if val:
            try: entry[col] = float(val)
            except ValueError: pass
    entry['compute_spec'] = row.get('Compute Spec', '')
    entry['has_perf_model'] = row.get('has_perf_model', '').lower() == 'true'
    op_roofline[name].append(entry)

# ── 3. Read ops_unique_args.csv (Attention kernel breakdown) ──
unique_args = []
unique_path = os.path.join(report_dir, 'ops_unique_args.csv')
if os.path.isfile(unique_path):
    with open(unique_path) as f:
        for row in csv.DictReader(f): unique_args.append(row)

attn_shapes = [r for r in unique_args if r.get('op category', '') == 'SDPA_fwd']

# ── 4. Build bottlenecks list ──
bottlenecks = []
for row in ops:
    name = row.get('name', '')
    pct = float(row.get('Percentage (%)', 0))
    total_ms = float(row.get('total_direct_kernel_time_ms', 0))
    count = int(float(row.get('Count', 0)))
    cats = row.get('Categories', '')

    optimizable = True; reason = ''
    if 'GEMM' in cats:       reason = 'GEMM/rocBLAS'; optimizable = False
    elif 'SDPA' in cats or 'attention' in name.lower(): reason = 'Attention'; optimizable = True
    elif 'norm' in name.lower() or 'rms' in name.lower(): reason = 'Normalization'; optimizable = True
    elif 'silu' in name.lower() or 'gelu' in name.lower(): reason = 'Activation'; optimizable = True
    elif 'copy' in name.lower() or 'memcpy' in name.lower(): reason = 'Memory op'; optimizable = False
    elif 'elementwise' in cats: reason = 'Elementwise'; optimizable = True
    else: reason = 'Other'

    entry = {'name': name, 'count': count, 'total_dur_ms': total_ms, 'cuda_time_percent': pct,
             'optimizable': optimizable, 'reason': reason, 'phase': phase_key, 'categories': cats}

    shapes = op_roofline.get(name, [])
    if shapes:
        top_shapes = sorted(shapes, key=lambda s: s.get('Percentage (%)', 0), reverse=True)[:5]
        entry['top_shapes'] = []
        for s in top_shapes:
            se = {'input_dims': s['input_dims'], 'count': s['count']}
            for k, jk in [('GFLOPS','gflops'),('Data Moved (MB)','data_moved_mb'),('FLOPS/Byte','flops_per_byte'),
                          ('TFLOPS/s_mean','tflops_per_s'),('TB/s_mean','tb_per_s'),('Percentage (%)','pct_of_total')]:
                if k in s: se[jk] = s[k]
            if s.get('compute_spec'): se['compute_spec'] = s['compute_spec']
            if 'FLOPS/Byte' in s and 'TFLOPS/s_mean' in s and 'TB/s_mean' in s:
                eff, bound = roofline_eff(s['FLOPS/Byte'], s['TFLOPS/s_mean'], s['TB/s_mean'])
                se['roofline_efficiency_pct'] = round(eff, 2)
                se['bound'] = bound
            entry['top_shapes'].append(se)
    bottlenecks.append(entry)

# ══════════════════════════════════════════════════════════════════════════
# 5. Print summary
# ══════════════════════════════════════════════════════════════════════════
total_ms = sum(b['total_dur_ms'] for b in bottlenecks)
print(f'\nTotal GPU time ({phase_key}): {total_ms:.2f} ms')
print(f'Top bottlenecks:')
print(f'  {\"#\":>3s}  {\"Op\":45s} {\"Time%\":>6s} {\"Time(ms)\":>9s} {\"Count\":>6s}  Category')
print(f'  {\"─\"*85}')
for i, b in enumerate(bottlenecks[:15], 1):
    flag = '🔒' if not b['optimizable'] else '  '
    print(f'  {i:3d}. {flag}{b[\"name\"][:43]:<43s} {b[\"cuda_time_percent\"]:5.1f}% {b[\"total_dur_ms\"]:9.2f} {b[\"count\"]:6d}  {b[\"reason\"]}')
    for s in b.get('top_shapes', [])[:2]:
        tflops = f'{s[\"tflops_per_s\"]:.1f} TFLOPS/s' if 'tflops_per_s' in s else ''
        tbps = f'{s[\"tb_per_s\"]:.2f} TB/s' if 'tb_per_s' in s else ''
        print(f'       └─ {s[\"input_dims\"][:60]:60s} {s.get(\"pct_of_total\",0):5.1f}%  {tflops}  {tbps}')

# ══════════════════════════════════════════════════════════════════════════
# 6. GEMM Roofline Efficiency (per-shape)
# ══════════════════════════════════════════════════════════════════════════
gemm_shapes = []
for row in unified:
    if row.get('op category', '') != 'GEMM': continue
    fb = row.get('FLOPS/Byte', ''); ts = row.get('TFLOPS/s_mean', ''); bs = row.get('TB/s_mean', '')
    if not (fb and ts and bs): continue
    fb, ts, bs = float(fb), float(ts), float(bs)
    eff, bound = roofline_eff(fb, ts, bs)
    gemm_shapes.append({
        'dims': row.get('Input Dims',''), 'count': int(float(row.get('operation_count',0) or 0)),
        'pct': float(row.get('Percentage (%)',0)), 'gflops': float(row.get('GFLOPS',0)),
        'data_mb': float(row.get('Data Moved (MB)',0)), 'flops_byte': fb,
        'tflops_s': ts, 'tb_s': bs, 'eff': eff, 'bound': bound,
    })

if gemm_shapes:
    print(f'\n{\"═\"*110}')
    print(f'  GEMM ROOFLINE EFFICIENCY ({phase_key} phase, {gpu_name})')
    print(f'  Peak: {PEAK_BF16_TFLOPS} TFLOPS BF16 | {PEAK_MEM_BW_TBPS} TB/s | Ridge: {RIDGE_POINT:.1f} FLOPS/Byte')
    print(f'{\"═\"*110}')
    print(f'  {\"Input Dims\":45s} {\"Time%\":>6s} {\"GFLOPS\":>8s} {\"Data(MB)\":>9s} {\"FL/B\":>7s} {\"TFLOPS/s\":>9s} {\"TB/s\":>7s} {\"Bound\":>8s} {\"Eff%\":>6s}')
    print(f'  {\"─\"*106}')
    for s in sorted(gemm_shapes, key=lambda x: x['pct'], reverse=True):
        dims_short = s['dims'].replace('((','(').replace('))',')')[:45]
        bound_tag = 'MEM' if s['bound'] == 'memory' else 'COMP'
        print(f'  {dims_short:45s} {s[\"pct\"]:5.1f}% {s[\"gflops\"]:8.1f} {s[\"data_mb\"]:9.1f} {s[\"flops_byte\"]:7.1f} {s[\"tflops_s\"]:9.2f} {s[\"tb_s\"]:7.3f} {bound_tag:>8s} {s[\"eff\"]:5.1f}%')

# ══════════════════════════════════════════════════════════════════════════
# 7. Attention Kernel Breakdown (per-shape from ops_unique_args.csv)
# ══════════════════════════════════════════════════════════════════════════
if attn_shapes:
    print(f'\n{\"═\"*110}')
    print(f'  ATTENTION (SDPA) KERNEL BREAKDOWN ({phase_key} phase)')
    print(f'  Note: TraceLens does not compute GFLOPS/TB/s for SDPA — showing kernel time breakdown')
    print(f'{\"═\"*110}')
    print(f'  {\"Input Dims\":50s} {\"Time%\":>6s} {\"Count\":>6s} {\"Avg(us)\":>9s} {\"Sub-kernels\":40s}')
    print(f'  {\"─\"*115}')
    for row in attn_shapes:
        dims = row.get('Input Dims','')[:50]
        pct = float(row.get('Percentage (%)',0))
        cnt = int(float(row.get('operation_count',0)))
        avg = float(row.get('total_direct_kernel_time_mean',0))
        # Parse kernel breakdown from trunc_kernel_details
        kdetails = row.get('trunc_kernel_details', '')
        sub_kernels = []
        if kdetails:
            try:
                parts = ast.literal_eval(kdetails.replace('np.float64(','').replace(')',''))
                for p in parts:
                    kname = p.get('name','')[:30]
                    kdur = p.get('mean_duration_us', 0)
                    sub_kernels.append(f'{kname}={kdur:.1f}us')
            except Exception:
                sub_kernels = ['(parse error)']
        sk_str = ', '.join(sub_kernels)[:40] if sub_kernels else ''
        print(f'  {dims:50s} {pct:5.1f}% {cnt:6d} {avg:9.1f} {sk_str:40s}')
    # Estimate attention mem BW utilization from raw kernel time + data size heuristic
    for row in attn_shapes:
        dims_str = row.get('Input Dims','')
        try:
            parsed = ast.literal_eval(dims_str)
            if len(parsed) >= 4:
                q_shape = parsed[0]; k_shape = parsed[1]; v_shape = parsed[2]
                batch = q_shape[0]; heads = q_shape[1]; head_dim = q_shape[2]
                kv_heads = k_shape[1]; seq_est = 1
                data_bytes = 2 * (batch * heads * head_dim + 2 * batch * kv_heads * head_dim + batch * heads * head_dim)
                data_mb = data_bytes / 1e6
                avg_us = float(row.get('total_direct_kernel_time_mean',0))
                if avg_us > 0:
                    est_tb_s = (data_mb / 1e6) / (avg_us / 1e6)
                    est_bw_eff = (est_tb_s / PEAK_MEM_BW_TBPS) * 100
                    print(f'  Attention est. data/call: {data_mb:.3f} MB, est. BW: {est_tb_s:.3f} TB/s, est. BW eff: {est_bw_eff:.1f}% (Q/K/V only, excl. KV cache)')
        except Exception:
            pass

print(f'\n{\"═\"*110}')

with open('bottlenecks.json', 'w') as f:
    json.dump(bottlenecks, f, indent=2)
print(f'Saved bottlenecks.json ({len(bottlenecks)} entries from {phase_key} phase)')

# Also save per-phase category summary for quick reference
phase_summary = {}
for label, rdir in report_dirs.items():
    cat_path = os.path.join(rdir, 'ops_summary_by_category.csv')
    if os.path.isfile(cat_path):
        with open(cat_path) as f:
            phase_summary[label] = list(csv.DictReader(f))
with open('phase_category_summary.json', 'w') as f:
    json.dump(phase_summary, f, indent=2)
print(f'Saved phase_category_summary.json ({len(phase_summary)} phases)')
"
```

## ⚠️ CRITICAL: Use roofline data for optimization decisions
The `unified_perf_summary.csv` in each phase report contains per-op roofline analysis with GFLOPS,
TB/s, arithmetic intensity, and compute spec. Use this to determine whether each operator is
**memory-bound** or **compute-bound**, and to prioritize optimization targets accordingly.
- **Memory-bound ops** (low arithmetic intensity): optimize memory access patterns, fusion
- **Compute-bound ops** (high arithmetic intensity): optimize compute throughput, tiling
- **Prefill phase**: larger batch dimensions, more compute-bound
- **Decode phase**: tiny batch dimensions (batch=1 per token), heavily memory-bound

## Step 5: Save model shapes

```bash
source <output_dir>/venv/bin/activate
python3 -c "
import json
from transformers import AutoConfig
c = AutoConfig.from_pretrained('$1', trust_remote_code=True)
shapes = {
    'hidden_size': getattr(c, 'hidden_size', None),
    'intermediate_size': getattr(c, 'intermediate_size', None),
    'num_attention_heads': getattr(c, 'num_attention_heads', None),
    'num_key_value_heads': getattr(c, 'num_key_value_heads', None),
    'head_dim': getattr(c, 'hidden_size', 0) // max(getattr(c, 'num_attention_heads', 1), 1),
    'num_hidden_layers': getattr(c, 'num_hidden_layers', None),
    'vocab_size': getattr(c, 'vocab_size', None),
}
with open('<output_dir>/profile/model_shapes.json', 'w') as f:
    json.dump(shapes, f, indent=2)
print(json.dumps(shapes, indent=2))
"
```

VALIDATE
```

**If validation fails, fix the issue and re-run from the failing step.**

Update progress.json: phases_completed.append("profile")


---
## TraceLens Roofline Analysis (Decode Phase)
Include the top ops from `decode_report/unified_perf_summary.csv` with roofline metrics.
This data was collected with `--enforce-eager` and `torch_profiler_record_shapes: true`.

| Op | Input Dims | % GPU Time | TFLOPS/s | TB/s | FLOPS/Byte | Bound |
|----|-----------|-----------|---------|------|-----------|-------|
| aten::mm | (16,4096)x(4096,24576) | 48.2% | 5.5 | 0.34 | 15.9 | Memory |
| ...      | ...   | ...       | ...     | ...  | ...       | ...   |

## TraceLens Roofline Analysis (Prefill-Decode Phase)
| Op | Input Dims | % GPU Time | TFLOPS/s | TB/s | FLOPS/Byte | Bound |
|----|-----------|-----------|---------|------|-----------|-------|
| aten::mm | (2048,4096)x(4096,24576) | 38.2% | 120.9 | 0.09 | 1293.5 | Compute |
| ...      | ...   | ...       | ...     | ...  | ...       | ...   |

## Files Generated
- profile/bottlenecks.json - Kernel bottleneck ranking
- profile/analysis_summary.json - TraceLens analysis summary (all phases)
- profile/phase_traces/ - Split trace files (steady-state, prefill-decode, decode-only)
- profile/prefilldecode_report/ - TraceLens CSVs for prefill-decode phase
- profile/decode_report/ - TraceLens CSVs for decode-only phase

# EXECUTION INSTRUCTIONS
Execute phases: 0 → 1 → 4 (Phases 2-3 handled by vLLM).
Begin with Phase 0.
