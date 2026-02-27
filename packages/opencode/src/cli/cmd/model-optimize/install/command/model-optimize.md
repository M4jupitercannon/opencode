---
description: "End-to-end vLLM model optimization pipeline. Usage: /model-optimize <model_name> [output_dir]"
agent: model-opt
---

# End-to-End vLLM Model Optimization Pipeline

## Target
- **HuggingFace Model**: $1
- **Output Directory**: $2 (if not specified, use `/tmp/model_opt_<model_short_name>`)

## First Steps
1. Parse model name from `$1`
2. Determine output directory: `$2` if provided, else `/tmp/model_opt_<short_name>` (**MUST be outside the working directory**)
3. Create directory structure + `.gitignore` (exclude venv/, model/, *.safetensors, etc.)
4. Copy helper scripts from `~/.config/opencode/scripts/` to `<output_dir>/scripts/`

## ⚠️ CRITICAL RULES
- **ALWAYS activate venv**: `source <output_dir>/venv/bin/activate`
- **ALL vLLM commands MUST redirect output to log files** (`&> logfile`) — NEVER dump vLLM logs into bash output
- **ALL decisions MUST be data-driven**
- **Optimized kernels MUST use @triton.jit** — torch rewrites are FORBIDDEN
- **Serving benchmarks MUST use `vllm bench serve --save-result`**

## ⛔ MANDATORY VALIDATION
After Phase 6 and Phase 7, run:
```bash
python <output_dir>/scripts/validate_pipeline.py --project-dir <output_dir> --phase all
```

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
    name = t['name']
    if '${TAG_PATTERN}' in name.lower():
        print(name)
" 2>/dev/null)
  if [ -n "$MATCHES" ]; then
    echo "Found matching tags (page $page):"
    echo "$MATCHES" | head -5
    # Prefer tag matching host ROCm version
    BEST=$(echo "$MATCHES" | grep "rocm${ROCM_MAJOR_MINOR}" | head -1)
    if [ -z "$BEST" ]; then
      BEST=$(echo "$MATCHES" | head -1)
    fi
    IMAGE_TAG="$BEST"
    break
  fi
done

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

⚠️ **For all subsequent phases**: if `env_type` is `docker`, prefix commands with `docker exec $CONTAINER_NAME bash -c "..."` and use `/workspace/output` as the output directory inside the container.


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

⚠️ **CRITICAL**: Two flags are mandatory for kernel shape analysis:
- `--enforce-eager` — disables CUDA Graphs so GPU kernels retain their `External id` linkage to CPU ops
- `--profiler-config` with `record_shapes=True` — records tensor `Input Dims` on every CPU op

Without both, `analyze_kernel_shapes.py` will produce only `(unattributed)` shapes.

```bash
source <output_dir>/venv/bin/activate
mkdir -p <output_dir>/profile/traces
TRACE_DIR=$(realpath <output_dir>/profile/traces)

# Build profiler config JSON (record_shapes is the key flag)
PROFILER_CFG=$(python3 -c "
import json; print(json.dumps({
  'profiler': 'torch',
  'torch_profiler_dir': '$TRACE_DIR',
  'torch_profiler_record_shapes': True,
  'torch_profiler_with_stack': True,
  'torch_profiler_with_flops': True,
  'torch_profiler_with_memory': False,
  'torch_profiler_use_gzip': True,
}))
")

# Start vLLM WITH profiler + enforce-eager — output to log file
VLLM_TORCH_PROFILER_DIR="$TRACE_DIR" \
vllm serve $1 \
  --dtype auto \
  --max-model-len 4096 \
  --port 8193 \
  --disable-log-requests \
  --enforce-eager \
  --profiler-config "$PROFILER_CFG" &> <output_dir>/vllm_trace.log &
VLLM_PID=$!
echo "Trace vLLM PID: $VLLM_PID (log: <output_dir>/vllm_trace.log)"

# Wait silently
for i in $(seq 1 60); do
  curl -s http://localhost:8193/health > /dev/null 2>&1 && break
  sleep 5
done
curl -s http://localhost:8193/health > /dev/null 2>&1 && echo "✓ Trace server ready" || { echo "✗ Failed"; tail -5 <output_dir>/vllm_trace.log; }

# Start profiling via API
curl -s -X POST http://localhost:8193/start_profile && echo "Profiler started"

# Send requests for trace — output to file
vllm bench serve \
  --model $1 --port 8193 \
  --dataset-name random \
  --input-len 1024 --output-len 1024 \
  --num-prompts 30 --max-concurrency 16 \
  --request-rate inf --save-result \
  --result-dir <output_dir>/profile --result-filename trace_benchmark.json \
  --label trace &> <output_dir>/profile/bench_trace.log

# Stop profiling and flush trace
curl -s -X POST http://localhost:8193/stop_profile && echo "Profiler stopped"
sleep 15
kill $VLLM_PID 2>/dev/null; wait $VLLM_PID 2>/dev/null

echo "Trace files:"
ls -lh <output_dir>/profile/traces/ 2>/dev/null | head -5
```

## Step 3: Extract Kernel Bottlenecks from Trace

```bash
cd <output_dir>/profile
cp <output_dir>/scripts/vllm_trace_extractor.py .

TRACE_FILE=$(ls -t traces/*.json traces/*.json.gz 2>/dev/null | head -1)
echo "Analyzing: $TRACE_FILE"

python3 vllm_trace_extractor.py -i "$TRACE_FILE" \
  --full-csv kernel_full.csv \
  --unique-csv kernel_unique.csv
```

## Step 4: Generate bottlenecks.json

```bash
cd <output_dir>/profile
python3 -c "
import csv, json

kernels = []
with open('kernel_unique.csv') as f:
    for row in csv.DictReader(f):
        kernels.append({
            'name': row['name'], 'count': int(row['count']),
            'total_dur_us': float(row['total_dur']),
            'avg_dur_us': float(row['avg_dur']),
            'median_dur_us': float(row['median_dur']),
        })

total = sum(k['total_dur_us'] for k in kernels)
bottlenecks = []
for k in kernels[:30]:
    pct = k['total_dur_us'] / total * 100 if total > 0 else 0
    name = k['name']
    optimizable = True
    reason = ''
    if 'Cijk_' in name or 'gemm' in name.lower() or 'hipblas' in name.lower():
        reason = 'GEMM/rocBLAS'; optimizable = False
    elif 'attn' in name.lower() or 'flash' in name.lower() or 'mha' in name.lower():
        reason = 'Attention'; optimizable = True
    elif 'norm' in name.lower() or 'rms' in name.lower():
        reason = 'Normalization'; optimizable = True
    elif 'elementwise' in name.lower() or 'vectorized' in name.lower():
        reason = 'Elementwise'; optimizable = True
    elif 'silu' in name.lower() or 'gelu' in name.lower():
        reason = 'Activation'; optimizable = True
    elif 'copy' in name.lower() or 'Cat' in name:
        reason = 'Memory op'; optimizable = False
    bottlenecks.append({**k, 'cuda_time_percent': pct, 'optimizable': optimizable, 'reason': reason})

# Print compact summary (not full data)
print(f'Total GPU time: {total/1000:.2f}ms, Top 10 kernels:')
for i, b in enumerate(bottlenecks[:10], 1):
    print(f\"  {i}. {b['name'][:45]:45s} {b['cuda_time_percent']:5.1f}%\")

with open('bottlenecks.json', 'w') as f:
    json.dump(bottlenecks, f, indent=2)
print(f'Saved bottlenecks.json ({len(bottlenecks)} kernels)')
"
```

## Step 5: Per-Shape Kernel Time Analysis

Analyze time proportion of **each shape** for each operator category.
This correlates GPU kernel durations with CPU-side operator shapes from the trace.

```bash
cd <output_dir>/profile
cp <output_dir>/scripts/analyze_kernel_shapes.py .

TRACE_FILE=$(ls -t traces/*.json traces/*.json.gz 2>/dev/null | head -1)
echo "Shape analysis on: $TRACE_FILE"

python3 analyze_kernel_shapes.py -i "$TRACE_FILE" -o .
```

This produces:
- `kernel_shape_analysis.json` — structured per-category, per-shape breakdown
- `kernel_shape_analysis.csv` — flat CSV for inspection

Example output:
```
  GEMM — 94.1% of total (15488.61ms, 10 shapes, 100% attributed)
  ──────────────────────────────────────────────────────────────────────────────────────
    Shape                                               %Total  %InCat  Time(ms)  Count  Avg(us)
    [4,4096]x[4096,24576]                                30.5%   32.4%  5023.08   9072    553.7
    [2,4096]x[4096,24576]                                15.4%   16.4%  2533.19   4644    545.5
    [4,12288]x[12288,4096]                               12.8%   13.6%  2112.64   9072    232.9
    [4,4096]x[4096,6144]                                  7.0%    7.4%  1144.21   9072    126.1
    ...

  Attention — 2.4% of total (390.56ms, 4 shapes, 100% attributed)
  ──────────────────────────────────────────────────────────────────────────────────────
    Shape                                               %Total  %InCat  Time(ms)  Count  Avg(us)
    [4,32,128]x[4,8,128]x[4,8,128]x[4,32,128]             1.5%   64.4%   251.52  27216      9.2
    ...
```
## ⚠️ CRITICAL: The shape must be real, do not mind the dimension
Use this data in Phase 5 to pick the right shapes for problem files and prioritize
which (operator, shape) combinations to optimize first.

## Step 6: Save model shapes

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

Update progress.json: phases_completed.append("profile")


---

# Phase 5: Generate Problem Files for Kernel Optimization 

## Goal
Convert bottleneck operators into Problem files for kernel-optimize.
**IMPORTANT**: Analyze operators for fusion opportunities BEFORE creating individual problem files.

## STEP 0: Review Per-Shape Kernel Analysis (from Phase 4)

Before creating problem files, review `<output_dir>/profile/kernel_shape_analysis.json` to understand:
- Which **operator categories** dominate GPU time (GEMM, Attention, Norm, Activation, ...)
- For each category, which **specific shapes** are the hottest
- Use the top (category, shape) pairs to set **priorities** and pick **exact dimensions** for problem files

```bash
cat <output_dir>/profile/kernel_shape_analysis.json | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(f'Total GPU time: {data[\"total_gpu_time_ms\"]:.2f}ms\n')
for cat in data['categories'][:8]:
    print(f'{cat[\"category\"]:12s} {cat[\"pct\"]:5.1f}%  ({cat[\"total_us\"]/1000:.2f}ms, {cat[\"num_shapes\"]} shapes)')
    for s in cat['shapes'][:5]:
        print(f'  {s[\"shape\"]:50s} {s[\"pct_of_total\"]:5.1f}% total, {s[\"count\"]:4d} calls, avg {s[\"avg_us\"]:.1f}us')
"
```

## STEP 1: Operator Fusion Analysis (CRITICAL)

A standalone `analyze_fusion.py` script is provided at `<output_dir>/scripts/analyze_fusion.py`.
Use it to detect fusable operator patterns:

```bash
cp <output_dir>/scripts/analyze_fusion.py <output_dir>/profile/
cd <output_dir>/profile
python analyze_fusion.py
cat fusion_opportunities.json
```

### Common Fusion Opportunities in LLMs

| Pattern | Operators to Fuse | Fused Name | Expected Speedup |
|---------|-------------------|------------|------------------|
| **ResidualNorm** | add + rmsnorm/layernorm | fused_residual_norm | 1.2-1.5x |
| **SwiGLU/GeGLU** | silu/gelu + mul | fused_swiglu | 1.3-1.8x |
| **BiasAdd** | matmul + add (bias) | fused_linear_bias | 1.1-1.3x |
| **RotaryEmbed** | rope_cos + rope_sin + cat | fused_rope | 1.2-1.5x |
| **QKV Projection** | 3x linear (q,k,v) | fused_qkv_proj | 1.2-1.4x |
| **MLP Block** | linear + activation + linear | fused_mlp | 1.3-2.0x |

## STEP 2: Create FUSED Problem Files (Priority)

**Create fused kernels BEFORE individual kernels!**

Use ACTUAL shapes from `<output_dir>/profile/kernel_shape_analysis.json` (per-shape time breakdown)
and `<output_dir>/profile/model_shapes.json`. Focus on the shapes with the highest `pct_of_total`.

### Example: Fused Residual + RMSNorm
```python
# problem_fused_residual_rmsnorm.py
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.float16))
        self.eps = eps
    def forward(self, hidden_states, residual):
        hidden_states = hidden_states + residual
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states

# ⚠️ Use shapes from shape_ranges.json!
batch_size = 1
seq_len = 64       # typical from profiling
hidden_size = 4096 # from model config

def get_inputs():
    return [
        torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float16, device='cuda'),
        torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float16, device='cuda'),
    ]
def get_init_inputs():
    return [hidden_size]
```

### Example: Fused SwiGLU
```python
# problem_fused_swiglu.py
import torch
import torch.nn as nn

class Model(nn.Module):
    def forward(self, gate, up):
        return torch.nn.functional.silu(gate) * up

batch_size, seq_len, intermediate_size = 1, 64, 11008
def get_inputs():
    return [
        torch.randn(batch_size, seq_len, intermediate_size, dtype=torch.float16, device='cuda'),
        torch.randn(batch_size, seq_len, intermediate_size, dtype=torch.float16, device='cuda'),
    ]
def get_init_inputs():
    return []
```

## STEP 3: Create Individual Problem Files (Lower Priority)

Only for operators that: cannot be fused, take > 5% time, and are not already optimized by vendor libs (rocBLAS GEMM).

### Common Operators

- **RMSNorm/LayerNorm**: `class Model` with weight param, forward does variance + rsqrt + mul
- **Attention**: `class Model` wrapping `torch.nn.functional.scaled_dot_product_attention`
- **SiLU/GELU**: `class Model` with activation function
- **RoPE**: `class Model` with cos/sin rotation

## STEP 4: Generate Optimization Manifest

Create `<output_dir>/problems/optimization_manifest.json`:

```json
{
  "model": "$1",
  "description": "Edit 'enabled' to control which optimizations to apply",
  "optimizations": [
    {"name": "fused_residual_rmsnorm", "file": "problem_fused_residual_rmsnorm.py",
     "type": "fused", "priority": "HIGH", "enabled": true},
    {"name": "linear_gemm", "file": "problem_linear.py",
     "type": "individual", "priority": "LOW", "enabled": false,
     "notes": "rocBLAS usually optimal"}
  ]
}
```

## Steps
1. Run fusion analysis
2. Create fused problem files (HIGH priority)
3. Create individual problem files (MEDIUM/LOW)
4. Generate optimization_manifest.json
5. Update progress.json



---

# Phase 6: Kernel Optimization 

## Goal
Write optimized Triton kernels for each problem file and verify speedup.

## ⚠️ NO external `opencode` command needed
Optimize kernels DIRECTLY in this session using the test scripts provided.

## Scripts Available
- `<output_dir>/scripts/kernel_test_runner.py` — test accuracy + benchmark
- `<output_dir>/scripts/kernel_finalize.py` — save best result to target file

## Workflow for EACH Problem File

For each `problem_*.py` file in `<output_dir>/problems/`:

### 1. Read the source file to understand the PyTorch operator
```bash
cat <output_dir>/problems/problem_XXX.py
```

### 2. Check GPU architecture
```bash
python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name()}, Arch: {torch.cuda.get_device_capability()}')"
```

### 3. Write the optimized Triton kernel
Create `<output_dir>/problems/problem_XXX_opt.py` with:
- `class ModelNew(nn.Module)` using `@triton.jit` Triton kernels
- Same `__init__` signature as `Model`
- Use `@triton.autotune` with 10-20 diverse configs

### 4. Test accuracy + benchmark
```bash
source <output_dir>/venv/bin/activate
python3 <output_dir>/scripts/kernel_test_runner.py \
  --src <output_dir>/problems/problem_XXX.py \
  --target <output_dir>/problems/problem_XXX_opt.py
```

The script prints: `RESULT_JSON: {"speedup": 1.5, "accuracy": "PASSED", ...}`

### 5. Iterate if needed
- Accuracy FAILED → fix kernel, re-run step 4
- Speedup too low → adjust block sizes, fusion strategy, re-run step 4

### 6. Finalize when satisfied
```bash
python3 <output_dir>/scripts/kernel_finalize.py \
  --target <output_dir>/problems/problem_XXX_opt.py
```

## Priority Order

| Priority | Kernel Type | Goal | Reason |
|----------|-------------|------|--------|
| **HIGH** | Fused Residual+RMSNorm | 1.5x | Memory traffic reduction |
| **HIGH** | Fused SwiGLU | 1.5x | Activation fusion |
| **HIGH** | Fused RoPE | 1.5x | Custom optimization |
| MEDIUM | Individual norms | 1.3x | If not covered by fused version |
| LOW | Linear/GEMM | 1.1x | rocBLAS usually optimal |
| **SKIP** | Simple add/copy | — | Overhead > benefit |

## When to SKIP a kernel
- If it's part of a fused kernel you already optimized
- If rocBLAS/vendor lib is already near-optimal
- If after 3 attempts speedup is < 1.0x at actual shapes

## Triton Optimization Guide

### Autotune Strategy
```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8, num_stages=2),
        # ... add 10-20 configs covering the space
    ],
    key=['N'],
)
```

### Common Patterns
- **Memory-bound**: Optimize access patterns, vectorization
- **Compute-bound**: Larger tiles, more arithmetic per memory access
- **Fused kernels**: Combine multiple ops to reduce memory traffic
- **FP32 accumulation**: Use `tl.float32` for acc, cast output at end

## After All Kernels Done

```bash
echo "=== Kernel Optimization Results ==="
cd <output_dir>/problems
for f in *_opt.py; do
  if [ -f "$f" ]; then
    tracker="${f%.py}_best.json"
    if [ -f "$tracker" ]; then
      speedup=$(python3 -c "import json; print(json.load(open('$tracker')).get('best_speedup', 0))")
      echo "  $f: ${speedup}x"
    fi
  fi
done
```

Copy successful optimizations to `<output_dir>/optimized/`:
```bash
cd <output_dir>/problems
for f in *_opt.py; do
  tracker="${f%.py}_best.json"
  if [ -f "$tracker" ]; then
    speedup=$(python3 -c "import json; d=json.load(open('$tracker')); print(d.get('best_speedup',0))")
    if python3 -c "exit(0 if $speedup > 1.0 else 1)"; then
      cp "$f" <output_dir>/optimized/
      echo "Copied $f (${speedup}x)"
    fi
  fi
done
```

Update progress.json: phases_completed.append("optimize")


---

# Phase 7: Integration & End-to-End Testing 

## Goal
Apply optimized kernels to vLLM via CustomOp and measure ACTUAL serving throughput.

## ⛔ MANDATORY: This phase REQUIRES real measured data

**This phase is NOT complete until:**
1. A patched vLLM server has ACTUALLY been started and served requests
2. `vllm bench serve` has been run against the patched server
3. `optimized_serving.json` has `"label": "optimized"` (NOT "baseline")
4. The validation script passes

**FORBIDDEN:**
- Estimating speedup with Amdahl's law
- Copying baseline numbers and modifying them
- Reporting "estimated" or "conservative" speedup
- Skipping the patched server benchmark

---

## Integration Mechanism: vLLM CustomOp.register_oot()

We use vLLM's OFFICIAL extension mechanism (not monkey-patching):
- Docs: https://docs.vllm.ai/en/latest/design/custom_op/
- Each optimized kernel is wrapped as a vLLM CustomOp subclass
- `CustomOp.register_oot()` replaces the default op at instantiation time
- If the optimized kernel fails, vLLM falls back to the default

---

## Step 1: Generate vLLM Plugin

The `generate_vllm_plugin.py` script auto-creates a plugin from `*_opt.py` files:

```bash
source <output_dir>/venv/bin/activate
cd <output_dir>/optimized

# Copy all *_opt.py from problems
cp <output_dir>/problems/*_opt.py . 2>/dev/null

# Generate the plugin
python3 <output_dir>/scripts/generate_vllm_plugin.py \
  --kernel-dir <output_dir>/optimized

# Verify generated files
ls -la vllm_plugin/
cat vllm_plugin/manifest.json
```

This generates:
- `<output_dir>/optimized/vllm_plugin/__init__.py` — registers CustomOps
- `<output_dir>/optimized/run_patched_vllm.py` — launcher script
- `<output_dir>/optimized/vllm_plugin/manifest.json` — registration summary

## Step 2: Test Plugin Registration (dry run)

Verify that the plugin loads without errors:

```bash
source <output_dir>/venv/bin/activate
python3 -c "
import sys; sys.path.insert(0, '<output_dir>/optimized')
import vllm_plugin
print('Plugin loaded successfully')
"
```

## Step 3: ⛔ MANDATORY — Benchmark Baseline

Use existing `baseline_serving.json` from Phase 4, or re-run:

```bash
source <output_dir>/venv/bin/activate

# ALL vLLM output to log files — NEVER to stdout
vllm serve $1 --dtype auto --max-model-len 4096 --port 8192 --disable-log-requests &> <output_dir>/vllm_baseline_e2e.log &
VLLM_PID=$!
echo "Baseline PID: $VLLM_PID"
for i in $(seq 1 60); do curl -s http://localhost:8192/health > /dev/null 2>&1 && break; sleep 5; done
curl -s http://localhost:8192/health > /dev/null 2>&1 && echo "✓ Ready" || { echo "✗ Failed"; tail -3 <output_dir>/vllm_baseline_e2e.log; }

vllm bench serve \
  --model $1 --port 8192 \
  --dataset-name random \
  --input-len 1024 --output-len 1024 \
  --num-prompts 100 --max-concurrency 16 \
  --request-rate inf --save-result \
  --result-dir <output_dir>/report --result-filename baseline_serving.json --label baseline \
  &> <output_dir>/report/bench_baseline.log

kill $VLLM_PID 2>/dev/null; wait $VLLM_PID 2>/dev/null

# Show only key metrics
python3 -c "
import json
with open('<output_dir>/report/baseline_serving.json') as f: d=json.load(f)
print('=== Baseline ===')
for k in ['output_throughput','mean_tpot_ms','mean_ttft_ms','completed']:
    print(f'  {k}: {d.get(k,\"N/A\")}')
"
```

## Step 4: ⛔ MANDATORY — Start Patched vLLM and Benchmark

```bash
source <output_dir>/venv/bin/activate

# Start patched vLLM — ALL output to log file
python3 <output_dir>/optimized/run_patched_vllm.py serve \
  --model $1 --dtype auto --max-model-len 4096 \
  --port 8193 --disable-log-requests &> <output_dir>/vllm_patched.log &
PATCHED_PID=$!
echo "Patched PID: $PATCHED_PID (log: <output_dir>/vllm_patched.log)"

# Wait silently
for i in $(seq 1 60); do curl -s http://localhost:8193/health > /dev/null 2>&1 && break; sleep 5; done
curl -s http://localhost:8193/health > /dev/null 2>&1 && echo "✓ Patched server ready" || { echo "✗ Failed"; tail -5 <output_dir>/vllm_patched.log; }

# Verify correct model (compact output)
curl -s http://localhost:8193/v1/models | python3 -c "
import json,sys; d=json.load(sys.stdin)
models=[m['id'] for m in d.get('data',[])]
print(f'Models: {models}')
assert '$1' in models, f'Wrong model!'
"

# Quick correctness test
curl -s http://localhost:8193/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"$1","prompt":"Hello","max_tokens":5}' \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print('✓ OK' if 'choices' in d else f'✗ {d}')"

# Benchmark — output to file
vllm bench serve \
  --model $1 --port 8193 \
  --dataset-name random \
  --input-len 1024 --output-len 1024 \
  --num-prompts 100 --max-concurrency 16 \
  --request-rate inf --save-result \
  --result-dir <output_dir>/report --result-filename optimized_serving.json --label optimized \
  &> <output_dir>/report/bench_optimized.log

kill $PATCHED_PID 2>/dev/null; wait $PATCHED_PID 2>/dev/null

# Show only key metrics
python3 -c "
import json
with open('<output_dir>/report/optimized_serving.json') as f: d=json.load(f)
print('=== Optimized ===')
for k in ['output_throughput','mean_tpot_ms','mean_ttft_ms','completed']:
    print(f'  {k}: {d.get(k,\"N/A\")}')
"
```

**If the patched server fails to start or crashes:**
1. Check `run_patched_vllm.py` output for registration errors
2. Try removing problematic kernels from `vllm_plugin/` and regenerate
3. If ALL patches fail, run benchmark anyway (it measures "no-change" as the honest result)

## Step 5: ⛔ MANDATORY — Validate Results

```bash
source <output_dir>/venv/bin/activate
python3 << 'VALIDATE'
import json, sys, os

report_dir = "<output_dir>/report"
baseline_path = os.path.join(report_dir, "baseline_serving.json")
optimized_path = os.path.join(report_dir, "optimized_serving.json")
errors = []

for path, name, expected_label in [
    (baseline_path, "baseline", "baseline"),
    (optimized_path, "optimized", "optimized"),
]:
    if not os.path.exists(path):
        errors.append(f"MISSING: {name}_serving.json — you must run vllm bench serve")
        continue
    with open(path) as f:
        data = json.load(f)
    label = data.get("label", "")
    if label != expected_label:
        errors.append(f"{name}_serving.json label='{label}', expected '{expected_label}'")
    completed = data.get("completed", 0)
    if completed == 0 and name == "optimized":
        errors.append(f"optimized_serving.json has completed=0 — patched server did not work")
    if name == "optimized" and os.path.exists(baseline_path):
        with open(baseline_path) as f:
            bl = json.load(f)
        if data.get("date") == bl.get("date"):
            errors.append("SUSPICIOUS: same date on baseline and optimized — were these separate runs?")

if errors:
    print("⛔ VALIDATION FAILED:")
    for e in errors:
        print(f"  - {e}")
    print("\nYou must fix the issues above. Phase 7 is NOT complete.")
    sys.exit(1)

with open(baseline_path) as f: baseline = json.load(f)
with open(optimized_path) as f: optimized = json.load(f)

b_otps = baseline.get("output_throughput", 0)
o_otps = optimized.get("output_throughput", 0)
speedup = o_otps / b_otps if b_otps > 0 else 1.0

print("✅ VALIDATION PASSED — Real measurements confirmed")
print(f"  Baseline OTPS:  {b_otps:.2f} tok/s (completed={baseline.get('completed',0)})")
print(f"  Optimized OTPS: {o_otps:.2f} tok/s (completed={optimized.get('completed',0)})")
print(f"  Speedup:        {speedup:.3f}x")

os.makedirs(os.path.join(report_dir, "comparison_outputs"), exist_ok=True)
with open(os.path.join(report_dir, "comparison_outputs", "comparison_results.json"), "w") as f:
    json.dump({
        "validated": True,
        "baseline_otps": b_otps, "optimized_otps": o_otps, "speedup_otps": speedup,
        "baseline_tpot_ms": baseline.get("mean_tpot_ms", 0),
        "optimized_tpot_ms": optimized.get("mean_tpot_ms", 0),
        "baseline_ttft_ms": baseline.get("mean_ttft_ms", 0),
        "optimized_ttft_ms": optimized.get("mean_ttft_ms", 0),
        "concurrency": 16, "input_len": 1024, "output_len": 1024,
    }, f, indent=2)
VALIDATE
```

**If validation fails, fix the issue and re-run from the failing step.**

Update progress.json: phases_completed.append("integrate")


---

# Phase 8: Generate Final Report 

## Goal
Create a comprehensive optimization report.

## Create Report: `<output_dir>/report/optimization_report.md`

**⚠️ CRITICAL**: Include **ACTUAL MEASURED** end-to-end speedup from comparison_results.json.

```markdown
# Model Optimization Report

## Model Information
- **Model**: $1
- **Optimization Date**: [DATE]

## Summary
- **ACTUAL End-to-End Speedup**: X.Xx (measured, NOT estimated)
- **Kernels Optimized**: N
- **Baseline Inference Time**: X.Xs
- **Optimized Inference Time**: X.Xs

## Bottleneck Analysis
| Operator | Original Time (ms) | % of Total | Optimized | Speedup |
|----------|-------------------|------------|-----------|---------|
| ...      | ...               | ...        | ...       | ...     |

## Performance Results (ACTUAL MEASURED)
| Metric | Original | Optimized | Speedup |
|--------|----------|-----------|---------|
| End-to-End Inference Time | X.Xs | X.Xs | **X.Xx** |

## Comparison Outputs (Seed=42)
Outputs generated with fixed random seed for verification.

### Text Models:
| Original | Optimized |
|----------|-----------|
| [text]   | [text]    |

### Image Models:
| Original | Optimized |
|:--------:|:---------:|
| ![Original](comparison_outputs/original_output.png) | ![Optimized](comparison_outputs/optimized_output.png) |

## Files Generated
- model/ - Downloaded model
- demo/demo.py - Working demo
- profile/bottlenecks.json - Profiling results
- problems/ - Problem files + optimized kernels
- optimized/integrate.py - Integration script
- report/optimization_report.md - This report

## Recommendations
1. ...
```

## Steps
1. Gather all results from previous phases
2. Generate the comprehensive report
3. Update progress.json: phase="complete", phases_completed.append("report")



---

# EXECUTION INSTRUCTIONS
Execute phases: 0 → 1 → 4 → 5 → 6 → 7 → 8 (Phases 2-3 handled by vLLM).
**ALL vLLM output to log files. Run validate_pipeline.py after Phase 6 and 7.**
Begin with Phase 0.
