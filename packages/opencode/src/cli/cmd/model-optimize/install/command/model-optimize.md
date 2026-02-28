---
description: "End-to-end vLLM model optimization pipeline. Usage: /model-optimize <model_name> [output_dir]"
agent: model-opt
---

# End-to-End vLLM Model Optimization Pipeline

## Target
- **HuggingFace Model**: `$1` (referred to as `$HF_MODEL` below)
- **Output Directory**: `$2` if provided, else `/tmp/model_opt_<short_name>` (referred to as `$OUTPUT_DIR` below; **MUST be outside the working directory**)

## First Steps
1. Initialize `HF_MODEL` and `OUTPUT_DIR`:
   ```bash
   HF_MODEL="$1"
   SHORT_NAME=$(basename "$HF_MODEL" | tr '/:' '__')
   OUTPUT_DIR="${2:-/tmp/model_opt_${SHORT_NAME}}"
   ```
2. Create directory structure: `mkdir -p $OUTPUT_DIR/{profile/traces,problems,optimized,report,scripts}`
3. Add `.gitignore` (exclude `venv/`, `*.safetensors`, `__pycache__/`, etc.)

## Execution Context (run once after Phase 0, before Phase 1)
Phase 0 creates `$OUTPUT_DIR/env_info.json`. Read it **once** after Phase 0 completes and keep these variables for all subsequent phases.

```bash
ENV_TYPE=$(python3 -c "import json; print(json.load(open('$OUTPUT_DIR/env_info.json')).get('env_type','venv'))" 2>/dev/null || echo "venv")
CONTAINER_NAME=$(python3 -c "import json; print(json.load(open('$OUTPUT_DIR/env_info.json')).get('container','vllm_model_opt'))" 2>/dev/null || echo "vllm_model_opt")
BEST_GPU=$(python3 -c "import json; print(json.load(open('$OUTPUT_DIR/env_info.json')).get('best_gpu',0))" 2>/dev/null || echo 0)

if [ "$ENV_TYPE" = "docker" ]; then
  echo "Running in Docker mode: $CONTAINER_NAME (HIP_VISIBLE_DEVICES=$BEST_GPU)"
  RUN_PREFIX="docker exec -e HIP_VISIBLE_DEVICES=$BEST_GPU $CONTAINER_NAME bash -lc"
else
  echo "Running in venv mode: $OUTPUT_DIR/venv"
  source "$OUTPUT_DIR/venv/bin/activate"
  RUN_PREFIX=""
fi
```

In Docker mode, prefix GPU-dependent commands with `$RUN_PREFIX "<command>"`.
In venv mode, `RUN_PREFIX` is empty — commands run directly on the host.

## Critical Rules
- **ALL vLLM output to log files** (`&> logfile`) — NEVER dump vLLM logs into bash output
- **ALL decisions MUST be data-driven** — no estimated speedups
- **Optimized kernels MUST use `@triton.jit`** — torch rewrites are FORBIDDEN
- **Serving benchmarks MUST use `vllm bench serve --save-result`**
- **Use one execution context**: apply `RUN_PREFIX` in Docker mode; in venv mode commands run directly

## Hard-Stop Rules (Instruction Strictness)
- If any command in a phase fails, **STOP** and fix root cause before continuing.
- Never skip a phase gate: if required artifacts are missing, **do not proceed**.
- Never claim success without file-backed evidence (`*.json`, `*.csv`, logs, or generated code).
- If profiling shape attribution is poor or missing, re-collect traces with required flags; do not continue with guessed shapes.
- If integration benchmark is missing or invalid, report failure explicitly; do not estimate speedup.

## Validation
After Phase 6 and Phase 7:
```bash
python3 $OUTPUT_DIR/scripts/validate_pipeline.py --project-dir $OUTPUT_DIR --phase all
```

## Phase Exit Gates (Mandatory)
- **Phase 0**: `$OUTPUT_DIR/env_info.json` exists and contains `env_type`.
- **Phase 1**: `$OUTPUT_DIR/model_config.json` exists and health/inference checks pass.
- **Phase 4**: `$OUTPUT_DIR/profile/bottlenecks.json`, `kernel_shape_analysis.json`, and at least one trace file exist.
- **Phase 5**: at least one `problem_*.py` exists under `$OUTPUT_DIR/problems/`.
- **Phase 6**: each finalized optimized file has a passing tracker (`*_best.json`) and speedup evidence.
- **Phase 7**: both `baseline_serving.json` and `optimized_serving.json` exist, labels are correct, and validation passes.
- **Phase 8**: `$OUTPUT_DIR/report/optimization_report.md` exists and references measured results.

---

# Phase 0: Environment Setup

## Goal
Find a compatible Docker image (`rocm/vllm-dev`) on Docker Hub and create a container. Fall back to a Python venv if no image is available.

## Steps

### 1. Detect host platform

```bash
ROCM_VERSION=$(cat /opt/rocm/.info/version 2>/dev/null | head -1 | cut -d'-' -f1 || echo "unknown")

GPU_ARCH=$(rocminfo 2>/dev/null | grep -oP 'gfx\w+' | head -1 || true)
if [ -z "$GPU_ARCH" ]; then
  # kfd sysfs fallback (works without /dev/kfd permissions)
  # gfx_target_version is packed decimal: major*10000 + minor*100 + stepping
  GPU_ARCH=$(cat /sys/class/kfd/kfd/topology/nodes/*/properties 2>/dev/null \
    | grep gfx_target_version \
    | awk '$2 > 0 {v=$2; maj=int(v/10000); min=int((v%10000)/100); step=v%100; printf "gfx%d%x%x\n", maj, min, step}' \
    | head -1 || echo "unknown")
fi

DOCKER_OK=$(docker info >/dev/null 2>&1 && echo "yes" || echo "no")
echo "ROCm: $ROCM_VERSION  GPU: $GPU_ARCH  Docker: $DOCKER_OK"
```

### 2. Search Docker Hub for a compatible image

```bash
# CDNA (gfx9xx) → mainline nightly tags; RDNA (gfx1xxx) → navi tags
if [[ "$GPU_ARCH" == gfx9* ]]; then
  TAG_PATTERN="nightly_main"
elif [[ "$GPU_ARCH" == gfx1* ]]; then
  TAG_PATTERN="navi"
else
  TAG_PATTERN=""
fi

ROCM_MAJOR_MINOR=$(echo "$ROCM_VERSION" | grep -oP '^\d+\.\d+')
echo "Searching rocm/vllm-dev for pattern=$TAG_PATTERN, prefer ROCm $ROCM_MAJOR_MINOR"

IMAGE_TAG=""
for page in 1 2 3 4 5; do
  MATCHES=$(curl -sL "https://hub.docker.com/v2/repositories/rocm/vllm-dev/tags?page_size=100&page=$page&ordering=last_updated" \
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
    BEST=$(echo "$MATCHES" | grep "rocm${ROCM_MAJOR_MINOR}" | head -1)
    [ -z "$BEST" ] && BEST=$(echo "$MATCHES" | head -1)
    IMAGE_TAG="$BEST"
    break
  fi
done

if [ -z "$IMAGE_TAG" ]; then
  echo "No matching Docker image found — will fall back to venv"
fi
```

### 3. Create container (if image found)

```bash
CONTAINER_NAME="vllm_model_opt"

if [ -n "$IMAGE_TAG" ]; then
  IMAGE="rocm/vllm-dev:$IMAGE_TAG"
  echo "Using image: $IMAGE"
  docker pull "$IMAGE" 2>/dev/null

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
      -v $OUTPUT_DIR:/workspace/output \
      -p 8192:8192 -p 8193:8193 \
      "$IMAGE" sleep infinity
    echo "Created container: $CONTAINER_NAME"
  fi

  # Verify and select best GPU
  docker exec "$CONTAINER_NAME" python3 -c "
import torch, vllm, json
print(f'PyTorch {torch.__version__}, vLLM {vllm.__version__}')
best, best_free = 0, 0
for i in range(torch.cuda.device_count()):
    free, total = torch.cuda.mem_get_info(i)
    print(f'  cuda:{i} — {torch.cuda.get_device_name(i)}, free={free/1e9:.1f}GB/{total/1e9:.1f}GB')
    if free > best_free: best, best_free = i, free
info = {'env_type': 'docker', 'container': '$CONTAINER_NAME', 'image': '$IMAGE',
        'pytorch': torch.__version__, 'vllm': vllm.__version__,
        'gpu_count': torch.cuda.device_count(), 'best_gpu': best}
with open('/workspace/output/env_info.json', 'w') as f:
    json.dump(info, f, indent=2)
print(f'Best GPU: cuda:{best}')
print(json.dumps(info, indent=2))
"
fi
```

### 4. Fallback: venv setup (if no Docker image)

```bash
if [ -z "$IMAGE_TAG" ]; then
  cd $OUTPUT_DIR

  if [ ! -d "venv" ]; then
    python3 -m venv venv --system-site-packages
  fi
  source venv/bin/activate

  python3 -c "import vllm" 2>/dev/null || pip install vllm --extra-index-url https://wheels.vllm.ai/rocm/
  python3 -c "import transformers" 2>/dev/null || pip install transformers
  python3 -c "import accelerate" 2>/dev/null || pip install accelerate

  python3 -c "
import json, torch, vllm
best, best_free = 0, 0
for i in range(torch.cuda.device_count()):
    free, _ = torch.cuda.mem_get_info(i)
    if free > best_free: best, best_free = i, free
info = {'env_type': 'venv', 'pytorch': torch.__version__,
        'vllm': vllm.__version__, 'gpu_count': torch.cuda.device_count(),
        'best_gpu': best}
with open('$OUTPUT_DIR/env_info.json', 'w') as f:
    json.dump(info, f, indent=2)
print(f'PyTorch {torch.__version__}, vLLM {vllm.__version__}, GPU: {torch.cuda.get_device_name(best)}, best_gpu={best}')
"
fi
```

### 5. Copy helper scripts

```bash
mkdir -p $OUTPUT_DIR/scripts
cp ~/.config/opencode/scripts/*.py $OUTPUT_DIR/scripts/ 2>/dev/null
ls $OUTPUT_DIR/scripts/
```

### 6. Update progress.json
Update progress.json: `phase="env"`, `phases_completed.append("env")`

---

# Phase 1: Model Serving with vLLM

## Goal
Verify `vllm serve` works with the target model. vLLM handles download automatically.

> In vLLM mode, Phases 1-3 (download, demo, compatibility) are handled in one step.

## Steps

### 1. Test vLLM serve
```bash
vllm serve $HF_MODEL \
  --dtype auto --max-model-len 2048 --port 8192 \
  --disable-log-requests &> $OUTPUT_DIR/vllm_serve.log &
VLLM_PID=$!

for i in $(seq 1 60); do
  curl -s http://localhost:8192/health > /dev/null 2>&1 && break; sleep 5
done
curl -s http://localhost:8192/health > /dev/null 2>&1 && echo "Server ready" || echo "FAILED — check $OUTPUT_DIR/vllm_serve.log"

curl -s http://localhost:8192/v1/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"$HF_MODEL\", \"prompt\": \"Hello, I am\", \"max_tokens\": 20}" \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print('Inference OK' if 'choices' in d else f'Error: {d}')"

kill $VLLM_PID 2>/dev/null; wait $VLLM_PID 2>/dev/null
```

### 2. Record model config
```bash
python3 -c "
from transformers import AutoConfig
import json
c = AutoConfig.from_pretrained('$HF_MODEL', trust_remote_code=True)
info = {k: getattr(c, k, None) for k in [
    'model_type','num_hidden_layers','hidden_size',
    'num_attention_heads','num_key_value_heads','intermediate_size','vocab_size']}
with open('$OUTPUT_DIR/model_config.json', 'w') as f:
    json.dump(info, f, indent=2)
print(json.dumps(info, indent=2))
"
```

### 3. Update progress.json
`phases_completed += ["download", "demo", "compatibility"]`

---

# Phase 4: Performance Profiling

## Goal
Baseline benchmark + GPU kernel trace with shape data for bottleneck analysis.

## Step 1: Baseline Throughput Benchmark

```bash
vllm serve $HF_MODEL \
  --dtype auto --max-model-len 4096 --port 8192 \
  --disable-log-requests &> $OUTPUT_DIR/vllm_baseline.log &
VLLM_PID=$!

for i in $(seq 1 60); do
  curl -s http://localhost:8192/health > /dev/null 2>&1 && break; sleep 5
done
curl -s http://localhost:8192/health > /dev/null 2>&1 && echo "Server ready" || { echo "FAILED"; tail -5 $OUTPUT_DIR/vllm_baseline.log; }

vllm bench serve \
  --model $HF_MODEL --port 8192 \
  --dataset-name random --input-len 1024 --output-len 1024 \
  --num-prompts 100 --max-concurrency 16 \
  --request-rate inf --save-result \
  --result-dir $OUTPUT_DIR/profile --result-filename baseline_benchmark.json \
  --label baseline &> $OUTPUT_DIR/profile/bench_baseline.log

kill $VLLM_PID 2>/dev/null; wait $VLLM_PID 2>/dev/null

python3 -c "
import json
with open('$OUTPUT_DIR/profile/baseline_benchmark.json') as f: d = json.load(f)
print('=== Baseline ===')
for k in ['output_throughput','request_throughput','mean_tpot_ms','mean_ttft_ms','mean_itl_ms','completed']:
    print(f'  {k}: {d.get(k,\"N/A\")}')
"
```

## Step 2: Collect Kernel Trace

Two flags are **mandatory** for shape analysis:
- `--enforce-eager` — disables CUDA Graphs so GPU kernels retain `External id` linkage to CPU ops
- `--profiler-config` with `record_shapes=True` — records tensor dimensions on every CPU op

Without both, `analyze_kernel_shapes.py` produces only `(unattributed)` shapes.
Do not continue to Phase 5 unless the trace contains shape metadata and `kernel_shape_analysis.json` reports meaningful attributed shapes.

```bash
mkdir -p $OUTPUT_DIR/profile/traces
TRACE_DIR=$(realpath $OUTPUT_DIR/profile/traces)

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

VLLM_TORCH_PROFILER_DIR="$TRACE_DIR" \
vllm serve $HF_MODEL \
  --dtype auto --max-model-len 4096 --port 8193 \
  --disable-log-requests --enforce-eager \
  --profiler-config "$PROFILER_CFG" &> $OUTPUT_DIR/vllm_trace.log &
VLLM_PID=$!

for i in $(seq 1 60); do
  curl -s http://localhost:8193/health > /dev/null 2>&1 && break; sleep 5
done
curl -s http://localhost:8193/health > /dev/null 2>&1 && echo "Trace server ready" || { echo "FAILED"; tail -5 $OUTPUT_DIR/vllm_trace.log; }

curl -s -X POST http://localhost:8193/start_profile && echo "Profiler started"

vllm bench serve \
  --model $HF_MODEL --port 8193 \
  --dataset-name random --input-len 1024 --output-len 1024 \
  --num-prompts 30 --max-concurrency 16 \
  --request-rate inf --save-result \
  --result-dir $OUTPUT_DIR/profile --result-filename trace_benchmark.json \
  --label trace &> $OUTPUT_DIR/profile/bench_trace.log

curl -s -X POST http://localhost:8193/stop_profile && echo "Profiler stopped"
sleep 15
kill $VLLM_PID 2>/dev/null; wait $VLLM_PID 2>/dev/null

ls -lh $OUTPUT_DIR/profile/traces/ 2>/dev/null | head -5
```

## Step 3: Extract Kernel Bottlenecks

```bash
cd $OUTPUT_DIR/profile
cp $OUTPUT_DIR/scripts/vllm_trace_extractor.py .

TRACE_FILE=$(ls -t traces/*.json traces/*.json.gz 2>/dev/null | head -1)
python3 vllm_trace_extractor.py -i "$TRACE_FILE" \
  --full-csv kernel_full.csv --unique-csv kernel_unique.csv
```

## Step 4: Generate bottlenecks.json

```bash
cd $OUTPUT_DIR/profile
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
    optimizable, reason = True, ''
    if 'Cijk_' in name or 'gemm' in name.lower() or 'hipblas' in name.lower():
        reason, optimizable = 'GEMM/rocBLAS', False
    elif 'attn' in name.lower() or 'flash' in name.lower() or 'mha' in name.lower():
        reason = 'Attention'
    elif 'norm' in name.lower() or 'rms' in name.lower():
        reason = 'Normalization'
    elif 'elementwise' in name.lower() or 'vectorized' in name.lower():
        reason = 'Elementwise'
    elif 'silu' in name.lower() or 'gelu' in name.lower():
        reason = 'Activation'
    elif 'copy' in name.lower() or 'Cat' in name:
        reason, optimizable = 'Memory op', False
    bottlenecks.append({**k, 'cuda_time_percent': pct, 'optimizable': optimizable, 'reason': reason})

print(f'Total GPU time: {total/1000:.2f}ms, Top 10:')
for i, b in enumerate(bottlenecks[:10], 1):
    print(f'  {i}. {b[\"name\"][:45]:45s} {b[\"cuda_time_percent\"]:5.1f}%')

with open('bottlenecks.json', 'w') as f:
    json.dump(bottlenecks, f, indent=2)
print(f'Saved bottlenecks.json ({len(bottlenecks)} kernels)')
"
```

## Step 5: Per-Shape Kernel Time Analysis

Correlates GPU kernel durations with CPU-side operator shapes from the trace.

```bash
cd $OUTPUT_DIR/profile
cp $OUTPUT_DIR/scripts/analyze_kernel_shapes.py .

TRACE_FILE=$(ls -t traces/*.json traces/*.json.gz 2>/dev/null | head -1)
python3 analyze_kernel_shapes.py -i "$TRACE_FILE" -o .
```

Produces `kernel_shape_analysis.json` and `kernel_shape_analysis.csv`.

Example output:
```
  GEMM — 94.1% of total (15488.61ms, 10 shapes, 100% attributed)
  ──────────────────────────────────────────────────────────────────────────────────────
    Shape                                               %Total  %InCat  Time(ms)  Count  Avg(us)
    [4,4096]x[4096,24576]                                30.5%   32.4%  5023.08   9072    553.7
    [2,4096]x[4096,24576]                                15.4%   16.4%  2533.19   4644    545.5
    ...

  Attention — 2.4% of total (390.56ms, 4 shapes, 100% attributed)
  ──────────────────────────────────────────────────────────────────────────────────────
    Shape                                               %Total  %InCat  Time(ms)  Count  Avg(us)
    [4,32,128]x[4,8,128]x[4,8,128]x[4,32,128]             1.5%   64.4%   251.52  27216      9.2
    ...
```

Use this data in Phase 5 to pick exact shapes for problem files — the shapes must be real traced shapes.

## Step 6: Save model shapes

```bash
python3 -c "
import json
from transformers import AutoConfig
c = AutoConfig.from_pretrained('$HF_MODEL', trust_remote_code=True)
shapes = {
    'hidden_size': getattr(c, 'hidden_size', None),
    'intermediate_size': getattr(c, 'intermediate_size', None),
    'num_attention_heads': getattr(c, 'num_attention_heads', None),
    'num_key_value_heads': getattr(c, 'num_key_value_heads', None),
    'head_dim': getattr(c, 'hidden_size', 0) // max(getattr(c, 'num_attention_heads', 1), 1),
    'num_hidden_layers': getattr(c, 'num_hidden_layers', None),
    'vocab_size': getattr(c, 'vocab_size', None),
}
with open('$OUTPUT_DIR/profile/model_shapes.json', 'w') as f:
    json.dump(shapes, f, indent=2)
print(json.dumps(shapes, indent=2))
"
```

Update progress.json: `phases_completed.append("profile")`

---

# Phase 5: Generate Problem Files

## Goal
Convert bottleneck operators into Problem files. Analyze fusion opportunities BEFORE creating individual files.

## Step 0: Review Per-Shape Kernel Analysis

```bash
python3 -c "
import json
with open('$OUTPUT_DIR/profile/kernel_shape_analysis.json') as f:
    data = json.load(f)
print(f'Total GPU time: {data[\"total_gpu_time_ms\"]:.2f}ms\n')
for cat in data['categories'][:8]:
    print(f'{cat[\"category\"]:12s} {cat[\"pct\"]:5.1f}%  ({cat[\"total_us\"]/1000:.2f}ms, {cat[\"num_shapes\"]} shapes)')
    for s in cat['shapes'][:5]:
        print(f'  {s[\"shape\"]:50s} {s[\"pct_of_total\"]:5.1f}% total, {s[\"count\"]:4d} calls, avg {s[\"avg_us\"]:.1f}us')
"
```

## Step 1: Operator Fusion Analysis

```bash
cp $OUTPUT_DIR/scripts/analyze_fusion.py $OUTPUT_DIR/profile/
cd $OUTPUT_DIR/profile
python3 analyze_fusion.py
cat fusion_opportunities.json
```

### Common Fusion Opportunities

| Pattern | Operators | Fused Name | Speedup |
|---------|-----------|------------|---------|
| **ResidualNorm** | add + rmsnorm | fused_residual_norm | 1.2-1.5x |
| **SwiGLU/GeGLU** | silu/gelu + mul | fused_swiglu | 1.3-1.8x |
| **RotaryEmbed** | rope_cos + rope_sin + cat | fused_rope | 1.2-1.5x |
| **QKV Projection** | 3x linear (q,k,v) | fused_qkv_proj | 1.2-1.4x |
| **MLP Block** | linear + activation + linear | fused_mlp | 1.3-2.0x |

## Step 2: Create FUSED Problem Files (Priority)

Create fused kernels BEFORE individual kernels. Use shapes from `kernel_shape_analysis.json` — focus on the shapes with the highest `pct_of_total`.

### Example: Fused Residual + RMSNorm
```python
import torch, torch.nn as nn

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

# Use ACTUAL shapes from kernel_shape_analysis.json
batch_size, seq_len, hidden_size = 1, 64, 4096

def get_inputs():
    return [torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float16, device='cuda'),
            torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float16, device='cuda')]
def get_init_inputs():
    return [hidden_size]
```

## Step 3: Create Individual Problem Files (Lower Priority)

Only for operators that cannot be fused, take >5% time, and are not already optimized by vendor libs (rocBLAS GEMM).

## Step 4: Generate Optimization Manifest

Create `$OUTPUT_DIR/problems/optimization_manifest.json`:
```json
{
  "model": "$HF_MODEL",
  "optimizations": [
    {"name": "fused_residual_rmsnorm", "file": "problem_fused_residual_rmsnorm.py",
     "type": "fused", "priority": "HIGH", "enabled": true},
    {"name": "linear_gemm", "file": "problem_linear.py",
     "type": "individual", "priority": "LOW", "enabled": false,
     "notes": "rocBLAS usually optimal"}
  ]
}
```

Update progress.json: `phases_completed.append("problems")`

---

# Phase 6: Kernel Optimization

## Goal
Write optimized Triton kernels for each problem file and verify speedup. Optimize DIRECTLY in this session.

## Scripts
- `$OUTPUT_DIR/scripts/kernel_test_runner.py` — test accuracy + benchmark
- `$OUTPUT_DIR/scripts/kernel_finalize.py` — save best result

## For EACH `problem_*.py` in `$OUTPUT_DIR/problems/`:

### 1. Read the source to understand the operator
### 2. Check GPU architecture
```bash
python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name()}, Arch: {torch.cuda.get_device_capability()}')"
```

### 3. Write optimized Triton kernel
Create `problem_XXX_opt.py` with `class ModelNew(nn.Module)` using `@triton.jit` kernels and `@triton.autotune` with 10-20 configs.

### 4. Test
```bash
python3 $OUTPUT_DIR/scripts/kernel_test_runner.py \
  --src $OUTPUT_DIR/problems/problem_XXX.py \
  --target $OUTPUT_DIR/problems/problem_XXX_opt.py
```

### 5. Iterate until accuracy passes and speedup meets goal
### 6. Finalize: `python3 $OUTPUT_DIR/scripts/kernel_finalize.py --target $OUTPUT_DIR/problems/problem_XXX_opt.py`

## Priority

| Priority | Kernel Type | Goal | Reason |
|----------|-------------|------|--------|
| **HIGH** | Fused Residual+RMSNorm | 1.5x | Memory traffic reduction |
| **HIGH** | Fused SwiGLU | 1.5x | Activation fusion |
| **HIGH** | Fused RoPE | 1.5x | Custom optimization |
| MEDIUM | Individual norms | 1.3x | If not covered by fused version |
| LOW | Linear/GEMM | 1.1x | rocBLAS usually optimal |
| **SKIP** | Simple add/copy | — | Overhead > benefit |

## Skip Criteria
- Part of a fused kernel already optimized
- rocBLAS/vendor lib is already near-optimal
- After 3 attempts, speedup is < 1.0x

## Triton Guide

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8, num_stages=2),
    ],
    key=['N'],
)
```

- **Memory-bound**: Optimize access patterns, vectorization
- **Compute-bound**: Larger tiles, more arithmetic per memory access
- **Fused kernels**: Combine ops to reduce memory traffic
- **FP32 accumulation**: Use `tl.float32` for acc, cast output at end

## After All Kernels Done

Copy successful optimizations (speedup > 1.0x) to `$OUTPUT_DIR/optimized/`.

Update progress.json: `phases_completed.append("optimize")`

---

# Phase 7: Integration & End-to-End Testing

## Goal
Apply optimized kernels to vLLM via CustomOp and measure ACTUAL serving throughput.

## This phase is NOT complete until:
1. A patched vLLM server has been started and served requests
2. `vllm bench serve` has been run against the patched server
3. `optimized_serving.json` has `"label": "optimized"`
4. The validation script passes

**FORBIDDEN**: Estimating speedup, copying baseline numbers, reporting "estimated" results.

## Integration: vLLM CustomOp.register_oot()

Official extension mechanism (not monkey-patching): each optimized kernel is wrapped as a `CustomOp` subclass. `register_oot()` replaces the default op; failures fall back to default.

## Step 1: Generate vLLM Plugin

```bash
cd $OUTPUT_DIR/optimized
cp $OUTPUT_DIR/problems/*_opt.py . 2>/dev/null
python3 $OUTPUT_DIR/scripts/generate_vllm_plugin.py --kernel-dir $OUTPUT_DIR/optimized
ls -la vllm_plugin/
```

## Step 2: Test Plugin Registration
```bash
python3 -c "
import sys; sys.path.insert(0, '$OUTPUT_DIR/optimized')
import vllm_plugin
print('Plugin loaded successfully')
"
```

## Step 3: Benchmark Baseline

Phase 4 produced `baseline_benchmark.json` in `profile/` (different workload). This step runs a fresh baseline in `report/` with the same parameters as the optimized benchmark for a fair comparison:

```bash
vllm serve $HF_MODEL --dtype auto --max-model-len 4096 --port 8192 \
  --disable-log-requests &> $OUTPUT_DIR/vllm_baseline_e2e.log &
VLLM_PID=$!
for i in $(seq 1 60); do curl -s http://localhost:8192/health > /dev/null 2>&1 && break; sleep 5; done

vllm bench serve \
  --model $HF_MODEL --port 8192 \
  --dataset-name random --input-len 1024 --output-len 1024 \
  --num-prompts 100 --max-concurrency 16 \
  --request-rate inf --save-result \
  --result-dir $OUTPUT_DIR/report --result-filename baseline_serving.json --label baseline \
  &> $OUTPUT_DIR/report/bench_baseline.log

kill $VLLM_PID 2>/dev/null; wait $VLLM_PID 2>/dev/null
```

## Step 4: Benchmark Patched vLLM

```bash
python3 $OUTPUT_DIR/optimized/run_patched_vllm.py serve \
  --model $HF_MODEL --dtype auto --max-model-len 4096 \
  --port 8193 --disable-log-requests &> $OUTPUT_DIR/vllm_patched.log &
PATCHED_PID=$!
for i in $(seq 1 60); do curl -s http://localhost:8193/health > /dev/null 2>&1 && break; sleep 5; done

curl -s http://localhost:8193/v1/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"$HF_MODEL\",\"prompt\":\"Hello\",\"max_tokens\":5}" \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print('OK' if 'choices' in d else f'Error: {d}')"

vllm bench serve \
  --model $HF_MODEL --port 8193 \
  --dataset-name random --input-len 1024 --output-len 1024 \
  --num-prompts 100 --max-concurrency 16 \
  --request-rate inf --save-result \
  --result-dir $OUTPUT_DIR/report --result-filename optimized_serving.json --label optimized \
  &> $OUTPUT_DIR/report/bench_optimized.log

kill $PATCHED_PID 2>/dev/null; wait $PATCHED_PID 2>/dev/null
```

If the patched server fails: check registration errors, remove problematic kernels, regenerate.

## Step 5: Validate Results

```bash
OUTPUT_DIR="$OUTPUT_DIR" python3 << 'VALIDATE'
import json, sys, os

output_dir = os.environ["OUTPUT_DIR"]
report_dir = os.path.join(output_dir, "report")
baseline_path = os.path.join(report_dir, "baseline_serving.json")
optimized_path = os.path.join(report_dir, "optimized_serving.json")
errors = []

for path, name, expected_label in [
    (baseline_path, "baseline", "baseline"),
    (optimized_path, "optimized", "optimized"),
]:
    if not os.path.exists(path):
        errors.append(f"MISSING: {name}_serving.json"); continue
    with open(path) as f: data = json.load(f)
    if data.get("label", "") != expected_label:
        errors.append(f"{name}_serving.json label mismatch")
    if data.get("completed", 0) == 0 and name == "optimized":
        errors.append("optimized_serving.json has completed=0")

if errors:
    print("VALIDATION FAILED:"); [print(f"  - {e}") for e in errors]; sys.exit(1)

with open(baseline_path) as f: baseline = json.load(f)
with open(optimized_path) as f: optimized = json.load(f)
b_otps = baseline.get("output_throughput", 0)
o_otps = optimized.get("output_throughput", 0)
speedup = o_otps / b_otps if b_otps > 0 else 1.0

print(f"VALIDATION PASSED")
print(f"  Baseline:  {b_otps:.2f} tok/s")
print(f"  Optimized: {o_otps:.2f} tok/s")
print(f"  Speedup:   {speedup:.3f}x")

os.makedirs(os.path.join(report_dir, "comparison_outputs"), exist_ok=True)
with open(os.path.join(report_dir, "comparison_outputs", "comparison_results.json"), "w") as f:
    json.dump({"validated": True, "baseline_otps": b_otps, "optimized_otps": o_otps,
               "speedup_otps": speedup,
               "baseline_tpot_ms": baseline.get("mean_tpot_ms", 0),
               "optimized_tpot_ms": optimized.get("mean_tpot_ms", 0),
               "baseline_ttft_ms": baseline.get("mean_ttft_ms", 0),
               "optimized_ttft_ms": optimized.get("mean_ttft_ms", 0),
               "concurrency": 16, "input_len": 1024, "output_len": 1024}, f, indent=2)
VALIDATE
```

Update progress.json: `phases_completed.append("integrate")`

---

# Phase 8: Generate Final Report

## Goal
Create `$OUTPUT_DIR/report/optimization_report.md` with ACTUAL MEASURED data.

Include:
- Model information and optimization date
- Bottleneck analysis table (from `bottlenecks.json`)
- Per-shape kernel time breakdown (from `kernel_shape_analysis.json`)
- Performance results: baseline vs optimized (from `comparison_results.json`)
- Kernels optimized and individual speedups
- Files generated listing
- Recommendations for further optimization

Update progress.json: `phase="complete"`, `phases_completed.append("report")`

---

# EXECUTION INSTRUCTIONS
Execute phases: **0 → 1 → 4 → 5 → 6 → 7 → 8** (Phases 2-3 handled by vLLM).
ALL vLLM output to log files. Run `validate_pipeline.py` after Phase 6 and 7.
Begin with Phase 0.
