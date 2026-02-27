# Skill: Profiling vLLM Models on AMD GPUs

## Overview

End-to-end workflow to profile any HuggingFace model served by vLLM on AMD GPUs (MI300/MI355 series). Covers free GPU selection, server launch with the PyTorch profiler, controlled request generation, large trace JSON parsing, and categorized kernel-level bottleneck analysis.

---

## Phase 1: Find a Free GPU

AMD multi-GPU nodes often run mixed workloads. Check **both** utilization and VRAM before choosing.

```bash
# Overview: utilization + VRAM
rocm-smi --showuse --showmeminfo vram

# Concise table view
rocm-smi
```

**Decision rule**: Pick a GPU with near-zero VRAM used (only the small base allocation). Avoid GPUs with significant VRAM usage even if utilization shows 0% — they may have reserved memory.

**Verify programmatically** before committing:

```python
import torch, os
GPU_ID = <candidate_id>
os.environ['HIP_VISIBLE_DEVICES'] = str(GPU_ID)
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_ID)
torch.cuda.init()
free, total = torch.cuda.mem_get_info(0)
print(f"Free: {free/1024**3:.2f} GiB / Total: {total/1024**3:.2f} GiB")
```

> **Gotcha**: Do NOT set `ROCR_VISIBLE_DEVICES` in the parent Python process — it blocks CUDA init entirely. Only pass `CUDA_VISIBLE_DEVICES` and `HIP_VISIBLE_DEVICES` to the server subprocess environment.

---

## Phase 2: Launch vLLM Server with Torch Profiler

### Key Configuration

| Parameter | Value | Why |
|-----------|-------|-----|
| `--enforce-eager` | `True` | Disables CUDA graph capture for cleaner kernel-level traces |
| `--profiler-config.profiler` | `"torch"` | Enables the PyTorch profiler backend |
| `--profiler-config.torch_profiler_dir` | absolute path | Directory where the trace JSON is written |
| `--profiler-config.ignore_frontend` | `True` | Reduces overhead; profiles only the engine core |
| `--profiler-config.torch_profiler_use_gzip` | `False` | Faster write; gzip on multi-GB traces is very slow |
| `--profiler-config.torch_profiler_with_stack` | `True` | Enables Python call stack in the trace |
| `--profiler-config.torch_profiler_with_flops` | `True` | Enables FLOP counting for GEMM ops |
| `--profiler-config.torch_profiler_record_shapes` | `True` | Records tensor shapes per operator |
| `--profiler-config.torch_profiler_dump_cuda_time_total` | `True` | Dumps a text summary of CUDA/HIP time totals |

### Launch Command

```bash
CUDA_VISIBLE_DEVICES=$GPU_ID HIP_VISIBLE_DEVICES=$GPU_ID \
nohup python3 -m vllm.entrypoints.openai.api_server \
  --model $MODEL \
  --port $PORT \
  --max-model-len $MAX_MODEL_LEN \
  --gpu-memory-utilization 0.90 \
  --enforce-eager \
  --profiler-config '{
    "profiler":"torch",
    "torch_profiler_dir":"'$PROFILER_DIR'",
    "torch_profiler_with_stack":true,
    "torch_profiler_with_flops":true,
    "torch_profiler_record_shapes":true,
    "torch_profiler_use_gzip":false,
    "torch_profiler_dump_cuda_time_total":true,
    "ignore_frontend":true
  }' \
  > $PROFILER_DIR/server.log 2>&1 &

echo $! > $PROFILER_DIR/server.pid
```

### Wait for Server Ready

```bash
for i in $(seq 1 100); do
  STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:$PORT/health 2>/dev/null || echo "down")
  if [ "$STATUS" = "200" ]; then
    echo "Server ready after ~$((i*3))s"; break
  fi
  if ! kill -0 $(cat $PROFILER_DIR/server.pid) 2>/dev/null; then
    echo "Server died! Check server.log"; break
  fi
  sleep 3
done
```

> **Gotcha**: If you see `ValueError: Free memory on device cuda:0 ... is less than desired GPU memory utilization`, the GPU has hidden allocations. Verify with `torch.cuda.mem_get_info()` and try a different GPU.

---

## Phase 3: Generate Controlled Requests

### Random Prompt Generator

Use common single-token English words so that word count closely matches token count for most tokenizers:

```python
import random

def generate_random_text(approx_tokens=1024):
    vocab = [
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
        "cat", "bird", "fish", "tree", "house", "car", "road", "river",
        "mountain", "sky", "cloud", "rain", "sun", "moon", "star", "light",
        "dark", "red", "blue", "green", "yellow", "white", "black", "big",
        "small", "fast", "slow", "hot", "cold", "new", "old", "good",
        "bad", "happy", "sad", "long", "short", "high", "low", "open",
        "close", "run", "walk", "talk", "sing", "play", "work", "read",
        "write", "think", "know", "see", "hear", "feel", "want", "need",
        "give", "take", "make", "come", "go", "say", "tell", "ask",
        "help", "call", "try", "use", "find", "get", "put", "set",
        "move", "turn", "show", "look", "live", "die", "grow", "cut",
        "eat", "drink", "sleep", "wake", "sit", "stand", "fall", "rise",
        "begin", "end", "start", "stop", "wait", "hold", "keep", "let",
        "bring", "buy", "sell", "pay", "send", "build", "break", "change",
    ]
    return " ".join(random.choice(vocab) for _ in range(approx_tokens))
```

**Always verify** the actual token count with the model's tokenizer:

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
text = generate_random_text(1024)
print(len(tokenizer.encode(text)))
```

### Profiler Lifecycle + Request Sending

```python
import requests, time

BASE_URL = f"http://localhost:{PORT}"

# 1) Start profiler BEFORE sending requests
requests.post(f"{BASE_URL}/start_profile", timeout=30)
time.sleep(2)  # Let profiler stabilize

# 2) Send requests sequentially for a clean trace
for i, prompt in enumerate(prompts):
    r = requests.post(f"{BASE_URL}/v1/completions", json={
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": OUTPUT_LEN,
        "temperature": 0.0,
    }, timeout=120)
    usage = r.json().get("usage", {})
    print(f"Req {i}: in={usage.get('prompt_tokens')} out={usage.get('completion_tokens')}")

# 3) Stop profiler — trace export can take minutes for large files!
requests.post(f"{BASE_URL}/stop_profile", timeout=300)
```

> **Gotcha**: `stop_profile` can take **minutes** to serialize the trace JSON (multi-GB files). Use `timeout=300` or higher. If it times out, the trace is still being written — poll the output directory for file appearance.

---

## Phase 4: Collect Trace Output

After `stop_profile` completes, two files appear in the profiler directory:

| File | Description |
|------|-------------|
| `*-rank-0.*.pt.trace.json` | Full Chrome Trace Event JSON (loadable in `chrome://tracing` or Perfetto) |
| `profiler_out_0.txt` | PyTorch profiler tabular summary — a quick-look text reference |

The text summary (`profiler_out_0.txt`) gives an instant overview without parsing the large JSON. It lists each kernel/operator with Self CUDA time, percentage, call count, and total FLOP count.

---

## Phase 5: Parse and Analyze the Trace JSON

### Chrome Trace Event Format

The trace JSON contains an array of events. The most important fields:

```json
{
  "ph": "X",           // "X" = complete duration event (the ones to analyze)
  "name": "kernel_name",
  "cat": "kernel",     // "kernel" = device-side GPU, "cpu_op" = CPU-side, etc.
  "ts": 1234567,       // start timestamp in microseconds
  "dur": 42,           // duration in microseconds
  "tid": 7,            // thread ID
  "pid": 0,            // process ID
  "args": {}           // extra metadata (shapes, etc.)
}
```

### Critical: Separating True GPU Time from CPU Overhead

The trace contains **both** device-side kernel events and CPU-side HIP runtime calls. You **MUST** filter by the `cat` field to avoid inflating GPU time:

```python
true_gpu_kernels = []
hip_runtime_calls = []
cpu_ops = []

for ev in events:
    if ev.get('ph') != 'X':
        continue
    cat = ev.get('cat', '')
    name = ev.get('name', '')

    if cat == 'kernel' or cat in ('gpu_memcpy', 'gpu_memset'):
        true_gpu_kernels.append(ev)       # Actual device-side execution
    elif name.startswith('hip'):
        hip_runtime_calls.append(ev)      # CPU-side launch/sync overhead
    else:
        cpu_ops.append(ev)                # Python/PyTorch CPU ops
```

> **Gotcha**: If you lump HIP runtime calls (e.g. `hipLaunchKernel`, `hipModuleLaunchKernel`) into "GPU time", your analysis will show a large phantom "Other" category that inflates totals and misleads optimization decisions.

### AMD ROCm Kernel Name Patterns

ROCm kernel names are verbose. These patterns help categorize them:

| Kernel Name Pattern | Operation Category | Typical Phase |
|---------------------|--------------------|---------------|
| `wvSplitK*` | GEMM — small-batch (rocBLAS) | Decode |
| `Cijk_*` | GEMM — large-batch (rocBLAS) | Prefill |
| `kernel_unified_attention_3d` | Paged attention (single-token query) | Decode |
| `kernel_unified_attention_2d` | Attention (multi-token query) | Prefill |
| `reduce_segments` | Attention output reduction | Decode |
| `*rms_norm_kernel*` | RMSNorm | Both |
| `*fused_add_rms_norm*` or `*_typeConvert*` | Residual + RMSNorm fused | Both |
| `*act_and_mul_kernel*` or `*silu*` | SiLU activation + gate mul | Both |
| `*rotary_embedding_kernel*` | Rotary positional embedding (RoPE) | Both |
| `reshape_and_cache_kernel_flash` | KV cache write | Both |
| `Memcpy DtoD` | Device-to-device memory copy | Both |

### Categorization Code

```python
categories = {
    'GEMM (small-batch/decode)':    ['wvSplitK'],
    'GEMM (large-batch/prefill)':   ['Cijk_'],
    'Attention (decode)':           ['kernel_unified_attention_3d'],
    'Attention (prefill)':          ['kernel_unified_attention_2d'],
    'Attention (reduce)':           ['reduce_segments'],
    'RMSNorm':                      ['rms_norm_kernel'],
    'Fused Add + RMSNorm':          ['fused_add_rms_norm', '_typeConvert'],
    'SiLU + Mul':                   ['act_and_mul_kernel', 'silu'],
    'Rotary Embedding':             ['rotary_embedding_kernel'],
    'KV Cache':                     ['reshape_and_cache'],
    'Memory Copy':                  ['Memcpy DtoD'],
    'Sampling':                     ['reduce_kernel', 'gather_kernel'],
}

from collections import defaultdict
cat_stats = defaultdict(lambda: {'total_us': 0, 'count': 0})

for ev in true_gpu_kernels:
    name = ev['name']
    dur = ev.get('dur', 0)
    matched = False
    for cat_name, keywords in categories.items():
        if any(kw in name for kw in keywords):
            cat_stats[cat_name]['total_us'] += dur
            cat_stats[cat_name]['count'] += 1
            matched = True
            break
    if not matched:
        cat_stats['Other']['total_us'] += dur
        cat_stats['Other']['count'] += 1
```

### Prefill vs Decode from CPU Markers

vLLM stamps each engine step with a named marker following this pattern:

```
execute_context_<N>(<ctx_tokens>)_generation_<M>(<gen_tokens>)
```

- **Prefill**: `ctx_tokens > 0` and `gen_tokens == 0` (e.g. `execute_context_1(1024)_generation_0(0)`)
- **Decode**: `ctx_tokens == 0` and `gen_tokens > 0` (e.g. `execute_context_0(0)_generation_1(1)`)

```python
import re

prefill_count = decode_count = 0
prefill_total_us = decode_total_us = 0

pattern = re.compile(r'execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)')

for ev in cpu_events:
    m = pattern.search(ev.get('name', ''))
    if not m:
        continue
    ctx_tokens = int(m.group(2))
    gen_tokens = int(m.group(4))
    if ctx_tokens > 0 and gen_tokens == 0:
        prefill_count += 1
        prefill_total_us += ev.get('dur', 0)
    elif ctx_tokens == 0 and gen_tokens > 0:
        decode_count += 1
        decode_total_us += ev.get('dur', 0)
```

---

## Phase 6: Interpret Results

### Aggregation

```python
total_gpu_us = sum(s['total_us'] for s in cat_stats.values())
sorted_cats = sorted(cat_stats.items(), key=lambda x: x[1]['total_us'], reverse=True)

print(f"Total GPU kernel time: {total_gpu_us/1e3:.1f} ms\n")
for cat_name, s in sorted_cats:
    pct = 100 * s['total_us'] / total_gpu_us
    print(f"  {cat_name:<35} {s['total_us']/1000:>8.1f} ms  {pct:>5.1f}%  ({s['count']} calls)")
```

### Computing Per-Iteration Breakdown

Use the model architecture and kernel call counts to derive per-step timings:

```python
# Typical transformer: N layers, each with K GEMM calls
# Total GEMM calls per engine step = num_layers * gemms_per_layer
# decode_steps = decode_gemm_call_count / (num_layers * gemms_per_layer)
# per_step_ms = total_kernel_ms / decode_steps

gemms_per_layer = 4  # typical: QKV, O, gate_up, down (varies by architecture)
decode_steps = decode_gemm_calls / (num_layers * gemms_per_layer)
per_step_gemm_ms = decode_gemm_total_ms / decode_steps
```

### Common Patterns to Expect

- **Decode phase** is usually dominated by **small-batch GEMM** (`wvSplitK`), which is memory-bandwidth-bound (not compute-bound) since batch=1 has very low arithmetic intensity.
- **Prefill phase** is dominated by **large GEMM** (`Cijk_*`), which is typically compute-bound.
- **Normalization** (RMSNorm / Fused Add+RMSNorm) often appears as the second-largest category due to high call count (2 per layer per step).
- **Attention** has different kernels for prefill (2D, longer) and decode (3D, shorter but more calls).

---

## Common Gotchas & Solutions

| Problem | Symptom | Solution |
|---------|---------|----------|
| GPU not visible to subprocess | `RuntimeError: No HIP GPUs are available` | Only set `CUDA_VISIBLE_DEVICES` + `HIP_VISIBLE_DEVICES` in subprocess env. Never set `ROCR_VISIBLE_DEVICES` in the parent process. |
| `stop_profile` timeout | `ReadTimeout` after default timeout | Increase timeout to 300s+. Trace export of multi-GB files takes minutes. The file is still being written — poll the output directory. |
| GPU "free" but server fails | `ValueError: Free memory < desired utilization` | GPU has hidden allocations. Use `torch.cuda.mem_get_info()` directly instead of trusting `rocm-smi`. |
| Trace JSON too large to load | `MemoryError` on `json.load()` | Multi-GB traces need substantial RAM. Use a machine with enough memory, or stream-parse with the `ijson` library. |
| Inflated "Other" GPU category | HIP runtime calls counted as GPU time | Filter strictly: only count events with `cat == "kernel"` as true GPU kernel time. |
| Can't tell prefill vs decode GEMM | Different kernel names for each phase | Decode uses `wvSplitK` (small M), prefill uses `Cijk_*` (large M). They are distinct rocBLAS kernels. |
| No `profiler_out_0.txt` | Text summary file missing | Ensure `torch_profiler_dump_cuda_time_total: true` in the profiler config. |

---

## Reusable Script Template

A minimal complete script — change the configuration variables at the top for any model:

```python
#!/usr/bin/env python3
"""Profile any vLLM model on AMD GPU."""
import subprocess, os, sys, time, random, json, requests, signal
from collections import defaultdict

# ========== CONFIGURATION (edit these) ==========
GPU_ID = 0                          # Index of a free GPU
MODEL = "your-org/your-model"       # Any HuggingFace model ID
PORT = 8192
PROFILER_DIR = "/tmp/vllm_profile"
NUM_REQUESTS = 10
INPUT_LEN = 1024
OUTPUT_LEN = 32
MAX_MODEL_LEN = 4096
# =================================================

os.makedirs(PROFILER_DIR, exist_ok=True)

def gen_prompt(n=1024):
    vocab = "the quick brown fox jumps over lazy dog cat bird fish tree house car".split()
    return " ".join(random.choice(vocab) for _ in range(n))

# 1. Start server
env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
env["HIP_VISIBLE_DEVICES"] = str(GPU_ID)
prof_cfg = json.dumps({"profiler":"torch","torch_profiler_dir":PROFILER_DIR,
    "torch_profiler_with_stack":True,"torch_profiler_with_flops":True,
    "torch_profiler_record_shapes":True,"torch_profiler_use_gzip":False,
    "torch_profiler_dump_cuda_time_total":True,"ignore_frontend":True})
proc = subprocess.Popen([sys.executable,"-m","vllm.entrypoints.openai.api_server",
    "--model",MODEL,"--port",str(PORT),"--max-model-len",str(MAX_MODEL_LEN),
    "--gpu-memory-utilization","0.90","--enforce-eager","--profiler-config",prof_cfg],
    env=env, stdout=open(f"{PROFILER_DIR}/server.log","w"), stderr=subprocess.STDOUT)

# 2. Wait for ready
for _ in range(100):
    try:
        if requests.get(f"http://localhost:{PORT}/health",timeout=5).status_code==200: break
    except: pass
    time.sleep(3)

# 3. Profile
random.seed(42)
prompts = [gen_prompt(INPUT_LEN) for _ in range(NUM_REQUESTS)]
requests.post(f"http://localhost:{PORT}/start_profile", timeout=30); time.sleep(2)
for i, p in enumerate(prompts):
    r = requests.post(f"http://localhost:{PORT}/v1/completions",
        json={"model":MODEL,"prompt":p,"max_tokens":OUTPUT_LEN,"temperature":0.0}, timeout=120)
    u = r.json().get("usage", {})
    print(f"Req {i}: in={u.get('prompt_tokens')} out={u.get('completion_tokens')}")
requests.post(f"http://localhost:{PORT}/stop_profile", timeout=300)
time.sleep(15)

# 4. Shutdown
proc.send_signal(signal.SIGTERM); proc.wait(timeout=30)

# 5. Analyze
trace_file = [f for f in os.listdir(PROFILER_DIR) if f.endswith('.trace.json')][0]
with open(os.path.join(PROFILER_DIR, trace_file)) as f: data = json.load(f)
events = data.get('traceEvents', data) if isinstance(data, dict) else data
kernel_stats = defaultdict(lambda: {'count': 0, 'total_us': 0})
for ev in events:
    if ev.get('ph') == 'X' and ev.get('cat') == 'kernel':
        s = kernel_stats[ev['name']]; s['count'] += 1; s['total_us'] += ev.get('dur', 0)
total = sum(s['total_us'] for s in kernel_stats.values())
print(f"\nTotal GPU kernel time: {total/1e3:.1f} ms")
for name, s in sorted(kernel_stats.items(), key=lambda x: x[1]['total_us'], reverse=True)[:15]:
    print(f"  {s['total_us']/1000:8.1f} ms ({100*s['total_us']/total:5.1f}%) "
          f"[{s['count']:>6} calls] {name[:80]}")
```

---

## Files Produced

| File | Description |
|------|-------------|
| `*-rank-0.*.pt.trace.json` | Full Chrome trace (open in `chrome://tracing` or Perfetto UI) |
| `profiler_out_0.txt` | PyTorch profiler text summary table |
| `server.log` | vLLM server stdout/stderr (startup info, throughput stats) |
