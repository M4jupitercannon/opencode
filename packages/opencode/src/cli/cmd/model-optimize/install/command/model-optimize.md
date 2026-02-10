---
description: "End-to-end HuggingFace model optimization pipeline. Usage: /model-optimize <model_name> [output_dir]"
agent: model-opt
---

# End-to-End Model Optimization Pipeline

## Target Model
- **HuggingFace Model**: $1
- **Output Directory**: $2 (if not specified, use `./model_opt_<model_short_name>`)

## First Steps
1. Parse the model name from `$1` (e.g., "Qwen/Qwen3-8B" → short name "Qwen3-8B")
2. Determine output directory: use `$2` if provided, otherwise `./model_opt_<short_name>`
3. Create the output directory structure:

```
<output_dir>/
├── venv/           # Project-specific Python virtual environment
├── model/          # Downloaded model files
├── demo/           # Demo scripts for running the model
├── profile/        # Profiling results
├── problems/       # Kernel problems for optimization
├── optimized/      # Optimized kernels
├── report/         # Final optimization report
├── config.json     # Configuration
└── progress.json   # Progress tracking
```

4. Create `config.json` with model info and directory paths
5. Create `progress.json` to track phases

## ⚠️ CRITICAL RULES
- **ALWAYS activate venv before running Python**: `source <output_dir>/venv/bin/activate`
- **NEVER modify system Python packages** in /opt/, /usr/, or site-packages/
- **ALL optimization decisions MUST be based on actual profiling data**
- **Update progress.json after each phase!**

---

# Phase 0: Environment Setup

## Goal
Create an isolated Python virtual environment with ROCm detection.

## Steps

### 1. Detect ROCm Version
```bash
ROCM_VERSION=$(cat /opt/rocm/.info/version 2>/dev/null | head -1 | cut -d'-' -f1 || echo "6.0")
echo "Detected ROCm version: $ROCM_VERSION"
```

### 2. Create venv with system site-packages
```bash
cd <output_dir>
python3 -m venv venv --system-site-packages
source venv/bin/activate
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python3 -c "import triton; print('Triton available')"
```

### 3. Install missing packages
```bash
source <output_dir>/venv/bin/activate
python3 -c "import transformers" 2>/dev/null || pip install transformers
python3 -c "import diffusers" 2>/dev/null || pip install diffusers
python3 -c "import accelerate" 2>/dev/null || pip install accelerate
```

Update progress.json: phases_completed.append("env")

---

# Phase 1: Model Download

## Goal
Download the HuggingFace model to `<output_dir>/model/`

## Steps
1. Check if model already exists (look for config.json in model dir)
2. Download using huggingface_hub:

```python
from huggingface_hub import snapshot_download
snapshot_download("$1", local_dir="<output_dir>/model")
```

Update progress.json: phases_completed.append("download")

---

# Phase 2: Generate Demo Script

## Goal
Create a working inference demo script in `<output_dir>/demo/demo.py`.

## Steps
1. Read model's config.json to detect model type (text-gen, image-gen, etc.)
2. Generate appropriate demo.py
3. Create `<output_dir>/demo/patches/__init__.py` for compatibility fixes
4. Test: `python demo.py`

### For Text Generation Models
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
MODEL_PATH = "<output_dir>/model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, device_map="cuda", trust_remote_code=True)
inputs = tokenizer("Hello, I am", return_tensors="pt").to("cuda")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
```

### For Image Generation Models
```python
import torch
from diffusers import DiffusionPipeline
pipe = DiffusionPipeline.from_pretrained("<output_dir>/model", torch_dtype=torch.float16).to("cuda")
image = pipe("A cat sitting on a couch").images[0]
image.save("output.png")
```

Update progress.json: phases_completed.append("demo")

---

# Phase 3: Fix Compatibility Issues

## Goal
If demo.py fails, diagnose and fix using monkey-patching in `<output_dir>/demo/patches/`.

- Never modify system packages
- Create patches in the project's patches/ directory
- Update demo.py to import patches first

Update progress.json: phases_completed.append("compatibility")

---

# Phase 4: Performance Profiling

## Goal
Profile the model to identify bottleneck operators/kernels.

## ⚠️ Use rocprof for GPU profiling on ROCm
PyTorch's profiler doesn't show GPU times on ROCm!

```bash
source <output_dir>/venv/bin/activate
cd <output_dir>/profile
rocprof --stats -o rocprof_results.csv python <output_dir>/demo/demo.py
```

Parse rocprof_results.stats.csv to identify bottlenecks. Save to `<output_dir>/profile/bottlenecks.json`.

### Shape Capture
If `~/.config/opencode/scripts/shape_capture.py` exists, copy and use it:
```bash
cp ~/.config/opencode/scripts/shape_capture.py <output_dir>/profile/
python shape_capture.py --config <output_dir>/config.json
```

Otherwise create a lightweight shape capture script to record input shapes during inference.

Update progress.json: phases_completed.append("profile")

---

# Phase 5: Generate Problem Files

## Goal
Convert bottlenecks into Problem files for kernel-optimize.

## CRITICAL: Analyze Fusion Opportunities FIRST

If `~/.config/opencode/scripts/analyze_fusion.py` exists, use it:
```bash
cp ~/.config/opencode/scripts/analyze_fusion.py <output_dir>/profile/
python analyze_fusion.py --config <output_dir>/config.json
```

### Common Fusion Patterns
| Pattern | Operators | Expected Speedup |
|---------|-----------|------------------|
| ResidualNorm | add + rmsnorm | 1.2-1.5x |
| SwiGLU | silu + mul | 1.3-1.8x |
| RoPE | cos + sin + cat | 1.2-1.5x |

### Create Fused Problem Files First (HIGH priority)
Then individual problem files for operators >5% time that can't be fused.

### Problem File Format
```python
import torch, torch.nn as nn
class Model(nn.Module):
    def forward(self, *inputs):
        # Reference PyTorch implementation
        ...
# Use ACTUAL shapes from profiling!
def get_inputs():
    return [torch.randn(..., device='cuda')]
def get_init_inputs():
    return []
```

### Generate optimization_manifest.json
```json
{"model": "$1", "optimizations": [
  {"name": "fused_rmsnorm", "file": "problem_fused_rmsnorm.py", "priority": "HIGH", "enabled": true},
  {"name": "linear_gemm", "file": "problem_linear.py", "priority": "LOW", "enabled": false}
]}
```

Update progress.json: phases_completed.append("problems")

---

# Phase 6: Run Kernel Optimization

## Goal
Optimize kernels using `opencode kernel-optimize` in parallel.

```bash
cd <output_dir>/problems
# HIGH priority - run in parallel
opencode kernel-optimize --src problem_fused_rmsnorm.py --goal 1.5 &
opencode kernel-optimize --src problem_fused_rope.py --goal 1.5 &
wait
```

### Priority Order
| Priority | Kernel | Goal |
|----------|--------|------|
| HIGH | Fused Residual+Norm | 1.5x |
| HIGH | Fused RoPE | 1.5x |
| MEDIUM | SwiGLU/GELU | 1.3x |
| LOW | Linear/GEMM | 1.1x |
| SKIP | Simple add/copy | - |

### Verify at ACTUAL shapes before integrating
Only copy kernels faster at real inference shapes to `<output_dir>/optimized/`.

### Phase 6.5: Additional Optimizations
Consider AITER Flash Attention if attention >10% and seq >64. Always benchmark first.

Update progress.json: phases_completed.append("optimize")

---

# Phase 7: Integration & Final Testing

## Goal
Integrate optimized kernels via monkey-patching and MEASURE ACTUAL end-to-end performance.

## ⚠️ MUST:
1. Measure ACTUAL end-to-end speedup (not estimated)
2. Generate BOTH original AND optimized outputs with same seed (SEED=42)
3. Save comparison outputs to `<output_dir>/report/comparison_outputs/`

### Create integrate.py
- Import all optimized kernels (ModelNew from *_opt.py)
- Provide `apply_all_patches(model)` function
- Patch RMSNorm/LayerNorm, activations, attention

### Create measure_e2e.py
- Load model WITHOUT optimizations → benchmark
- Apply patches → benchmark WITH optimizations
- Generate comparison outputs with fixed seed
- Save results to comparison_results.json

Update progress.json: phases_completed.append("integrate")

---

# Phase 8: Generate Final Report

## Goal
Create `<output_dir>/report/optimization_report.md` with:

- ACTUAL MEASURED speedup (from comparison_results.json)
- Bottleneck analysis table
- Per-kernel optimization results
- Comparison outputs (text or images)
- Recommendations

Update progress.json: phase="complete", phases_completed.append("report")

---

# EXECUTION INSTRUCTIONS

1. **Execute phases in order**: 0 → 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8
2. **Update progress.json after each phase**
3. **If a phase fails, debug and fix before proceeding**
4. **For kernel-optimize, use the existing `opencode kernel-optimize` command**
5. **NEVER modify system libraries - only use monkey patching and project venv**

## Optimization Priority
1. FIRST: Fused kernels (residual+norm, swiglu)
2. THEN: Individual kernels
3. SKIP: Vendor-optimized (rocBLAS GEMM) and trivial ops

Begin now with Phase 0: Environment Setup.

