# Phase 5: Generate Problem Files for Kernel Optimization {{SKIP_LABEL}}

## Goal
Convert bottleneck operators into Problem files for kernel-optimize.
**IMPORTANT**: Analyze operators for fusion opportunities BEFORE creating individual problem files.

## STEP 0: Review Per-Shape Kernel Analysis (from Phase 4)

Before creating problem files, review `{{PROFILE_DIR}}/analysis_summary.json` and TraceLens `unified_perf_summary.csv` to understand:
- Which **operator categories** dominate GPU time (GEMM, Attention, Norm, Activation, ...)
- For each category, which **specific shapes** are the hottest
- Use the top (category, shape) pairs to set **priorities** and pick **exact dimensions** for problem files

```bash
cat {{PROFILE_DIR}}/analysis_summary.json | python3 -c "
import json, sys
data = json.load(sys.stdin)
for phase, pdata in data.get('phases', {}).items():
    print(f'=== Phase: {phase} ===')
    for cat in pdata.get('categories', []):
        print(f'  {cat[\"category\"]:15s} {cat[\"percentage\"]:5.1f}%  ({cat[\"total_kernel_time_ms\"]:.2f}ms, {cat[\"count\"]} ops)')
    print()
    for op in pdata.get('unified_perf_summary', [])[:10]:
        eff_str = f'{op[\"tflops_per_s_mean\"]:.1f} TFLOPS/s' if 'tflops_per_s_mean' in op else ''
        print(f'  {op[\"name\"]:45s} {op[\"percentage\"]:5.1f}%  {eff_str}')
"
```

## STEP 1: Operator Fusion Analysis (CRITICAL)

A standalone `analyze_fusion.py` script is provided at `{{OUTPUT_DIR}}/scripts/analyze_fusion.py`.
Use it to detect fusable operator patterns:

```bash
cp {{OUTPUT_DIR}}/scripts/analyze_fusion.py {{PROFILE_DIR}}/
cd {{PROFILE_DIR}}
python3 analyze_fusion.py
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

Use ACTUAL shapes from `{{PROFILE_DIR}}/analysis_summary.json` (TraceLens roofline data)
and `{{PROFILE_DIR}}/model_shapes.json`. Focus on the shapes with the highest percentage of total GPU time.

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

# Use ACTUAL shapes from analysis_summary.json / unified_perf_summary.csv
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

Create `{{PROBLEMS_DIR}}/optimization_manifest.json`:

```json
{
  "model": "{{HF_MODEL}}",
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
1. Review analysis_summary.json and unified_perf_summary.csv for shape priorities
2. Run fusion analysis
3. Create fused problem files (HIGH priority)
4. Create individual problem files (MEDIUM/LOW)
5. Generate optimization_manifest.json
6. Update progress.json
