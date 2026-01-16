# Model Optimize - Usage Guide

## Overview

`opencode model-optimize` is an end-to-end tool for optimizing HuggingFace models on AMD GPUs. It automates the entire workflow from model download to optimized inference.

## Prerequisites

- AMD GPU (MI300X/MI355X recommended)
- ROCm installed
- Python 3.10+
- `opencode` CLI installed

## Installation

```bash
# Install opencode from source
git clone https://github.com/amd/opencode.git
cd opencode
make install

# Verify installation
opencode --version
```

## Quick Start

```bash
# Set API key (required for amd-* providers)
export LLM_GATEWAY_KEY=your-api-key

# Optimize a model
opencode model-optimize -m Qwen/Qwen3-8B
```

## Command Syntax

```bash
opencode model-optimize [options]
```

### Required Options

| Option | Alias | Description |
|--------|-------|-------------|
| `--model` | `-m` | HuggingFace model name (e.g., `Qwen/Qwen3-8B`) |

### Optional Options

| Option | Alias | Default | Description |
|--------|-------|---------|-------------|
| `--output` | `-o` | `./model_opt_<name>` | Output directory |
| `--llm` | | `amd-anthropic/claude-opus-4-5` | LLM model for optimization |
| `--skip-download` | | `false` | Skip model download if exists |

## Usage Examples

### Example 1: Optimize Qwen3-8B

```bash
# Full optimization pipeline
opencode model-optimize -m Qwen/Qwen3-8B -o ./qwen3_optimized

# Expected output:
# ============================================
# Model Optimization Pipeline
# ============================================
# HuggingFace Model: Qwen/Qwen3-8B
# Output Directory:  ./qwen3_optimized
# ============================================
```

### Example 2: Continue Optimization (Skip Download)

```bash
# If model already downloaded, skip download phase
opencode model-optimize -m Qwen/Qwen3-8B -o ./qwen3_optimized --skip-download
```

### Example 3: Use Alternative LLM Provider

```bash
# Use free GLM-4.7 model (no API key required)
opencode model-optimize -m Qwen/Qwen3-8B --llm opencode/glm-4.7-free

# Use Minimax M2
opencode model-optimize -m Qwen/Qwen3-8B --llm opencode/minimax-m2
```

### Example 4: Optimize Different Model Types

```bash
# LLaMA model
opencode model-optimize -m meta-llama/Llama-2-7b-hf

# Mistral model
opencode model-optimize -m mistralai/Mistral-7B-v0.1

# Stable Diffusion (experimental)
opencode model-optimize -m stabilityai/stable-diffusion-2-1
```

## Complete Workflow Example: Qwen3-8B

### Step 1: Run Optimization

```bash
export LLM_GATEWAY_KEY=your-api-key
opencode model-optimize -m Qwen/Qwen3-8B -o ./qwen3_8b_opt
```

### Step 2: Monitor Progress

The tool outputs progress in real-time:
```
============================================
Model Optimization Pipeline
============================================
HuggingFace Model: Qwen/Qwen3-8B
Output Directory:  ./qwen3_8b_opt
============================================

Session created, sending prompt...
[Event: message.updated]
|  todowrite 8 todos
|  bash    Create output directory structure
|  bash    Check if model already exists
|  bash    Download model from HuggingFace
...
```

### Step 3: Check Results

After completion (~45 minutes for 8B model):

```bash
# View optimization report
cat ./qwen3_8b_opt/report/optimization_report.md

# View progress summary
cat ./qwen3_8b_opt/progress.json | python3 -m json.tool
```

### Step 4: Use Optimized Model

```python
import sys
sys.path.insert(0, "./qwen3_8b_opt/demo")
sys.path.insert(0, "./qwen3_8b_opt/optimized")

from transformers import AutoModelForCausalLM, AutoTokenizer
from integrate import patch_model

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "./qwen3_8b_opt/model",
    torch_dtype=torch.float16,
    device_map="cuda"
)

# Apply optimizations
model = patch_model(model)

# Run inference (now using optimized kernels)
tokenizer = AutoTokenizer.from_pretrained("./qwen3_8b_opt/model")
inputs = tokenizer("Hello, I am", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
```

## Output Directory Structure

```
qwen3_8b_opt/
├── config.json              # Project configuration
├── progress.json            # Progress tracking
├── model/                   # Downloaded model (16GB)
│   ├── config.json
│   ├── model-*.safetensors
│   ├── tokenizer.json
│   └── tokenizer_config.json
├── demo/
│   ├── demo.py              # Working inference script
│   └── patches/             # Compatibility fixes
├── profile/
│   ├── trace.json           # Chrome trace (open in chrome://tracing)
│   ├── bottlenecks.json     # Top operators by time
│   └── operator_summary.txt # Profiler output
├── problems/
│   ├── problem_linear.py       # Linear/GEMM problem
│   ├── problem_linear_opt.py   # Attempted optimization
│   ├── problem_rmsnorm.py      # RMSNorm problem
│   ├── problem_rmsnorm_opt.py  # Optimized kernel
│   ├── problem_rope.py         # RoPE problem
│   ├── problem_rope_opt.py     # Optimized kernel
│   └── ...
├── optimized/
│   ├── integrate.py            # Model patching script
│   ├── test_integration.py     # Integration test
│   └── problem_*_opt.py        # Production kernels
└── report/
    ├── optimization_report.md  # Comprehensive report
    └── integration_results.json
```

## Understanding the Report

The final report (`report/optimization_report.md`) includes:

### 1. Model Information
- Architecture details (hidden_size, num_layers, etc.)
- Parameter count

### 2. Bottleneck Analysis
| Operator | CUDA Time | % of Total | Optimized |
|----------|-----------|------------|-----------|
| aten::mm (GEMM) | 87.92ms | 42.46% | No (rocBLAS optimal) |
| RMSNorm | 15.08ms | 7.28% | Yes (1.05x) |
| RoPE | 6.03ms | 2.91% | Yes (1.44x) |

### 3. Optimization Results
- Per-kernel speedup
- Integration correctness verification
- End-to-end performance improvement

### 4. Recommendations
- Suggestions for further optimization
- Known limitations

## Troubleshooting

### Issue: LLM_GATEWAY_KEY not set

```bash
# Error: LLM_GATEWAY_KEY environment variable is not set
export LLM_GATEWAY_KEY=your-api-key

# Or use a free model:
opencode model-optimize -m Qwen/Qwen3-8B --llm opencode/glm-4.7-free
```

### Issue: Out of GPU Memory

```bash
# For large models, use smaller batch size in profiling
# Or use gradient checkpointing - edit demo/demo.py after generation
```

### Issue: Model Download Fails

```bash
# Check HuggingFace authentication
huggingface-cli login

# Or set token in environment
export HF_TOKEN=your-hf-token
```

### Issue: Triton Kernel Compilation Fails

```bash
# Clear Triton cache
rm -rf ~/.triton/cache

# Check ROCm version compatibility
rocminfo
```

## Performance Tips

1. **Run on dedicated GPU**: Avoid running other GPU workloads during optimization
2. **Use SSD storage**: Model download and profiling benefit from fast storage
3. **Monitor GPU memory**: Use `rocm-smi` to monitor memory usage
4. **Check Chrome trace**: Open `profile/trace.json` in `chrome://tracing` for detailed analysis

## Available LLM Models

| Model | Provider | Requires Key |
|-------|----------|--------------|
| `amd-anthropic/claude-opus-4-5` | AMD Gateway | Yes |
| `amd-anthropic/claude-sonnet-4` | AMD Gateway | Yes |
| `opencode/glm-4.7-free` | OpenCode | No |
| `opencode/minimax-m2` | OpenCode | No |
| `opencode/kimi-k2` | OpenCode | No |

## Further Reading

- [Design Document](./DESIGN.md) - Architecture and implementation details
- [kernel-optimize Documentation](../kernel-optimize/) - Standalone kernel optimization
- [Triton Tutorial](https://triton-lang.org/main/getting-started/tutorials/) - Learn Triton programming

