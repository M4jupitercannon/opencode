# Phase 0: Environment Setup {{SKIP_LABEL}}

## Goal
Create an isolated Python virtual environment with vLLM-rocm and all required dependencies.

## Steps

### 1. Detect ROCm Version
```bash
ROCM_VERSION=$(cat /opt/rocm/.info/version 2>/dev/null | head -1 | cut -d'-' -f1 || echo "6.0")
echo "Detected ROCm version: $ROCM_VERSION"
```

### 2. Create venv with system site-packages
```bash
cd {{OUTPUT_DIR}}

if [ ! -d "venv" ]; then
  python3 -m venv venv --system-site-packages
  echo "Created venv with system site-packages access"
fi

source venv/bin/activate

python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python3 -c "import triton; print('Triton available')"
```

### 3. Install vLLM-rocm
```bash
source {{OUTPUT_DIR}}/venv/bin/activate
python3 -c "import vllm; print(f'vLLM {vllm.__version__}')" 2>/dev/null || \
  pip install vllm --extra-index-url https://wheels.vllm.ai/rocm/
```

### 4. Install other missing packages
```bash
source {{OUTPUT_DIR}}/venv/bin/activate
python3 -c "import transformers" 2>/dev/null || pip install transformers
python3 -c "import accelerate" 2>/dev/null || pip install accelerate
```

### 5. Verify Installation
```bash
source {{OUTPUT_DIR}}/venv/bin/activate
python3 -c "
import torch, vllm
print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')
print(f'vLLM {vllm.__version__}')
print(f'GPU: {torch.cuda.get_device_name()}')
"
```

### 6. Update progress.json
Update progress.json: phase="env", phases_completed.append("env")
