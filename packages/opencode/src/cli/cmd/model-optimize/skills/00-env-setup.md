# Phase 0: Environment Setup {{SKIP_LABEL}}

## Goal
Create an isolated Python virtual environment for the project with all required dependencies.

## Steps

### 1. Detect ROCm Version
```bash
ROCM_VERSION=$(cat /opt/rocm/.info/version 2>/dev/null | head -1 | cut -d'-' -f1 || echo "6.0")
ROCM_MAJOR=$(echo $ROCM_VERSION | cut -d'.' -f1)
ROCM_MINOR=$(echo $ROCM_VERSION | cut -d'.' -f2)
echo "Detected ROCm version: $ROCM_VERSION (major=$ROCM_MAJOR, minor=$ROCM_MINOR)"
```

### 2. Create venv with system site-packages access (FAST - no install needed!)
```bash
cd {{OUTPUT_DIR}}

if [ ! -d "venv" ]; then
  python3 -m venv venv --system-site-packages
  echo "Created venv with system site-packages access"
fi

source venv/bin/activate

python3 -c "import torch; print(f'PyTorch {torch.__version__} available')"
python3 -c "import triton; print('Triton available')"
```

### 3. Install Only Missing Small Packages (if needed)
```bash
source {{OUTPUT_DIR}}/venv/bin/activate
python3 -c "import transformers" 2>/dev/null || pip install transformers
python3 -c "import diffusers" 2>/dev/null || pip install diffusers
python3 -c "import accelerate" 2>/dev/null || pip install accelerate
```

### 4. Verify Installation
```bash
source {{OUTPUT_DIR}}/venv/bin/activate
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
```

### 5. Update progress.json
Update progress.json: phase="env", phases_completed.append("env")

