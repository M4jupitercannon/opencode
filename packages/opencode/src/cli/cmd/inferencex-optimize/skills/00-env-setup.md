# Phase 0: Environment Setup {{SKIP_LABEL}}

## Objective
Verify that all prerequisites are installed and the InferenceX repository is available.

## Steps

### 1. Check Docker
```bash
docker --version
```
If Docker is not available, report an error and stop.

### 2. Check GPU Availability
Detect GPU vendor:
```bash
# For AMD GPUs
ls /dev/kfd /dev/dri 2>/dev/null && echo "AMD GPU detected"

# For NVIDIA GPUs
nvidia-smi 2>/dev/null && echo "NVIDIA GPU detected"
```

### 3. Clone or Update InferenceX Repository
```bash
REPO_DIR="{{REPO_DIR}}"
REPO_URL="{{REPO_URL}}"

if [ -d "$REPO_DIR/.github/configs" ]; then
    echo "Using existing repo at $REPO_DIR, pulling latest..."
    cd "$REPO_DIR" && git pull --ff-only || true
else
    echo "Cloning InferenceX repo..."
    git clone "$REPO_URL" "$REPO_DIR"
fi
```

### 4. Verify Config File Exists
```bash
CONFIG_KEY="{{CONFIG_KEY}}"
if [[ "$CONFIG_KEY" == *mi3* ]]; then
    CONFIG_FILE=".github/configs/amd-master.yaml"
else
    CONFIG_FILE=".github/configs/nvidia-master.yaml"
fi

ls "{{REPO_DIR}}/$CONFIG_FILE"
```

### 5. Check Python3 and Dependencies
```bash
python3 --version
python3 -c "import yaml; import json; print('Dependencies OK')"
```

### 6. Verify HuggingFace Cache
```bash
HF_CACHE="{{HF_CACHE}}"
mkdir -p "$HF_CACHE"
echo "HF cache directory: $HF_CACHE"
```

## Completion
Update progress.json:
```json
{
  "phase": "env",
  "phases_completed": ["env"],
  "current_step": "environment verified"
}
```
