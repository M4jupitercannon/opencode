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
Detect GPU vendor and architecture:
```bash
# Try AMD GPU detection via rocminfo
if command -v rocminfo &>/dev/null; then
    AMD_ARCH=$(rocminfo 2>/dev/null | grep -oP 'gfx\w+' | head -1 | tr '[:upper:]' '[:lower:]')
    if [ -n "$AMD_ARCH" ]; then
        echo "AMD GPU detected: $AMD_ARCH"
    fi
fi

# Try NVIDIA GPU detection via nvidia-smi
if command -v nvidia-smi &>/dev/null; then
    NVIDIA_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.' | sed 's/^/sm_/')
    if [ -n "$NVIDIA_ARCH" ]; then
        echo "NVIDIA GPU detected: $NVIDIA_ARCH"
    fi
fi

# Fail if no GPU was found
if [ -z "$AMD_ARCH" ] && [ -z "$NVIDIA_ARCH" ]; then
    echo "ERROR: No GPU detected" >&2
    exit 1
fi
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
