# Phase 0: Environment Setup {{SKIP_LABEL}}

## Goal
Search for latest Docker images (e.g. `rocm/vllm-dev`) on Docker Hub that are compatible with vLLM and the host platform, and create a container as an isolated environment with all required dependencies.

If no suitable Docker image is available, fall back to creating an isolated Python virtual environment with vLLM-rocm.

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

Look for a `rocm/vllm-dev` image whose ROCm version and GPU architecture match the host.

```bash
# CDNA (gfx90a, gfx942, …) → mainline nightly tags
# RDNA (gfx1100, gfx1201, …) → navi-specific tags
if [[ "$GPU_ARCH" == gfx9* ]]; then
  TAG_PATTERN="nightly_main"
elif [[ "$GPU_ARCH" == gfx1* ]]; then
  TAG_PATTERN="navi"
else
  TAG_PATTERN=""
fi

ROCM_MAJOR_MINOR=$(echo "$ROCM_VERSION" | grep -oP '^\d+\.\d+')
echo "Searching rocm/vllm-dev tags matching: $TAG_PATTERN (prefer ROCm $ROCM_MAJOR_MINOR)"

# Search Docker Hub tags (navi tags may not be in the most recent page)
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
      -v {{OUTPUT_DIR}}:/workspace/output \
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

  # Pick the GPU with most free memory
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

  # Patch vLLM BlockSize for hybrid architectures (mamba/linear_attention)
  # that require non-standard block sizes (e.g. block_size=528).
  # Safe: the computed block size is always a multiple of 16 (kernel alignment).
  docker exec "$CONTAINER_NAME" bash -c "
    python3 -c \"
from typing import get_args
from vllm.config.cache import BlockSize
sizes = get_args(BlockSize)
if max(sizes) < 512:
    cache_file = '/usr/local/lib/python3.12/dist-packages/vllm/config/cache.py'
    with open(cache_file) as f: src = f.read()
    old = f'BlockSize = Literal[{\\\", \\\".join(str(s) for s in sizes)}]'
    new = old.rstrip(']') + ', 528]'
    if old in src:
        with open(cache_file, 'w') as f: f.write(src.replace(old, new))
        print('Patched BlockSize to include 528 (hybrid arch support)')
    else:
        print('BlockSize definition not found — patch skipped')
else:
    print(f'BlockSize already includes large values: {sizes}')
\"
  "

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

Only execute this if Step 2/3 did not find a suitable image.

```bash
if [ -z "$IMAGE_TAG" ]; then
  echo "Setting up venv environment..."
  cd {{OUTPUT_DIR}}

  if [ ! -d "venv" ]; then
    python3 -m venv venv --system-site-packages
    echo "Created venv with system site-packages access"
  fi

  source venv/bin/activate

  python3 -c "import vllm; print(f'vLLM {vllm.__version__}')" 2>/dev/null || \
    pip install vllm --extra-index-url https://wheels.vllm.ai/rocm/

  python3 -c "import transformers" 2>/dev/null || pip install transformers
  python3 -c "import accelerate" 2>/dev/null || pip install accelerate

  python3 -c "
import torch, vllm
print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')
print(f'vLLM {vllm.__version__}')
print(f'GPU: {torch.cuda.get_device_name()}')
"

  python3 -c "
import json, torch, vllm
best, best_free = 0, 0
for i in range(torch.cuda.device_count()):
    free, _ = torch.cuda.mem_get_info(i)
    if free > best_free: best, best_free = i, free
info = {
    'env_type': 'venv',
    'pytorch': torch.__version__,
    'vllm': vllm.__version__,
    'gpu_count': torch.cuda.device_count(),
    'best_gpu': best,
}
with open('{{OUTPUT_DIR}}/env_info.json', 'w') as f:
    json.dump(info, f, indent=2)
print(json.dumps(info, indent=2))
"
fi
```

### 5. Copy helper scripts

```bash
mkdir -p {{OUTPUT_DIR}}/scripts
cp ~/.config/opencode/scripts/*.py {{OUTPUT_DIR}}/scripts/ 2>/dev/null
ls {{OUTPUT_DIR}}/scripts/
```

### 6. Verify GEAK availability (for Phase 6)

Check if GEAK is installed in the container. If not, attempt to install it.
GEAK is required for automated kernel optimization in Phase 6.

```bash
if [ "$ENV_TYPE" = "docker" ]; then
  docker exec "$CONTAINER_NAME" bash -c "
    geak --help >/dev/null 2>&1 && echo 'GEAK: available' || \
    (cd /workspace/GEAK 2>/dev/null && pip install -e . >/dev/null 2>&1 && echo 'GEAK: installed') || \
    echo 'GEAK: NOT available (Phase 6 will use manual fallback)'
  "
fi
```

### 7. Update progress.json
Update progress.json: phase="env", phases_completed.append("env")

⚠️ **CRITICAL for all subsequent phases**: If `env_type` is `docker` in `env_info.json`, prefix all commands with `docker exec $CONTAINER_NAME bash -c "..."` and use `/workspace/output` as the output directory inside the container. Set `HIP_VISIBLE_DEVICES=$BEST_GPU` to target the GPU with the most free memory.
