#!/usr/bin/env bash
set -euo pipefail

#
# Local benchmark runner — reads all parameters from the master YAML config
# so you don't have to set TP, ISL, OSL, etc. manually.
# Can be run from anywhere — auto-clones the InferenceX repo if needed.
#
# Usage:
#   ./run_local.sh <config-key> [--conc <N>] [--seq-len <ISLxOSL>] [--dry-run]
#                               [--hf-cache <path>] [--repo-dir <path>]
#                               [--profile]
#
# Examples:
#   # Run from anywhere — repo is auto-cloned into ./InferenceX
#   ./run_local.sh kimik2.5-int4-mi355x-vllm
#
#   # Point to an existing clone
#   ./run_local.sh kimik2.5-int4-mi355x-vllm --repo-dir /path/to/InferenceX
#
#   # Run only concurrency=64
#   ./run_local.sh kimik2.5-int4-mi355x-vllm --conc 64
#
#   # Run only 1k1k sequence length
#   ./run_local.sh kimik2.5-int4-mi355x-vllm --seq-len 1k1k
#
#   # Preview the docker commands without running
#   ./run_local.sh kimik2.5-int4-mi355x-vllm --dry-run
#
#   # Run with profiling enabled (trace saved to ./profiles/)
#   ./run_local.sh kimik2.5-int4-mi355x-vllm --seq-len 1k1k --conc 4 --profile
#

CONFIG_KEY="${1:?Usage: $0 <config-key> [--conc N] [--seq-len ISLxOSL] [--dry-run] [--profile]}"
shift

FILTER_CONC=""
FILTER_SEQ=""
DRY_RUN=false
PROFILE=false
HF_CACHE="${HF_HUB_CACHE:-${HOME}/.cache/huggingface}"
REPO_DIR=""
REPO_URL="https://github.com/SemiAnalysisAI/InferenceX.git"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --conc)       FILTER_CONC="$2"; shift 2 ;;
        --seq-len)    FILTER_SEQ="$2"; shift 2 ;;
        --dry-run)    DRY_RUN=true; shift ;;
        --profile)    PROFILE=true; shift ;;
        --hf-cache)   HF_CACHE="$2"; shift 2 ;;
        --repo-dir)   REPO_DIR="$2"; shift 2 ;;
        *)            echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Clone the repo if not running from inside InferenceX
if [[ -f "$(pwd)/.github/configs/amd-master.yaml" ]]; then
    SCRIPT_DIR="$(pwd)"
elif [[ -n "$REPO_DIR" && -d "$REPO_DIR" ]]; then
    SCRIPT_DIR="$(cd "$REPO_DIR" && pwd)"
else
    REPO_DIR="${REPO_DIR:-InferenceX}"
    if [[ ! -d "$REPO_DIR" ]]; then
        echo "Cloning InferenceX repo..."
        git clone "$REPO_URL" "$REPO_DIR"
    else
        echo "Using existing repo at $REPO_DIR, pulling latest..."
        git -C "$REPO_DIR" pull --ff-only || true
    fi
    SCRIPT_DIR="$(cd "$REPO_DIR" && pwd)"
fi

# Detect config file based on runner type in the config key
if [[ "$CONFIG_KEY" == *mi3* ]]; then
    CONFIG_FILE=".github/configs/amd-master.yaml"
else
    CONFIG_FILE=".github/configs/nvidia-master.yaml"
fi

PROFILE_DIR="${SCRIPT_DIR}/profiles"
if $PROFILE; then
    mkdir -p "$PROFILE_DIR"
fi

echo "=== InferenceX Local Runner ==="
echo "Config key : $CONFIG_KEY"
echo "Config file: $CONFIG_FILE"
echo "HF cache   : $HF_CACHE"
echo "Profiling  : $PROFILE"
if $PROFILE; then
    echo "Profile dir: $PROFILE_DIR"
fi
echo ""

# Generate the full matrix from the master config (run from repo dir so relative paths work)
CONFIGS=$(cd "$SCRIPT_DIR" && python3 utils/matrix_logic/generate_sweep_configs.py \
    test-config --config-files "$CONFIG_FILE" --config-keys "$CONFIG_KEY")

NUM_TOTAL=$(echo "$CONFIGS" | python3 -c "import sys,json; print(len(json.load(sys.stdin)))")
echo "Total benchmark points in config: $NUM_TOTAL"

# Apply filters and run each config point
echo "$CONFIGS" | python3 -c "
import sys, json

configs = json.load(sys.stdin)
filter_conc = '${FILTER_CONC}'
filter_seq = '${FILTER_SEQ}'

seq_map = {'1k1k': (1024,1024), '1k8k': (1024,8192), '8k1k': (8192,1024)}

for c in configs:
    if filter_conc and str(c['conc']) != filter_conc:
        continue
    if filter_seq:
        want_isl, want_osl = seq_map.get(filter_seq, (None, None))
        if want_isl is None:
            print(f'Unknown seq-len: {filter_seq}', file=sys.stderr)
            sys.exit(1)
        if c['isl'] != want_isl or c['osl'] != want_osl:
            continue
    print(json.dumps(c))
" | while IFS= read -r cfg; do
    IMAGE=$(echo "$cfg"   | python3 -c "import sys,json; print(json.load(sys.stdin)['image'])")
    MODEL=$(echo "$cfg"   | python3 -c "import sys,json; print(json.load(sys.stdin)['model'])")
    PREFIX=$(echo "$cfg"  | python3 -c "import sys,json; print(json.load(sys.stdin)['model-prefix'])")
    PREC=$(echo "$cfg"    | python3 -c "import sys,json; print(json.load(sys.stdin)['precision'])")
    FW=$(echo "$cfg"      | python3 -c "import sys,json; print(json.load(sys.stdin)['framework'])")
    RUNNER=$(echo "$cfg"  | python3 -c "import sys,json; print(json.load(sys.stdin)['runner'])")
    ISL=$(echo "$cfg"     | python3 -c "import sys,json; print(json.load(sys.stdin)['isl'])")
    OSL=$(echo "$cfg"     | python3 -c "import sys,json; print(json.load(sys.stdin)['osl'])")
    TP=$(echo "$cfg"      | python3 -c "import sys,json; print(json.load(sys.stdin)['tp'])")
    EP=$(echo "$cfg"      | python3 -c "import sys,json; print(json.load(sys.stdin).get('ep', 1))")
    CONC=$(echo "$cfg"    | python3 -c "import sys,json; print(json.load(sys.stdin)['conc'])")
    MML=$(echo "$cfg"     | python3 -c "import sys,json; print(json.load(sys.stdin)['max-model-len'])")
    EXPNAME=$(echo "$cfg" | python3 -c "import sys,json; print(json.load(sys.stdin)['exp-name'])")

    RESULT_FILENAME="${EXPNAME}_${PREC}_${FW}_tp${TP}-ep${EP}_conc${CONC}"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Running: $EXPNAME | tp=$TP conc=$CONC isl=$ISL osl=$OSL"
    echo "  Image  : $IMAGE"
    echo "  Result : $RESULT_FILENAME"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    BENCHMARK_SCRIPT="benchmarks/single_node/${EXPNAME%%_*}_${PREC}_${RUNNER}.sh"
    if [[ ! -f "$SCRIPT_DIR/$BENCHMARK_SCRIPT" ]]; then
        echo "WARNING: Benchmark script not found: $BENCHMARK_SCRIPT, trying with framework suffix..."
        BENCHMARK_SCRIPT="benchmarks/single_node/${EXPNAME%%_*}_${PREC}_${RUNNER}_${FW}.sh"
    fi

    # Build docker flags based on GPU vendor
    if [[ "$RUNNER" == mi* ]]; then
        GPU_FLAGS="--device=/dev/kfd --device=/dev/dri --group-add video --security-opt seccomp=unconfined"
    else
        GPU_FLAGS="--gpus all"
    fi

    PROFILE_ENV=""
    if $PROFILE; then
        PROFILE_ENV="-e PROFILE=1 \
        -e SGLANG_TORCH_PROFILER_DIR=/workspace/profiles \
        -e VLLM_TORCH_PROFILER_DIR=/workspace/profiles"
    fi

    DOCKER_CMD="docker run --rm \
        --entrypoint /bin/bash \
        $GPU_FLAGS \
        --shm-size 64g \
        --ipc=host \
        --network=host \
        -v ${SCRIPT_DIR}:/workspace \
        -v ${HF_CACHE}:/root/.cache/huggingface \
        -w /workspace \
        -e MODEL=$MODEL \
        -e TP=$TP \
        -e EP_SIZE=$EP \
        -e CONC=$CONC \
        -e ISL=$ISL \
        -e OSL=$OSL \
        -e MAX_MODEL_LEN=$MML \
        -e RANDOM_RANGE_RATIO=0.5 \
        -e RESULT_FILENAME=$RESULT_FILENAME \
        -e PRECISION=$PREC \
        -e FRAMEWORK=$FW \
        -e EXP_NAME=$EXPNAME \
        -e HF_HOME=/root/.cache/huggingface \
        -e HF_HUB_CACHE=/root/.cache/huggingface/hub \
        $PROFILE_ENV \
        $IMAGE \
        $BENCHMARK_SCRIPT"

    if $DRY_RUN; then
        echo "[DRY RUN] Would execute:"
        echo "  $DOCKER_CMD"
    else
        if $PROFILE; then
            echo "Starting benchmark with profiling..."
        else
            echo "Starting benchmark..."
        fi
        eval $DOCKER_CMD
        echo "Finished: $RESULT_FILENAME"
        if $PROFILE; then
            echo "Profile traces saved to: $PROFILE_DIR/"
            ls -lh "$PROFILE_DIR"/*.trace.json* 2>/dev/null || echo "  (no trace files found — check server/framework profiler support)"
        fi
    fi
done

echo ""
if $PROFILE; then
    echo "=== All benchmark + profiling runs complete ==="
    echo "Trace files: $PROFILE_DIR/"
    echo "View traces at: https://ui.perfetto.dev/"
else
    echo "=== All benchmark runs complete ==="
fi