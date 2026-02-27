#!/bin/bash
#
# model-optimize & kernel-optimize installer for opencode
#
# One-click install:
#   bash install.sh
#
# Or from remote:
#   curl -fsSL https://raw.githubusercontent.com/vivienfanghuagood/opencode/opt-vllm/packages/opencode/src/cli/cmd/model-optimize/install/install.sh | bash
#
# After install, use in opencode TUI:
#   /model-optimize Qwen/Qwen3-8B
#   /kernel-optimize problem_rmsnorm.py 1.5
#

set -e

OPENCODE_DIR="${OPENCODE_CONFIG_DIR:-${XDG_CONFIG_HOME:-$HOME/.config}/opencode}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}" 2>/dev/null || echo ".")" && pwd)"

echo "╔═══════════════════════════════════════════════════╗"
echo "║  model-optimize & kernel-optimize for opencode    ║"
echo "╚═══════════════════════════════════════════════════╝"
echo ""
echo "Install directory: $OPENCODE_DIR"
echo ""

mkdir -p "$OPENCODE_DIR/agent" "$OPENCODE_DIR/command" "$OPENCODE_DIR/scripts"

BASE_URL="https://raw.githubusercontent.com/vivienfanghuagood/opencode/opt-vllm/packages/opencode/src/cli/cmd/model-optimize/install"

AGENT_FILES="model-opt.md"
COMMAND_FILES="model-optimize.md kernel-optimize.md"
SCRIPT_FILES="kernel_test_runner.py kernel_finalize.py shape_capture.py analyze_fusion.py analyze_kernel_shapes.py vllm_trace_extractor.py vllm_benchmark.py generate_vllm_plugin.py validate_pipeline.py"

if [ -f "$SCRIPT_DIR/agent/model-opt.md" ]; then
    echo "Installing from local files..."
    cp "$SCRIPT_DIR/agent/model-opt.md" "$OPENCODE_DIR/agent/"
    for f in $COMMAND_FILES; do cp "$SCRIPT_DIR/command/$f" "$OPENCODE_DIR/command/"; done
    cp "$SCRIPT_DIR/scripts/"*.py "$OPENCODE_DIR/scripts/"
else
    echo "Downloading from GitHub..."
    for f in $AGENT_FILES; do curl -fsSL "$BASE_URL/agent/$f" -o "$OPENCODE_DIR/agent/$f"; done
    for f in $COMMAND_FILES; do curl -fsSL "$BASE_URL/command/$f" -o "$OPENCODE_DIR/command/$f"; done
    for f in $SCRIPT_FILES; do curl -fsSL "$BASE_URL/scripts/$f" -o "$OPENCODE_DIR/scripts/$f"; done
fi

echo ""
echo "✅ Installed successfully!"
echo ""
echo "Commands installed:"
echo "  /model-optimize  — end-to-end HuggingFace model optimization"
echo "  /kernel-optimize — optimize a single PyTorch op to Triton"
echo ""
echo "Scripts installed (${OPENCODE_DIR}/scripts/):"
for f in $SCRIPT_FILES; do echo "  $f"; done
echo ""
echo "╔═══════════════════════════════════════════════════╗"
echo "║  Usage (in opencode TUI):                         ║"
echo "║                                                   ║"
echo "║  /model-optimize Qwen/Qwen3-8B                    ║"
echo "║  /kernel-optimize problem_rmsnorm.py 1.5           ║"
echo "║                                                   ║"
echo "╚═══════════════════════════════════════════════════╝"
echo ""
echo "To uninstall:"
echo "  rm $OPENCODE_DIR/agent/model-opt.md"
echo "  rm $OPENCODE_DIR/command/{model-optimize,kernel-optimize}.md"
echo "  rm $OPENCODE_DIR/scripts/{$(echo $SCRIPT_FILES | tr ' ' ',')}"
