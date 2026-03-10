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

mkdir -p "$OPENCODE_DIR/agent" "$OPENCODE_DIR/command" "$OPENCODE_DIR/scripts" "$OPENCODE_DIR/skills"

BASE_URL="https://raw.githubusercontent.com/vivienfanghuagood/opencode/opt-vllm/packages/opencode/src/cli/cmd/model-optimize/install"
SKILLS_URL="https://raw.githubusercontent.com/vivienfanghuagood/opencode/opt-vllm/packages/opencode/src/cli/cmd/model-optimize/skills"

AGENT_FILES="model-opt.md"
COMMAND_FILES="model-analyzer.md model-optimize.md kernel-optimize.md"
SCRIPT_FILES="kernel_test_runner.py kernel_finalize.py shape_capture.py analyze_fusion.py analyze_kernels.py vllm_trace_extractor.py vllm_benchmark.py generate_vllm_plugin.py validate_pipeline.py split_vllm_trace_annotation.py"
SKILL_FILES="00-env-setup.md 01-model-download.md 02-demo-generate.md 03-compat-fix.md 04-profiling.md 05-problem-generate.md 06-kernel-optimize.md 07-integration.md 08-report-generate.md agent-config.md"

if [ -f "$SCRIPT_DIR/agent/model-opt.md" ]; then
    echo "Installing from local files..."
    cp "$SCRIPT_DIR/agent/model-opt.md" "$OPENCODE_DIR/agent/"
    for f in $COMMAND_FILES; do cp "$SCRIPT_DIR/command/$f" "$OPENCODE_DIR/command/"; done
    cp "$SCRIPT_DIR/scripts/"*.py "$OPENCODE_DIR/scripts/"
    # Skills live one level up from install/ in the source tree
    SKILLS_SRC="$SCRIPT_DIR/../skills"
    if [ -d "$SKILLS_SRC" ]; then
        for f in $SKILL_FILES; do cp "$SKILLS_SRC/$f" "$OPENCODE_DIR/skills/"; done
    fi
else
    echo "Downloading from GitHub..."
    for f in $AGENT_FILES; do curl -fsSL "$BASE_URL/agent/$f" -o "$OPENCODE_DIR/agent/$f"; done
    for f in $COMMAND_FILES; do curl -fsSL "$BASE_URL/command/$f" -o "$OPENCODE_DIR/command/$f"; done
    for f in $SCRIPT_FILES; do curl -fsSL "$BASE_URL/scripts/$f" -o "$OPENCODE_DIR/scripts/$f"; done
    for f in $SKILL_FILES; do curl -fsSL "$SKILLS_URL/$f" -o "$OPENCODE_DIR/skills/$f"; done
fi

echo ""
echo "✅ Installed successfully!"
echo ""
echo "Commands installed:"
echo "  /model-analyze   — environment setup, serving, profiling, bottleneck analysis"
echo "  /model-optimize  — kernel optimization, integration, benchmarking, report"
echo "  /kernel-optimize — optimize a single PyTorch op to Triton"
echo ""
echo "Skills installed (${OPENCODE_DIR}/skills/):"
for f in $SKILL_FILES; do echo "  $f"; done
echo ""
echo "Scripts installed (${OPENCODE_DIR}/scripts/):"
for f in $SCRIPT_FILES; do echo "  $f"; done
echo ""
echo "╔═══════════════════════════════════════════════════╗"
echo "║  Usage (in opencode TUI):                         ║"
echo "║                                                   ║"
echo "║  /model-analyze Qwen/Qwen3-8B                     ║"
echo "║  /model-optimize Qwen/Qwen3-8B                    ║"
echo "║  /kernel-optimize problem_rmsnorm.py 1.5           ║"
echo "║                                                   ║"
echo "╚═══════════════════════════════════════════════════╝"
echo ""
echo "To uninstall:"
echo "  rm $OPENCODE_DIR/agent/model-opt.md"
echo "  rm $OPENCODE_DIR/command/{model-analyzer,model-optimize,kernel-optimize}.md"
echo "  rm -rf $OPENCODE_DIR/skills/"
echo "  rm $OPENCODE_DIR/scripts/{$(echo $SCRIPT_FILES | tr ' ' ',')}"
