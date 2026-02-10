#!/bin/bash
#
# model-optimize installer for opencode
#
# One-click install:
#   bash install.sh
#
# Or from remote:
#   curl -fsSL https://raw.githubusercontent.com/vivienfanghuagood/opencode/skill/packages/opencode/src/cli/cmd/model-optimize/install/install.sh | bash
#
# After install, use in opencode TUI:
#   /model-optimize Qwen/Qwen3-8B
#   /model-optimize Qwen/Qwen3-8B ./my_output_dir
#

set -e

# Detect install directory
OPENCODE_DIR="${OPENCODE_CONFIG_DIR:-${XDG_CONFIG_HOME:-$HOME/.config}/opencode}"

echo "╔════════════════════════════════════════════╗"
echo "║  model-optimize installer for opencode     ║"
echo "╚════════════════════════════════════════════╝"
echo ""
echo "Install directory: $OPENCODE_DIR"
echo ""

# Determine script directory (works for both local and piped execution)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}" 2>/dev/null || echo ".")" && pwd)"

# Create directories
mkdir -p "$OPENCODE_DIR/agent"
mkdir -p "$OPENCODE_DIR/command"
mkdir -p "$OPENCODE_DIR/scripts"

# Check if files exist locally (local install) or need to be downloaded
if [ -f "$SCRIPT_DIR/agent/model-opt.md" ]; then
    echo "Installing from local files..."
    cp "$SCRIPT_DIR/agent/model-opt.md" "$OPENCODE_DIR/agent/"
    cp "$SCRIPT_DIR/command/model-optimize.md" "$OPENCODE_DIR/command/"
    cp "$SCRIPT_DIR/scripts/"*.py "$OPENCODE_DIR/scripts/"
else
    echo "Downloading from GitHub..."
    BASE_URL="https://raw.githubusercontent.com/vivienfanghuagood/opencode/skill/packages/opencode/src/cli/cmd/model-optimize/install"
    curl -fsSL "$BASE_URL/agent/model-opt.md" -o "$OPENCODE_DIR/agent/model-opt.md"
    curl -fsSL "$BASE_URL/command/model-optimize.md" -o "$OPENCODE_DIR/command/model-optimize.md"
    curl -fsSL "$BASE_URL/scripts/shape_capture.py" -o "$OPENCODE_DIR/scripts/shape_capture.py"
    curl -fsSL "$BASE_URL/scripts/analyze_fusion.py" -o "$OPENCODE_DIR/scripts/analyze_fusion.py"
fi

echo ""
echo "✅ Installed successfully!"
echo ""
echo "Files installed:"
echo "  $OPENCODE_DIR/agent/model-opt.md"
echo "  $OPENCODE_DIR/command/model-optimize.md"
echo "  $OPENCODE_DIR/scripts/shape_capture.py"
echo "  $OPENCODE_DIR/scripts/analyze_fusion.py"
echo ""
echo "╔════════════════════════════════════════════╗"
echo "║  Usage (in opencode TUI):                  ║"
echo "║                                            ║"
echo "║  /model-optimize Qwen/Qwen3-8B             ║"
echo "║  /model-optimize Qwen/Qwen3-8B ./output    ║"
echo "║                                            ║"
echo "╚════════════════════════════════════════════╝"
echo ""
echo "To uninstall:"
echo "  rm $OPENCODE_DIR/agent/model-opt.md"
echo "  rm $OPENCODE_DIR/command/model-optimize.md"
echo "  rm $OPENCODE_DIR/scripts/shape_capture.py"
echo "  rm $OPENCODE_DIR/scripts/analyze_fusion.py"

