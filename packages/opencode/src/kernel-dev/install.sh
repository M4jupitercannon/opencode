#!/bin/bash
# Install kernel-optimize tool
# Run from opencode project root: ./packages/opencode/src/kernel-dev/install.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="${INSTALL_DIR:-/usr/local/bin}"

echo "Installing kernel-optimize tool..."

# Check if opencode is installed
if ! command -v opencode &> /dev/null; then
    echo "Warning: opencode is not installed in PATH"
    echo "Please install opencode first:"
    echo "  cd /work/opencode/packages/opencode"
    echo "  bun run build --single"
    echo "  cp dist/opencode-linux-x64/bin/opencode /usr/local/bin/"
fi

# Install kernel-optimize
cp "$SCRIPT_DIR/kernel-optimize" "$INSTALL_DIR/kernel-optimize"
chmod +x "$INSTALL_DIR/kernel-optimize"

echo "Installed: $INSTALL_DIR/kernel-optimize"
echo ""
echo "Usage:"
echo "  export LLM_GATEWAY_KEY=your-api-key"
echo "  kernel-optimize --src source.py --target output.py"
echo ""
echo "Run 'kernel-optimize --help' for more information."

