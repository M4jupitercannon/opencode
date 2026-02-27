#!/bin/bash
set -e

# OpenCode with AMD LLM Gateway support installer
# Usage: curl -fsSL https://raw.githubusercontent.com/vivienfanghuagood/opencode/dev/install.sh | bash

REPO="vivienfanghuagood/opencode"
BRANCH="opt-vllm"
INSTALL_DIR="/usr/local/bin"

echo "Installing OpenCode with AMD Gateway support..."

# Check dependencies
command -v bun >/dev/null 2>&1 || {
    echo "Installing bun..."
    curl -fsSL https://bun.sh/install | bash
    export PATH="$HOME/.bun/bin:$PATH"
}

# Create temp directory
TMPDIR=$(mktemp -d)
cd "$TMPDIR"

echo "Downloading source..."
curl -fsSL "https://github.com/$REPO/archive/refs/heads/$BRANCH.tar.gz" | tar xz
cd opencode-*

echo "Installing dependencies..."
# Use a fresh cache dir to avoid stale root-owned bun cache entries
export BUN_INSTALL_CACHE_DIR="$TMPDIR/.bun-cache"
mkdir -p "$BUN_INSTALL_CACHE_DIR"
bun install --ignore-scripts

echo "Building..."

# The build internally runs 'bun add' which triggers husky (#!/usr/bin/env node).
# Create a temporary node -> bun symlink if node is not installed.
NODE_LINK=""
if ! command -v node >/dev/null 2>&1; then
    BUN_DIR="$(dirname "$(command -v bun)")"
    NODE_LINK="$BUN_DIR/node"
    ln -sf "$(command -v bun)" "$NODE_LINK"
fi

cd packages/opencode
bun run build --single

# Clean up temporary symlink
if [ -n "$NODE_LINK" ] && [ -L "$NODE_LINK" ]; then
    rm -f "$NODE_LINK"
fi

echo "Installing to $INSTALL_DIR..."
sudo cp dist/opencode-linux-x64/bin/opencode "$INSTALL_DIR/"
sudo chmod +x "$INSTALL_DIR/opencode"

# Cleanup
cd /
rm -rf "$TMPDIR"

echo ""
echo "✓ OpenCode installed successfully!"
echo ""
echo "Usage:"
echo "  export LLM_GATEWAY_KEY=your-key"
echo "  opencode kernel-optimize --src source.py --goal 2.0"
echo ""

