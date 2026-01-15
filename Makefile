.PHONY: install build clean

# One-line install after cloning: make install
install: build
	@echo "Installing opencode to /usr/local/bin..."
	@sudo cp packages/opencode/dist/opencode-linux-x64/bin/opencode /usr/local/bin/ 2>/dev/null || \
		cp packages/opencode/dist/opencode-linux-x64/bin/opencode /usr/local/bin/
	@echo ""
	@echo "✓ OpenCode installed successfully!"
	@echo ""
	@echo "Usage:"
	@echo "  # With free model (no API key needed)"
	@echo "  opencode kernel-optimize --src source.py -m opencode/glm-4.7-free"
	@echo ""
	@echo "  # With AMD Gateway"
	@echo "  export LLM_GATEWAY_KEY=your-key"
	@echo "  opencode kernel-optimize --src source.py -m amd-anthropic/claude-opus-4-5"

build:
	@command -v bun >/dev/null 2>&1 || { \
		echo "Installing bun..."; \
		curl -fsSL https://bun.sh/install | bash; \
		export PATH="$$HOME/.bun/bin:$$PATH"; \
	}
	@echo "Installing dependencies..."
	@bun install
	@echo "Building opencode..."
	@cd packages/opencode && bun run build --single

clean:
	@rm -rf packages/opencode/dist
	@echo "Cleaned build artifacts"

