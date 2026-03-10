.PHONY: install build clean

# One-line install after cloning: make install
install: build
	@mkdir -p $(HOME)/bin
	@echo "Installing opencode to $(HOME)/bin..."
	@cp packages/opencode/dist/opencode-linux-x64/bin/opencode $(HOME)/bin/
	@echo ""
	@echo "✓ OpenCode installed successfully to $(HOME)/bin/opencode!"
	@echo ""
	@if echo "$$PATH" | grep -q "$(HOME)/bin"; then \
		echo "Usage:"; \
	else \
		echo "Add $(HOME)/bin to your PATH:"; \
		echo "  export PATH=\"\$$HOME/bin:\$$PATH\""; \
		echo "  # (add the above to ~/.bashrc to make it permanent)"; \
		echo ""; \
		echo "Usage:"; \
	fi
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
	}; \
	export PATH="$$HOME/.bun/bin:$$PATH"; \
	export HUSKY=0; \
	echo "Installing dependencies..."; \
	bun install --ignore-scripts; \
	echo "Building opencode..."; \
	cd packages/opencode && bun run build --single

clean:
	@rm -rf packages/opencode/dist
	@echo "Cleaned build artifacts"

