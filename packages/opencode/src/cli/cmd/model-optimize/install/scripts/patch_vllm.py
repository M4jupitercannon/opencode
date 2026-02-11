#!/usr/bin/env python3
"""
vLLM Kernel Patcher

Monkey-patches vLLM's internal layers with optimized Triton kernels.
Must be imported BEFORE vLLM loads the model.

Usage:
    # As a pre-import patch (recommended):
    python -c "import patch_vllm; patch_vllm.apply_all()" && vllm serve ...

    # Or inline:
    python -c "
    import sys; sys.path.insert(0, '/path/to/optimized')
    import patch_vllm; patch_vllm.apply_all()
    from vllm.entrypoints.openai.api_server import run_server
    # ... start server
    "

    # Or as part of a wrapper script:
    import patch_vllm
    patch_vllm.apply_all()  # Patches vLLM modules in-place
    # Now start vLLM normally - it will use patched modules

Configuration:
    Set PATCH_DIR env var to point to the directory containing *_opt.py files.
    Or pass the directory to apply_all(patch_dir="/path/to/optimized").

Part of the model-optimize pipeline. Can be used standalone.
"""
import importlib
import os
import sys
import torch
from typing import Any, Callable, Dict, Optional


# Registry of loaded optimized kernels
_registry: Dict[str, Any] = {}
_patch_stats: Dict[str, int] = {}


def load_optimized_kernels(patch_dir: str) -> Dict[str, Any]:
    """Load all *_opt.py files from the patch directory."""
    global _registry
    if patch_dir not in sys.path:
        sys.path.insert(0, patch_dir)

    for fname in os.listdir(patch_dir):
        if not fname.endswith("_opt.py"):
            continue
        module_name = fname[:-3]  # strip .py
        kernel_name = module_name.replace("problem_", "").replace("_opt", "")
        try:
            mod = importlib.import_module(module_name)
            if hasattr(mod, "ModelNew"):
                _registry[kernel_name] = mod.ModelNew
                print(f"  [OK] Loaded optimized kernel: {kernel_name} from {fname}")
        except Exception as e:
            print(f"  [SKIP] {kernel_name}: {e}")

    return _registry


def patch_rmsnorm(optimized_cls=None) -> int:
    """
    Patch vLLM's RMSNorm with an optimized Triton kernel.

    vLLM's RMSNorm dispatches via:
      - self.rocm_norm_func (for norm only)
      - self.rocm_norm_func_with_add (for fused residual + norm)

    We replace the forward_hip method to use our optimized kernel.
    """
    if optimized_cls is None:
        optimized_cls = _registry.get("fused_rmsnorm") or _registry.get("fused_residual_rmsnorm")
    if optimized_cls is None:
        print("  [SKIP] No RMSNorm optimized kernel available")
        return 0

    try:
        import vllm.model_executor.layers.layernorm as ln_module

        original_forward_hip = ln_module.RMSNorm.forward_hip
        kernel_instance_cache = {}

        def patched_forward_hip(self, x, residual=None):
            """Patched forward using optimized Triton RMSNorm."""
            hidden_size = self.weight.shape[0]

            # Get or create optimized kernel instance
            if hidden_size not in kernel_instance_cache:
                try:
                    kernel_instance_cache[hidden_size] = optimized_cls(hidden_size).to(
                        device=self.weight.device, dtype=self.weight.dtype
                    )
                except Exception:
                    # Fallback to original if kernel instantiation fails
                    return original_forward_hip(self, x, residual)

            opt_kernel = kernel_instance_cache[hidden_size]
            # Sync weights
            if hasattr(opt_kernel, 'weight'):
                opt_kernel.weight.data = self.weight.data

            try:
                if residual is not None:
                    # Fused residual + norm path
                    result = opt_kernel(x, residual)
                    if isinstance(result, tuple):
                        return result  # (normed, residual)
                    return result, residual
                else:
                    return opt_kernel(x)
            except Exception:
                # Fallback on any error
                return original_forward_hip(self, x, residual)

        ln_module.RMSNorm.forward_hip = patched_forward_hip
        count = 1  # Patched the class method
        _patch_stats["rmsnorm"] = count
        print(f"  [OK] Patched RMSNorm.forward_hip")
        return count

    except Exception as e:
        print(f"  [FAIL] RMSNorm patch failed: {e}")
        return 0


def patch_activation(optimized_cls=None) -> int:
    """
    Patch vLLM's activation functions (SiLU, GELU) with fused versions.

    vLLM uses vllm.model_executor.layers.activation.SiluAndMul, GeluAndMul, etc.
    """
    if optimized_cls is None:
        optimized_cls = _registry.get("fused_swiglu") or _registry.get("fused_silu_mul")
    if optimized_cls is None:
        print("  [SKIP] No activation optimized kernel available")
        return 0

    try:
        import vllm.model_executor.layers.activation as act_module

        if hasattr(act_module, 'SiluAndMul'):
            original_forward = act_module.SiluAndMul.forward_cuda

            def patched_silu_forward(self, x):
                try:
                    d = x.shape[-1] // 2
                    gate, up = x[..., :d], x[..., d:]
                    opt = optimized_cls()
                    return opt(gate, up)
                except Exception:
                    return original_forward(self, x)

            act_module.SiluAndMul.forward_cuda = patched_silu_forward
            _patch_stats["activation"] = 1
            print(f"  [OK] Patched SiluAndMul.forward_cuda")
            return 1

    except Exception as e:
        print(f"  [FAIL] Activation patch failed: {e}")
    return 0


def patch_rope(optimized_cls=None) -> int:
    """Patch vLLM's Rotary Position Embedding with optimized version."""
    if optimized_cls is None:
        optimized_cls = _registry.get("fused_rope") or _registry.get("rope")
    if optimized_cls is None:
        print("  [SKIP] No RoPE optimized kernel available")
        return 0

    # RoPE patching is model-specific and complex in vLLM
    # The LLM agent should implement this based on the specific model
    print("  [NOTE] RoPE patching requires model-specific implementation")
    return 0


def apply_all(patch_dir: str = None) -> Dict[str, int]:
    """
    Apply all available optimized kernel patches to vLLM.

    Args:
        patch_dir: Directory containing *_opt.py files.
                   Defaults to PATCH_DIR env var or current directory.

    Returns:
        Dict of patch stats (kernel_name -> count_patched)
    """
    if patch_dir is None:
        patch_dir = os.environ.get("PATCH_DIR", os.getcwd())

    print(f"\n{'='*50}")
    print(f"  vLLM Kernel Patcher")
    print(f"  Patch directory: {patch_dir}")
    print(f"{'='*50}\n")

    # Load optimized kernels
    load_optimized_kernels(patch_dir)
    if not _registry:
        print("No optimized kernels found. Skipping patching.")
        return _patch_stats

    print(f"\nLoaded {len(_registry)} optimized kernels: {list(_registry.keys())}")
    print("\nApplying patches...")

    # Apply patches
    patch_rmsnorm()
    patch_activation()
    patch_rope()

    print(f"\n{'='*50}")
    print(f"  Patch Summary: {_patch_stats}")
    print(f"{'='*50}\n")

    return _patch_stats


def get_stats() -> Dict[str, int]:
    """Get current patch statistics."""
    return _patch_stats.copy()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Patch vLLM with optimized Triton kernels")
    parser.add_argument("--patch-dir", default=os.environ.get("PATCH_DIR", "."),
                        help="Directory containing *_opt.py files")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only load kernels, don't apply patches")
    args = parser.parse_args()

    if args.dry_run:
        load_optimized_kernels(args.patch_dir)
        print(f"\nAvailable kernels: {list(_registry.keys())}")
    else:
        apply_all(args.patch_dir)

