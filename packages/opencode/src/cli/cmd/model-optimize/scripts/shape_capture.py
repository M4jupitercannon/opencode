#!/usr/bin/env python3
"""
Dynamic Shape Capture System

Hooks into PyTorch operators to capture actual shapes during inference.
Generates shape_ranges.json with min/typical/max for each dimension.

Usage:
    python shape_capture.py                          # uses config.json in parent dir
    python shape_capture.py --config /path/to/config.json

Part of the model-optimize pipeline. Can be used standalone.
"""
import torch
import torch.nn as nn
from collections import defaultdict
from typing import Dict, List, Any, Optional
import json
import os
import sys
import argparse


class ShapeCapture:
    """Lightweight hook system to capture operator shapes during inference."""

    def __init__(self):
        self.shape_records: Dict[str, List[Dict]] = defaultdict(list)
        self.hooks = []
        self.op_counts: Dict[str, int] = defaultdict(int)

    def _create_hook(self, name: str, op_type: str):
        def hook(module, inputs, output):
            record = {"op_type": op_type, "input_shapes": [], "output_shape": None, "dtype": None}
            for inp in inputs:
                if isinstance(inp, torch.Tensor):
                    record["input_shapes"].append(list(inp.shape))
                    if record["dtype"] is None:
                        record["dtype"] = str(inp.dtype)
                elif inp is None:
                    record["input_shapes"].append(None)
            if isinstance(output, torch.Tensor):
                record["output_shape"] = list(output.shape)
            elif isinstance(output, tuple) and len(output) > 0 and isinstance(output[0], torch.Tensor):
                record["output_shape"] = list(output[0].shape)
            self.shape_records[name].append(record)
            self.op_counts[name] += 1
        return hook

    def register_hooks(self, model: nn.Module, target_ops: Optional[List[str]] = None):
        if target_ops is None:
            target_ops = ["LayerNorm", "RMSNorm", "Linear", "Attention", "Conv", "Embedding"]
        for name, module in model.named_modules():
            class_name = module.__class__.__name__
            for op in target_ops:
                if op in class_name:
                    hook = module.register_forward_hook(self._create_hook(name, class_name))
                    self.hooks.append(hook)
                    break
        print(f"Registered {len(self.hooks)} shape capture hooks")
        return self

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def compute_shape_ranges(self) -> Dict[str, Any]:
        import numpy as np
        ranges = {}
        for name, records in self.shape_records.items():
            if not records:
                continue
            op_type = records[0]["op_type"]
            dtype = records[0]["dtype"]
            input_shapes_list = [r["input_shapes"] for r in records if r["input_shapes"]]
            output_shapes_list = [r["output_shape"] for r in records if r["output_shape"]]
            if not input_shapes_list:
                continue
            input_ranges = []
            num_inputs = len(input_shapes_list[0])
            for i in range(num_inputs):
                shapes_i = [s[i] for s in input_shapes_list if s[i] is not None]
                if not shapes_i:
                    input_ranges.append(None)
                    continue
                ndim = len(shapes_i[0])
                dim_ranges = []
                for d in range(ndim):
                    dims = [s[d] for s in shapes_i]
                    dim_ranges.append({
                        "min": int(min(dims)), "max": int(max(dims)),
                        "typical": int(np.median(dims)),
                        "values": sorted(list(set(dims)))[:10]
                    })
                input_ranges.append(dim_ranges)
            output_range = None
            if output_shapes_list:
                ndim = len(output_shapes_list[0])
                output_range = []
                for d in range(ndim):
                    dims = [s[d] for s in output_shapes_list]
                    output_range.append({"min": int(min(dims)), "max": int(max(dims)), "typical": int(np.median(dims))})
            ranges[name] = {
                "op_type": op_type, "dtype": dtype,
                "call_count": len(records),
                "input_shape_ranges": input_ranges,
                "output_shape_range": output_range
            }
        return ranges

    def save_shape_ranges(self, filepath: str):
        ranges = self.compute_shape_ranges()
        output = {
            "metadata": {"total_ops_captured": sum(self.op_counts.values()), "unique_ops": len(self.shape_records)},
            "shape_ranges": ranges
        }
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"Saved shape ranges to {filepath}")
        return output


def capture_shapes_during_inference(model, run_inference_fn, num_runs: int = 10) -> Dict:
    """Convenience function to capture shapes during inference."""
    capture = ShapeCapture()
    capture.register_hooks(model)
    print(f"Running {num_runs} inference passes to capture shape ranges...")
    for i in range(num_runs):
        with torch.no_grad():
            run_inference_fn()
    capture.remove_hooks()
    return capture.compute_shape_ranges()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture dynamic shapes during model inference")
    parser.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "..", "config.json"))
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    profile_dir = config["dirs"]["profile"]
    demo_dir = config["dirs"]["demo"]
    model_dir = config["dirs"]["model"]

    sys.path.insert(0, demo_dir)
    try:
        from patches import apply_all_patches
        apply_all_patches()
    except ImportError:
        pass

    # Try to load and run the model for shape capture
    print(f"Loading model from {model_dir}...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float16, device_map="cuda", trust_remote_code=True)

    capture = ShapeCapture()
    capture.register_hooks(model)

    prompts = ["Hello", "The quick brown fox jumps over the lazy dog",
               "In a hole in the ground there lived a hobbit"]
    for prompt in prompts:
        for _ in range(3):
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            with torch.no_grad():
                model.generate(**inputs, max_new_tokens=20)

    capture.remove_hooks()
    capture.save_shape_ranges(os.path.join(profile_dir, "shape_ranges.json"))

    print("\n=== Shape Ranges Summary ===")
    ranges = capture.compute_shape_ranges()
    for name, info in list(ranges.items())[:10]:
        print(f"\n{name} ({info['op_type']}):")
        print(f"  Calls: {info['call_count']}")
        for i, inp_range in enumerate(info['input_shape_ranges']):
            if inp_range:
                dims_str = ", ".join([f"[{r['min']}-{r['max']}]" for r in inp_range])
                print(f"  Input {i}: ({dims_str})")

