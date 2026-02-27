#!/usr/bin/env python3
"""
Operator Fusion Analysis

Analyzes profiling bottleneck data to detect fusable operator patterns.
Generates fusion_opportunities.json with recommended fusions.

Usage:
    python analyze_fusion.py                          # uses config.json in parent dir
    python analyze_fusion.py --config /path/to/config.json
    python analyze_fusion.py --bottlenecks /path/to/bottlenecks.json

Part of the model-optimize pipeline. Can be used standalone.
"""
import json
import os
import argparse


def analyze_fusion_opportunities(bottlenecks_file: str, output_dir: str = None):
    with open(bottlenecks_file) as f:
        bottlenecks = json.load(f)

    ops = [(b["name"], b.get("cuda_time_percent", 0)) for b in bottlenecks]
    fusion_opportunities = []

    # Residual + Norm fusion
    has_add = any("add" in op[0].lower() for op in ops)
    has_norm = any("norm" in op[0].lower() or "mean" in op[0].lower() for op in ops)
    if has_add and has_norm:
        add_pct = sum(op[1] for op in ops if "add" in op[0].lower())
        norm_pct = sum(op[1] for op in ops if "norm" in op[0].lower() or "mean" in op[0].lower())
        fusion_opportunities.append({
            "name": "fused_residual_rmsnorm",
            "operators": ["aten::add", "RMSNorm (aten::mean, aten::rsqrt, aten::mul)"],
            "combined_percent": add_pct + norm_pct,
            "expected_speedup": "1.3-1.5x",
            "priority": "HIGH" if add_pct + norm_pct > 10 else "MEDIUM"
        })

    # SiLU + mul fusion (SwiGLU)
    has_silu = any("silu" in op[0].lower() for op in ops)
    has_mul = any("mul" in op[0].lower() and "norm" not in op[0].lower() for op in ops)
    if has_silu and has_mul:
        fusion_opportunities.append({
            "name": "fused_swiglu",
            "operators": ["aten::silu", "aten::mul"],
            "combined_percent": sum(op[1] for op in ops if "silu" in op[0].lower() or
                                    ("mul" in op[0].lower() and "norm" not in op[0].lower())),
            "expected_speedup": "1.3-1.8x",
            "priority": "MEDIUM"
        })

    # QKV projection fusion
    mm_count = sum(1 for op in ops if "mm" in op[0].lower() or "linear" in op[0].lower())
    if mm_count >= 3:
        fusion_opportunities.append({
            "name": "fused_qkv_proj",
            "operators": ["3x aten::mm for Q, K, V"],
            "combined_percent": sum(op[1] for op in ops if "mm" in op[0].lower()) / 3 * 1.5,
            "expected_speedup": "1.2-1.4x",
            "priority": "LOW"
        })

    print("\n=== Fusion Opportunities ===")
    for f in sorted(fusion_opportunities, key=lambda x: x["combined_percent"], reverse=True):
        print(f"\n{f['name']} [{f['priority']}]")
        print(f"  Operators: {', '.join(f['operators'])}")
        print(f"  Combined time: {f['combined_percent']:.1f}%")
        print(f"  Expected speedup: {f['expected_speedup']}")

    if output_dir is None:
        output_dir = os.path.dirname(bottlenecks_file)
    output_file = os.path.join(output_dir, "fusion_opportunities.json")
    with open(output_file, "w") as f:
        json.dump(fusion_opportunities, f, indent=2)
    print(f"\nSaved to {output_file}")
    return fusion_opportunities


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze operator fusion opportunities")
    parser.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "..", "config.json"))
    parser.add_argument("--bottlenecks", help="Direct path to bottlenecks.json")
    args = parser.parse_args()

    if args.bottlenecks:
        analyze_fusion_opportunities(args.bottlenecks)
    else:
        with open(args.config) as f:
            config = json.load(f)
        bottlenecks_file = os.path.join(config["dirs"]["profile"], "bottlenecks.json")
        analyze_fusion_opportunities(bottlenecks_file)

