#!/usr/bin/env python3
"""
Problem file generator for the model-optimize pipeline.

Reads Phase 4 analysis outputs (bottlenecks, roofline data, fusion opportunities)
and auto-generates problem files for Phase 6 (GEAK optimization).

Usage:
    python generate_problems.py --profile-dir <dir> --problems-dir <dir>
    python generate_problems.py --review --profile-dir <dir>

Part of the model-optimize pipeline. Can be used standalone.
"""
import argparse
import ast
import csv
import json
import os
import sys

csv.field_size_limit(sys.maxsize)


def review_analysis(profile_dir):
    """STEP 0: Print per-phase category summary from analysis_summary.json."""
    path = os.path.join(profile_dir, "analysis_summary.json")
    if not os.path.isfile(path):
        print("No analysis_summary.json found")
        return
    data = json.load(open(path))
    for phase, pdata in data.get("phases", {}).items():
        print(f"=== Phase: {phase} ===")
        for cat in pdata.get("categories", []):
            print(f"  {cat['category']:15s} {cat['percentage']:5.1f}%  ({cat['total_kernel_time_ms']:.2f}ms, {cat['count']} ops)")
        print()
        for op in pdata.get("unified_perf_summary", [])[:10]:
            eff_str = f"{op['tflops_per_s_mean']:.1f} TFLOPS/s" if "tflops_per_s_mean" in op else ""
            print(f"  {op['name']:45s} {op['percentage']:5.1f}%  {eff_str}")
        print()


def generate_fusion_problems(profile_dir, problems_dir):
    """STEP 2: Generate fused problem files from fusion_opportunities.json."""
    shapes_path = os.path.join(profile_dir, "model_shapes.json")
    fusions_path = os.path.join(profile_dir, "fusion_opportunities.json")
    if not os.path.isfile(fusions_path):
        print("No fusion_opportunities.json -- run analyze_fusion.py first")
        return 0
    shapes = json.load(open(shapes_path))
    H = shapes.get("hidden_size", 4096)
    I = shapes.get("intermediate_size", 11008)

    TEMPLATES = {
        "fused_residual_rmsnorm": f"""import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.bfloat16))
        self.eps = eps
    def forward(self, hidden_states, residual):
        hidden_states = hidden_states + residual
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return (self.weight * hidden_states).to(torch.bfloat16)

batch_size, seq_len, hidden_size = 1, 1, {H}
def get_inputs():
    return [
        torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16, device="cuda"),
        torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16, device="cuda"),
    ]
def get_init_inputs():
    return [hidden_size]
""",
        "fused_swiglu": f"""import torch
import torch.nn as nn

class Model(nn.Module):
    def forward(self, x):
        gate, up = x.chunk(2, dim=-1)
        return torch.nn.functional.silu(gate) * up

batch_size, seq_len, intermediate_size = 1, 1, {I}
def get_inputs():
    return [torch.randn(batch_size, seq_len, intermediate_size * 2, dtype=torch.bfloat16, device="cuda")]
def get_init_inputs():
    return []
""",
    }

    fusions = json.load(open(fusions_path))
    generated = 0
    print(f"Fusion opportunities: {len(fusions)}")
    for f in fusions:
        name = f.get("name", "")
        pct = f.get("combined_percent", 0)
        priority = f.get("priority", "MEDIUM")
        fname = os.path.join(problems_dir, f"problem_{name}.py")
        if os.path.exists(fname):
            print(f"  [{priority}] {name} ({pct:.1f}%) -- already exists")
            continue
        template = TEMPLATES.get(name)
        if template:
            with open(fname, "w") as fw:
                fw.write(template)
            print(f"  [{priority}] {name} ({pct:.1f}%) -> {os.path.basename(fname)}")
            generated += 1
        else:
            print(f"  [{priority}] {name} ({pct:.1f}%) -- no template, agent should create manually")
    return generated


def generate_gemm_problems(profile_dir, problems_dir):
    """STEP 2b: Generate GEMM problem files for shapes with roofline efficiency < 80%."""
    all_gemms = []
    for phase in ["decode", "prefilldecode"]:
        path = os.path.join(profile_dir, f"{phase}_analysis.json")
        if not os.path.isfile(path):
            continue
        data = json.load(open(path))
        for g in data.get("gemm_roofline", []):
            eff = g.get("eff", 100)
            if eff >= 80:
                continue
            g["phase"] = phase
            all_gemms.append(g)

    all_gemms.sort(key=lambda g: g.get("eff", 100))

    print(f"GEMM shapes with roofline efficiency < 80%: {len(all_gemms)}")
    generated = 0
    for i, g in enumerate(all_gemms, 1):
        dims = g.get("dims", "")
        eff = g.get("eff", 0)
        pct = g.get("pct", 0)
        phase = g.get("phase", "")
        bound = g.get("bound", "?")
        try:
            parsed = ast.literal_eval(dims)
            M, K = parsed[0]
            _, N = parsed[1]
        except Exception:
            print(f"  {i:2d}. eff={eff:.0f}%  SKIP (parse error)")
            continue
        fname = os.path.join(problems_dir, f"problem_gemm_{phase}_{M}x{K}x{N}.py")
        if os.path.exists(fname):
            continue
        code = f"""import torch
import torch.nn as nn

class Model(nn.Module):
    def forward(self, a, b):
        return torch.mm(a, b)

M, K, N = {M}, {K}, {N}
def get_inputs():
    return [
        torch.randn(M, K, dtype=torch.bfloat16, device="cuda"),
        torch.randn(K, N, dtype=torch.bfloat16, device="cuda"),
    ]
def get_init_inputs():
    return []
"""
        with open(fname, "w") as fw:
            fw.write(code)
        print(f"  {i:2d}. eff={eff:.0f}%  {bound:8s}  pct={pct:.1f}%  -> {os.path.basename(fname)}")
        generated += 1
    return generated


def generate_attention_problems(profile_dir, problems_dir):
    """STEP 2c: Generate attention problem files for all attention kernel types >= 1% GPU time."""
    shapes = json.load(open(os.path.join(profile_dir, "model_shapes.json")))
    num_heads = shapes.get("num_attention_heads", 16)
    num_kv_heads = shapes.get("num_key_value_heads", 4)
    head_dim = shapes.get("head_dim", 128)

    ATTN_CATEGORIES = {"SDPA_fwd", "SDPA_bwd"}
    ATTN_NAME_PATTERNS = ["attention", "flash_attn", "linear_attn", "gdn_attention", "mha", "mla"]

    all_attn = []
    for phase in ["decode", "prefilldecode"]:
        bn_path = os.path.join(profile_dir, f"{phase}_bottlenecks.json")
        if not os.path.isfile(bn_path):
            continue
        for b in json.load(open(bn_path)):
            name = b.get("name", "")
            cats = b.get("categories", "")
            pct = b.get("cuda_time_percent", 0)
            reason = b.get("reason", "")
            is_attn = (
                reason == "Attention"
                or any(cat in cats for cat in ATTN_CATEGORIES)
                or any(pat in name.lower() for pat in ATTN_NAME_PATTERNS)
            )
            if is_attn and pct >= 1.0:
                all_attn.append({"name": name, "pct": pct, "phase": phase, "categories": cats})

    seen = {}
    for a in all_attn:
        key = a["name"]
        if key not in seen or a["pct"] > seen[key]["pct"]:
            seen[key] = a
    all_attn = sorted(seen.values(), key=lambda x: -x["pct"])

    print(f"Attention kernels >= 1% GPU time: {len(all_attn)}")
    generated = 0
    for a in all_attn:
        name_lower = a["name"].lower()
        if "sdpa" in a["categories"].lower() or "scaled_dot_product" in name_lower:
            attn_type, imports, forward_code = "sdpa", "import torch.nn.functional as F", "return F.scaled_dot_product_attention(q, k, v)"
        elif "flash" in name_lower:
            attn_type, imports, forward_code = "flash", "import torch.nn.functional as F", "return F.scaled_dot_product_attention(q, k, v)"
        elif "linear_attn" in name_lower or "gdn_attention" in name_lower:
            attn_type, imports, forward_code = "linear", "", "return torch.bmm(q, k.transpose(-2, -1)) * (q.shape[-1] ** -0.5)"
        else:
            attn_type, imports, forward_code = "generic", "import torch.nn.functional as F", "return F.scaled_dot_product_attention(q, k, v)"

        safe_name = a["name"].replace("::", "_").replace(".", "_")[:40]
        fname = os.path.join(problems_dir, f"problem_attn_{a['phase']}_{safe_name}.py")
        if os.path.exists(fname):
            continue
        code = f"""import torch
import torch.nn as nn
{imports}

class Model(nn.Module):
    def forward(self, q, k, v):
        {forward_code}

batch, num_heads, num_kv_heads, head_dim, seq_len = 1, {num_heads}, {num_kv_heads}, {head_dim}, 1
def get_inputs():
    return [
        torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.bfloat16, device="cuda"),
        torch.randn(batch, num_kv_heads, seq_len, head_dim, dtype=torch.bfloat16, device="cuda"),
        torch.randn(batch, num_kv_heads, seq_len, head_dim, dtype=torch.bfloat16, device="cuda"),
    ]
def get_init_inputs():
    return []
"""
        with open(fname, "w") as fw:
            fw.write(code)
        print(f"  {a['pct']:5.1f}%  {a['name']:50s}  type={attn_type:8s}  -> {os.path.basename(fname)}")
        generated += 1
    return generated


def main():
    ap = argparse.ArgumentParser(description="Generate problem files from Phase 4 analysis")
    ap.add_argument("--profile-dir", required=True, help="Path to profile/ directory with analysis JSONs")
    ap.add_argument("--problems-dir", help="Path to problems/ directory (default: sibling of profile-dir)")
    ap.add_argument("--review", action="store_true", help="Only print analysis summary, don't generate files")
    args = ap.parse_args()

    profile_dir = args.profile_dir
    if args.review:
        review_analysis(profile_dir)
        return 0

    problems_dir = args.problems_dir or os.path.join(os.path.dirname(profile_dir.rstrip("/")), "problems")
    os.makedirs(problems_dir, exist_ok=True)

    print(f"Profile dir: {profile_dir}")
    print(f"Problems dir: {problems_dir}")
    print()

    total = 0
    n = generate_fusion_problems(profile_dir, problems_dir)
    total += n or 0
    print()
    n = generate_gemm_problems(profile_dir, problems_dir)
    total += n or 0
    print()
    n = generate_attention_problems(profile_dir, problems_dir)
    total += n or 0

    print(f"\nTotal problem files generated: {total}")
    print("Files:")
    for f in sorted(os.listdir(problems_dir)):
        if f.endswith(".py"):
            print(f"  {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
