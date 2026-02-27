#!/usr/bin/env python3
"""
Kernel Shape Analyzer for the model-optimize pipeline.

Produces a per-shape GPU time breakdown for each operator category by
correlating GPU kernel events with CPU-side operator events from a torch
profiler trace.

Design
------
Classification and shapes come from **CPU ops** (always accurate), while
timing comes from **GPU kernel durations**.  When correlation fails the
kernel is classified by its GPU name and reported without a shape.

The torch profiler links events via ``External id`` / ``correlation``:

  CPU op  ──(External id N)──▶  GPU kernel  (GPU kernel args contain External id)
  CPU op  ──(External id N)──▶  cuda_runtime  ──(correlation M)──▶  GPU kernel

Two correlation paths are tried for every GPU kernel:
  1. **Direct** — GPU kernel ``External id`` (in args) matches a CPU op ``External id``
  2. **Bridge** — GPU kernel ``correlation`` matches a ``cuda_runtime`` event
     whose ``External id`` points to the real CPU compute op

When neither path yields a compute op with real tensor shapes the GPU kernel
is classified by its own name (regex rules) and reported without a shape.

⚠  ``--enforce-eager`` is required on the vLLM serve command during trace
collection.  With CUDA Graphs most correlation IDs are stale / incorrect.

Outputs
-------
- ``kernel_shape_analysis.json`` — structured per-category, per-shape data
- ``kernel_shape_analysis.csv``  — flat CSV for inspection

Usage::

    python analyze_kernel_shapes.py -i trace.pt.trace.json.gz
    python analyze_kernel_shapes.py -i trace.pt.trace.json.gz -o <output_dir>

Part of the model-optimize pipeline.  Can be used standalone.
"""
import argparse
import csv
import gzip
import json
import os
import re
import sys
from collections import defaultdict
from statistics import mean, median
from typing import Any, Dict, List, Optional, Tuple
import bisect

# ── CPU op name → category ──────────────────────────────────────────────

_CPU_OP_CATEGORY: Dict[str, str] = {}
for _names, _cat in [
    (["aten::mm", "aten::addmm", "aten::matmul", "aten::bmm",
      "aten::linear"], "GEMM"),
    (["vllm::rocm_unquantized_gemm", "vllm::cutlass_gemm",
      "vllm::gptq_gemm", "vllm::awq_gemm"], "GEMM"),
    (["vllm::unified_attention_with_output",
      "vllm::unified_attention", "vllm::paged_attention"], "Attention"),
    (["aten::scaled_dot_product_attention"], "Attention"),
    (["aten::_softmax", "aten::softmax"], "Softmax"),
    (["aten::layer_norm", "aten::group_norm", "aten::rms_norm",
      "vllm::rms_norm"], "Norm"),
    (["aten::silu", "aten::gelu", "aten::relu", "aten::sigmoid",
      "aten::tanh", "aten::swish"], "Activation"),
    (["aten::mul", "aten::add", "aten::sub", "aten::div"], "Elementwise"),
    (["aten::embedding", "aten::index_select"], "Embedding"),
    (["vllm::rotary_embedding", "vllm::apply_rotary_emb"], "RoPE"),
    (["aten::topk", "aten::argmax", "aten::sum",
      "aten::cumsum", "aten::amax"], "Reduce"),
    (["aten::multinomial", "vllm::sample"], "Sampling"),
]:
    for _n in _names:
        _CPU_OP_CATEGORY[_n] = _cat

# Prefixes that always count as compute ops (vllm custom ops)
_COMPUTE_PREFIXES = ("vllm::", "torch_custom::")

# ── GPU kernel name → category (fallback when no CPU op match) ───────

_GPU_KERNEL_RULES: List[Tuple[str, re.Pattern]] = [
    ("GEMM",        re.compile(r"Cijk_|gemm|hipblas|rocblas|cublas|cutlass|mm\b", re.I)),
    ("Attention",   re.compile(r"attn|flash|sdpa|mha|fmha|paged", re.I)),
    ("Norm",        re.compile(r"norm|rms_norm|layer_norm|rmsnorm|layernorm", re.I)),
    ("Activation",  re.compile(r"silu|gelu|relu|swish|tanh|sigmoid", re.I)),
    ("Softmax",     re.compile(r"softmax", re.I)),
    ("Elementwise", re.compile(r"elementwise|vectorized|pointwise|add_kernel|mul_kernel", re.I)),
    ("Embedding",   re.compile(r"embed|gather|index_select", re.I)),
    ("RoPE",        re.compile(r"rotary|rope", re.I)),
    ("Copy/Mem",    re.compile(r"copy|memcpy|memset|\bcat\b|concat|reshape|permute|transpose", re.I)),
    ("Reduce",      re.compile(r"reduce|\bsum\b|argmax|topk", re.I)),
    ("Sampling",    re.compile(r"sample|multinomial", re.I)),
]


def _classify_gpu_kernel(name: str) -> str:
    for cat, pat in _GPU_KERNEL_RULES:
        if pat.search(name):
            return cat
    return "Other"


def _classify_cpu_op(name: str) -> Optional[str]:
    cat = _CPU_OP_CATEGORY.get(name)
    if cat:
        return cat
    for prefix in _COMPUTE_PREFIXES:
        if name.startswith(prefix):
            return "Other"
    return None


# ── Shape helpers ────────────────────────────────────────────────────────

def _shape_signature(dims: Any) -> str:
    """Convert Input Dims to ``[M,K]x[K,N]`` style, dropping empty/scalar args."""
    if not dims:
        return ""
    parts = []
    for d in dims:
        if isinstance(d, list) and len(d) > 0:
            parts.append("[" + ",".join(str(x) for x in d) + "]")
    return "x".join(parts) if parts else ""


def _has_real_shapes(dims: Any) -> bool:
    if not dims:
        return False
    return any(isinstance(d, list) and len(d) > 0 for d in dims)


# ── Trace I/O ────────────────────────────────────────────────────────────

def _open(path: str):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "r")


def load_trace(path: str) -> List[Dict[str, Any]]:
    with _open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        events = data.get("traceEvents") or data.get("events") or []
    elif isinstance(data, list):
        events = data
    else:
        events = []
    if not events:
        raise ValueError("Trace JSON contains no events")
    return events


# ── Index building ───────────────────────────────────────────────────────

def build_indices(events: List[Dict[str, Any]]):
    """Parse events into lookup structures.

    Returns
    -------
    cpu_ops : dict[int, dict]
        ext_id → {name, category, shape, dims, ts, dur}
    gpu_kernels : list[dict]
        [{name, dur, ts, correlation}, ...]
    runtime_bridge : dict[int, int]
        gpu_correlation → cpu_ext_id  (via cuda_runtime events)
    """
    cpu_ops: Dict[int, Dict] = {}
    gpu_kernels: List[Dict] = []
    runtime_bridge: Dict[int, int] = {}

    for ev in events:
        cat_field = ev.get("cat", "")

        # ── GPU kernel ──
        if cat_field == "kernel":
            dur = ev.get("dur")
            if dur is not None and dur > 0:
                args = ev.get("args") or {}
                gpu_kernels.append({
                    "name": ev.get("name", ""),
                    "dur": dur,
                    "ts": ev.get("ts", 0),
                    "correlation": args.get("correlation"),
                    "ext_id": args.get("External id"),
                })
            continue

        # ── cuda_runtime (bridge) ──
        if cat_field == "cuda_runtime":
            args = ev.get("args") or {}
            ext = args.get("External id")
            corr = args.get("correlation")
            if ext is not None and corr is not None:
                runtime_bridge[corr] = ext
            continue

        # ── CPU op ──
        if cat_field == "cpu_op":
            args = ev.get("args") or {}
            ext_id = args.get("External id")
            if ext_id is None:
                continue
            name = ev.get("name", "")
            dims = args.get("Input Dims") or args.get("input_dims")
            cpu_ops[ext_id] = {
                "name": name,
                "category": _classify_cpu_op(name),
                "shape": _shape_signature(dims) if _has_real_shapes(dims) else "",
                "dims": dims,
                "ts": ev.get("ts", 0),
                "dur": ev.get("dur", 0),
            }

    return cpu_ops, gpu_kernels, runtime_bridge


# ── Correlation ──────────────────────────────────────────────────────────

class _CategoryTimeline:
    """Pre-computed per-category CPU-op timeline for fast temporal lookups."""

    def __init__(self, cpu_ops: Dict[int, Dict]):
        raw: Dict[str, List[Tuple[float, float, str]]] = defaultdict(list)
        for info in cpu_ops.values():
            cat = info.get("category")
            shape = info.get("shape")
            if not cat or not shape:
                continue
            ts = float(info.get("ts", 0))
            dur = float(info.get("dur", 0))
            raw[cat].append((ts, ts + max(dur, 0.0), shape))
        self._entries: Dict[str, List[Tuple[float, float, str]]] = {}
        self._starts: Dict[str, List[float]] = {}
        for cat, items in raw.items():
            items.sort(key=lambda x: x[0])
            self._entries[cat] = items
            self._starts[cat] = [x[0] for x in items]

    def lookup(
        self,
        category: str,
        kernel_ts: float,
        max_prev_gap_us: float = 2000.0,
    ) -> str:
        """Infer a shape by matching a kernel timestamp to same-category CPU-op timeline.

        Strategy:
          1) Prefer enclosing interval (start <= ts <= end).
          2) Else use nearest previous interval if gap <= max_prev_gap_us.
        """
        entries = self._entries.get(category)
        if not entries:
            return ""

        starts = self._starts[category]
        idx = bisect.bisect_right(starts, kernel_ts) - 1

        if 0 <= idx < len(entries):
            s, e, shape = entries[idx]
            if s <= kernel_ts <= e:
                return shape
            if 0.0 <= (kernel_ts - e) <= max_prev_gap_us:
                return shape

        return ""


def correlate(gpu_kernels, cpu_ops, runtime_bridge):
    """Match each GPU kernel to a CPU compute op.

    Returns list of (category, shape, gpu_dur) tuples.
    Also returns match statistics.
    """
    results: List[Tuple[str, str, float]] = []
    stats = {
        "direct": 0, "bridge": 0, "temporal": 0,
        "kernel_name": 0, "total": len(gpu_kernels),
        "correlated_cat_only": 0,
    }
    timeline = _CategoryTimeline(cpu_ops)

    for k in gpu_kernels:
        corr = k["correlation"]
        ext_id = k.get("ext_id")
        category = None
        shape = ""

        # ── Path 1: GPU kernel External id → CPU op (most reliable) ──
        if ext_id is not None and ext_id in cpu_ops:
            info = cpu_ops[ext_id]
            if info["category"]:
                category = info["category"]
                shape = info["shape"] or ""
                if shape:
                    stats["direct"] += 1
                else:
                    stats["correlated_cat_only"] += 1

        # ── Path 2: cuda_runtime bridge (correlation → runtime ext_id → CPU op) ──
        if not shape and corr is not None and corr in runtime_bridge:
            bridge_ext = runtime_bridge[corr]
            if bridge_ext in cpu_ops:
                info = cpu_ops[bridge_ext]
                if info["category"] and info["shape"]:
                    if category is None:
                        category = info["category"]
                    shape = info["shape"]
                    stats["bridge"] += 1

        # ── Fallback: classify by GPU kernel name, no shape ──
        if category is None:
            category = _classify_gpu_kernel(k["name"])
            shape = ""

        # ── Fallback: same-category temporal inference ──
        if not shape and category not in ("Other", "Copy/Mem"):
            inferred = timeline.lookup(
                category=category,
                kernel_ts=float(k.get("ts", 0)),
            )
            if inferred:
                shape = inferred
                stats["temporal"] += 1

        if not shape:
            stats["kernel_name"] += 1

        results.append((category, shape, k["dur"]))

    return results, stats


# ── Aggregation ──────────────────────────────────────────────────────────

def aggregate(
    results: List[Tuple[str, str, float]],
) -> Tuple[List[Dict], List[Dict], float]:
    """Group (category, shape, dur) triples into report structures.

    Returns
    -------
    per_category : list[dict]
    flat_rows    : list[dict]
    total_gpu_us : float
    """
    total_gpu_us = sum(dur for _, _, dur in results)
    if total_gpu_us == 0:
        return [], [], 0.0

    # (category, shape, kernel_name_placeholder) → [durs]
    bucket: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for cat, shape, dur in results:
        bucket[(cat, shape or "(unattributed)")].append(dur)

    # ── Flat rows ──
    flat_rows = []
    for (cat, shape), durs in bucket.items():
        flat_rows.append({
            "category": cat,
            "shape": shape,
            "count": len(durs),
            "total_us": sum(durs),
            "avg_us": mean(durs),
            "median_us": median(durs),
            "min_us": min(durs),
            "max_us": max(durs),
            "pct": sum(durs) / total_gpu_us * 100,
        })
    flat_rows.sort(key=lambda r: r["total_us"], reverse=True)

    # ── Per-category with nested shapes ──
    cat_shape: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for (cat, shape), durs in bucket.items():
        cat_shape[cat][shape].extend(durs)

    per_category = []
    for cat in sorted(cat_shape, key=lambda c: sum(sum(d) for d in cat_shape[c].values()), reverse=True):
        shapes_map = cat_shape[cat]
        cat_total = sum(sum(d) for d in shapes_map.values())

        attributed_total = sum(sum(d) for s, d in shapes_map.items() if s != "(unattributed)")

        shapes_list = []
        for sig in sorted(shapes_map, key=lambda s: sum(shapes_map[s]), reverse=True):
            durs = shapes_map[sig]
            shapes_list.append({
                "shape": sig,
                "count": len(durs),
                "total_us": sum(durs),
                "avg_us": mean(durs),
                "median_us": median(durs),
                "pct_of_category": sum(durs) / cat_total * 100 if cat_total else 0,
                "pct_of_total": sum(durs) / total_gpu_us * 100,
            })

        per_category.append({
            "category": cat,
            "total_us": cat_total,
            "pct": cat_total / total_gpu_us * 100,
            "num_shapes": len(shapes_list),
            "attributed_pct": attributed_total / cat_total * 100 if cat_total else 0,
            "shapes": shapes_list,
        })

    return per_category, flat_rows, total_gpu_us


def keep_top_hot_shapes(
    per_category: List[Dict],
    flat_rows: List[Dict],
    top_n: int,
) -> Tuple[List[Dict], List[Dict]]:
    """Keep only top-N hottest (category, shape) rows across all categories."""
    if top_n <= 0:
        return [], []

    top_rows = flat_rows[:top_n]
    keep_keys = {(r["category"], r["shape"]) for r in top_rows}

    trimmed_categories = []
    for cat in per_category:
        filtered_shapes = [
            s for s in cat["shapes"]
            if (cat["category"], s["shape"]) in keep_keys
        ]
        if not filtered_shapes:
            continue
        trimmed_categories.append({
            **cat,
            "num_shapes": len(filtered_shapes),
            "shapes": filtered_shapes,
        })

    return trimmed_categories, top_rows


# ── Output ───────────────────────────────────────────────────────────────

def write_json(path: str, per_category: List[Dict], total_gpu_us: float):
    out = {
        "total_gpu_time_us": total_gpu_us,
        "total_gpu_time_ms": total_gpu_us / 1000,
        "categories": per_category,
    }
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {path}")


def write_csv(path: str, flat_rows: List[Dict]):
    fields = ["category", "shape", "count", "total_us",
              "avg_us", "median_us", "min_us", "max_us", "pct"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(flat_rows)
    print(f"Wrote {path}")


def print_summary(per_category: List[Dict], total_gpu_us: float,
                  stats: Dict[str, int]):
    shaped = stats["direct"] + stats["bridge"] + stats["temporal"]
    cat_only = stats.get("correlated_cat_only", 0)
    correlated = shaped + cat_only
    total = stats["total"]
    corr_pct = (correlated / total * 100) if total else 0.0
    shape_pct = (shaped / total * 100) if total else 0.0
    print(f"\n{'='*90}")
    print(f"  KERNEL SHAPE ANALYSIS — Total GPU time: {total_gpu_us/1000:.2f}ms")
    print(f"  Correlated {correlated}/{total} GPU kernels to CPU ops "
          f"({corr_pct:.1f}%)")
    print(f"  Shape-attributed {shaped}/{total} "
          f"({shape_pct:.1f}%)"
          f"  [direct={stats['direct']}, bridge={stats['bridge']}, temporal={stats['temporal']}, "
          f"cat-only={cat_only}, kernel-name-only={stats['kernel_name']}]")
    if correlated > 0 and shaped == 0:
        print(f"\n  ⚠  WARNING: Trace has no 'Input Dims' data. Re-collect with "
              f"record_shapes=True in profiler config.")
    print(f"{'='*90}")

    for ci in per_category:
        cat = ci["category"]
        attr = ci["attributed_pct"]
        print(f"\n  {cat} — {ci['pct']:.1f}% of total "
              f"({ci['total_us']/1000:.2f}ms, "
              f"{ci['num_shapes']} shapes, {attr:.0f}% attributed)")
        print(f"  {'─'*86}")
        print(f"    {'Shape':<50} {'%Total':>7} {'%InCat':>7} "
              f"{'Time(ms)':>9} {'Count':>6} {'Avg(us)':>8}")

        for s in ci["shapes"][:15]:
            label = s["shape"][:50]
            print(f"    {label:<50} {s['pct_of_total']:6.1f}% "
                  f"{s['pct_of_category']:6.1f}% "
                  f"{s['total_us']/1000:8.2f} {s['count']:6d} "
                  f"{s['avg_us']:8.1f}")
        rest = ci["num_shapes"] - 15
        if rest > 0:
            print(f"    ... and {rest} more shapes")

    print(f"\n{'='*90}\n")


# ── Main ─────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Per-shape GPU time breakdown for each operator category")
    ap.add_argument("-i", "--input", required=True,
                    help="Torch profiler trace (.json or .json.gz)")
    ap.add_argument("-o", "--output-dir",
                    help="Output directory (default: same as input file)")
    ap.add_argument("--json-out", help="Override JSON output path")
    ap.add_argument("--csv-out", help="Override CSV output path")
    ap.add_argument("--top-n", type=int, default=20,
                    help="Keep only top-N hottest shapes in outputs (default: 20)")
    args = ap.parse_args()

    print(f"Loading trace: {args.input}")
    events = load_trace(args.input)
    print(f"Loaded {len(events)} events")

    cpu_ops, gpu_kernels, bridge = build_indices(events)
    print(f"CPU ops: {len(cpu_ops)}, GPU kernels: {len(gpu_kernels)}, "
          f"runtime bridge entries: {len(bridge)}")

    results, stats = correlate(gpu_kernels, cpu_ops, bridge)
    per_category, flat_rows, total_gpu_us = aggregate(results)

    if total_gpu_us == 0:
        print("ERROR: No GPU kernel events with duration found.")
        return 1

    per_category, flat_rows = keep_top_hot_shapes(
        per_category=per_category,
        flat_rows=flat_rows,
        top_n=args.top_n,
    )

    print(f"Keeping top {args.top_n} hottest shapes in outputs.")
    print_summary(per_category, total_gpu_us, stats)

    out_dir = args.output_dir or os.path.dirname(args.input) or "."
    os.makedirs(out_dir, exist_ok=True)
    json_path = args.json_out or os.path.join(out_dir, "kernel_shape_analysis.json")
    csv_path = args.csv_out or os.path.join(out_dir, "kernel_shape_analysis.csv")
    write_json(json_path, per_category, total_gpu_us)
    write_csv(csv_path, flat_rows)

    return 0


if __name__ == "__main__":
    sys.exit(main())
