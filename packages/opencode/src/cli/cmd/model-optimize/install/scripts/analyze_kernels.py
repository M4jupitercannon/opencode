#!/usr/bin/env python3
"""
Kernel & Trace Analyzer for the model-optimize pipeline.

Validates the profiling trace, splits it into per-phase traces (prefill-decode
vs decode-only) using TraceLens trace splitting, then runs TraceLens standalone
performance analysis on each phase.

Workflow
--------
1. Auto-select worker trace (rank-0, skip async_llm frontend traces)
2. Validate trace has shape data  (step 2b — mandatory gate)
3. Split trace into phases via TraceLens split_vllm_trace_annotation.py
4. Run TraceLens generate_perf_report_pytorch_vllm.py on each phase trace
5. Produce a summary JSON consumed by downstream pipeline steps

Outputs
-------
- ``<output_dir>/phase_traces/``          — split trace files
- ``<output_dir>/prefilldecode_report/``  — TraceLens CSVs for prefill-decode
- ``<output_dir>/decode_report/``         — TraceLens CSVs for decode-only
- ``<output_dir>/analysis_summary.json``  — machine-readable summary

Usage::

    # Validate only (step 2b gate — no TraceLens required)
    python analyze_kernels.py -i trace_dir/ --validate-only

    # Full analysis (requires TraceLens, auto-cloned if missing)
    python analyze_kernels.py -i trace_dir/ -o analysis_output/

    # Specify TraceLens location explicitly
    python analyze_kernels.py -i trace.json.gz -o out/ \\
        --tracelens-dir /path/to/TraceLens

Part of the model-optimize pipeline.  Can be used standalone.
"""
import argparse
import csv
import glob
import gzip
import json
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional


# ── Trace I/O (lightweight, no external deps) ───────────────────────────

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


# ── Trace auto-selection ─────────────────────────────────────────────────

def find_worker_trace(path: str) -> str:
    """If path is a directory, auto-select the worker trace (rank-0), rejecting async_llm.

    vLLM writes two trace files per profiling session:
      - *async_llm* — frontend-only (CPU python_function events, NO GPU kernels)
      - *rank-0*    — worker trace (CPU ops + CUDA kernels with shapes)

    If path is a file, returns it as-is (with a warning if it looks like async_llm).
    """
    if os.path.isfile(path):
        if "async_llm" in os.path.basename(path):
            print(f"WARNING: Input file appears to be an async_llm frontend trace.")
            print(f"  Frontend traces contain only Python function calls — no GPU kernels.")
            print(f"  For shape analysis, use the *rank-0* worker trace instead.")
        return path

    if not os.path.isdir(path):
        return path

    traces = sorted(
        [os.path.join(path, f) for f in os.listdir(path)
         if f.endswith(".json") or f.endswith(".json.gz")],
        key=os.path.getmtime, reverse=True,
    )
    if not traces:
        print(f"ERROR: No trace files found in directory: {path}")
        return ""

    for t in traces:
        bn = os.path.basename(t)
        if "rank" in bn and "async_llm" not in bn:
            print(f"Auto-selected worker trace: {bn}")
            return t

    for t in traces:
        if "async_llm" not in os.path.basename(t):
            print(f"Auto-selected trace (no rank marker): {os.path.basename(t)}")
            return t

    print(f"WARNING: Only async_llm frontend traces found in {path}")
    return traces[0]


# ── Trace Validation (step 2b) ───────────────────────────────────────────

def validate_trace_for_shapes(events: List[Dict[str, Any]]) -> bool:
    """Upfront validation that the trace has the data needed for analysis.

    Checks for:
      - GPU kernel events (cat=kernel)
      - CPU op events (cat=cpu_op)
      - Input Dims on CPU ops (torch_profiler_record_shapes was enabled)
      - External id on GPU kernels (--enforce-eager was used)

    Returns True if valid, False if analysis would be useless.

    Uses a single pass over events to avoid repeated full-list scans on traces
    with 100M+ events.
    """
    n_cpu_ops = 0
    n_with_shapes = 0
    n_gpu_kernels = 0
    n_with_ext_id = 0
    n_frontend = 0

    for e in events:
        cat = e.get("cat")
        if cat == "cpu_op":
            n_cpu_ops += 1
            if e.get("args", {}).get("Input Dims"):
                n_with_shapes += 1
        elif cat == "kernel":
            n_gpu_kernels += 1
            if e.get("args", {}).get("External id"):
                n_with_ext_id += 1
        elif cat == "python_function":
            n_frontend += 1

    print(f"\n  Trace validation:")
    print(f"    CPU ops (cpu_op):     {n_cpu_ops}")
    print(f"    CPU ops with shapes:  {n_with_shapes}")
    print(f"    GPU kernels:          {n_gpu_kernels}")
    print(f"    GPU kernels w/ ext_id:{n_with_ext_id}")

    problems = []
    if n_gpu_kernels == 0:
        if n_frontend > 0:
            problems.append(
                f"This is an async_llm frontend trace ({n_frontend} python_function events, 0 GPU kernels). "
                f"Use the *rank-0* worker trace instead."
            )
        else:
            problems.append("No GPU kernel events found.")
    if n_cpu_ops == 0:
        problems.append("No cpu_op events — --enforce-eager was likely missing on vllm serve.")
    if n_cpu_ops > 0 and n_with_shapes == 0:
        problems.append(
            "CPU ops exist but have no Input Dims — "
            "torch_profiler_record_shapes was likely not set in --profiler-config."
        )
    if n_gpu_kernels > 0 and n_with_ext_id == 0:
        problems.append(
            "GPU kernels exist but have no External id — "
            "--enforce-eager was likely missing (CUDA Graphs break correlation)."
        )

    if problems:
        print(f"\n  TRACE VALIDATION FAILED:")
        for p in problems:
            print(f"    - {p}")
        print(f"\n  Shape analysis requires ALL of:")
        print(f"    1. --enforce-eager on the vllm serve command")
        print(f"    2. --profiler-config JSON with torch_profiler_record_shapes: true, ignore_frontend: true")
        print(f"    3. /start_profile API before requests, /stop_profile after")
        print(f"    4. Select the *rank-0* worker trace, not the *async_llm* trace")
        return False

    pct = n_with_shapes / n_cpu_ops * 100 if n_cpu_ops else 0
    print(f"    PASSED — {pct:.0f}% of CPU ops have shapes, {n_with_ext_id} GPU kernels have External id")
    return True


# ── Bundled split script (shipped alongside this file) ───────────────────

_BUNDLED_SPLIT_SCRIPT = os.path.join(os.path.dirname(__file__), "split_vllm_trace_annotation.py")

# ── TraceLens Discovery ──────────────────────────────────────────────────

_TRACELENS_GIT_URL = "https://github.com/AMD-AGI/TraceLens.git"

_TRACELENS_SEARCH_PATHS = [
    "/TraceLens-internal",
    "/TraceLens",
    os.path.expanduser("~/TraceLens-internal"),
    os.path.expanduser("~/TraceLens"),
    os.path.join(os.path.dirname(__file__), "..", "TraceLens-internal"),
    os.path.join(os.path.dirname(__file__), "..", "TraceLens"),
]


def _is_valid_tracelens(d: str) -> bool:
    """Check if a directory contains the required TraceLens report script.

    The split script is bundled with this package and no longer needs to
    live inside the TraceLens tree.
    """
    report_script = os.path.join(
        d, "TraceLens", "Reporting", "generate_perf_report_pytorch_vllm.py"
    )
    return os.path.isfile(report_script)


def _clone_tracelens(target_dir: str) -> Optional[str]:
    """Clone TraceLens from GitHub into target_dir. Returns the path on success."""
    print(f"  TraceLens not found locally. Cloning from {_TRACELENS_GIT_URL} ...")
    try:
        result = subprocess.run(
            ["git", "clone", "--depth", "1", _TRACELENS_GIT_URL, target_dir],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            print(f"  git clone failed: {result.stderr.strip()}")
            return None
        if _is_valid_tracelens(target_dir):
            print(f"  Cloned TraceLens to: {target_dir}")
            return target_dir
        print(f"  Cloned but required scripts not found in {target_dir}")
        return None
    except FileNotFoundError:
        print("  git is not installed — cannot clone TraceLens")
        return None
    except subprocess.TimeoutExpired:
        print("  git clone timed out")
        return None


def find_tracelens(explicit_dir: Optional[str] = None) -> Optional[str]:
    """Locate the TraceLens installation directory.

    Search order:
      1. Explicit --tracelens-dir argument
      2. TRACELENS_DIR environment variable
      3. Common installation paths
      4. Auto-clone from GitHub as a last resort
    """
    candidates = []
    if explicit_dir:
        candidates.append(explicit_dir)
    env_dir = os.environ.get("TRACELENS_DIR")
    if env_dir:
        candidates.append(env_dir)
    candidates.extend(_TRACELENS_SEARCH_PATHS)

    for d in candidates:
        if _is_valid_tracelens(d):
            print(f"  Found TraceLens at: {d}")
            return d

    # Not found — try to clone into the first writable candidate location
    clone_targets = ["/TraceLens", os.path.expanduser("~/TraceLens")]
    if explicit_dir:
        clone_targets.insert(0, explicit_dir)

    for target in clone_targets:
        if os.path.exists(target):
            continue
        try:
            os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
        except OSError:
            continue
        result = _clone_tracelens(target)
        if result:
            return result

    return None


# ── Phase 1: Trace Splitting ─────────────────────────────────────────────

def split_trace(
    trace_path: str,
    output_dir: str,
    tracelens_dir: str,
    find_steady_state: bool = True,
) -> List[Dict[str, Any]]:
    """Split the trace into phase-specific traces using the bundled splitter.

    Uses ``--store-single-iteration`` so every iteration is saved as its own
    file.  When *find_steady_state* is True (default) the splitter also
    identifies the steady-state region and produces combined + phase-specific
    (prefill-decode / decode-only) traces.

    Returns a list of execution_details entries with paths to the generated
    files, or an empty list on failure.
    """
    split_script = _BUNDLED_SPLIT_SCRIPT
    phase_dir = os.path.join(output_dir, "phase_traces")
    os.makedirs(phase_dir, exist_ok=True)

    cmd = [
        sys.executable, split_script,
        trace_path,
        "-o", phase_dir,
        "--store-single-iteration",
    ]
    if find_steady_state:
        cmd.append("--find-steady-state")

    print(f"\n  Splitting trace into phases...")
    print(f"    Command: {' '.join(cmd)}")

    env = os.environ.copy()
    env["PYTHONPATH"] = tracelens_dir + os.pathsep + env.get("PYTHONPATH", "")

    result = subprocess.run(
        cmd, capture_output=True, text=True, env=env, timeout=1800,
    )

    if result.returncode != 0:
        print(f"\n  Trace splitting FAILED (exit code {result.returncode}):")
        print(result.stderr[-2000:] if result.stderr else "(no stderr)")
        print(result.stdout[-2000:] if result.stdout else "(no stdout)")
        return []

    print(result.stdout[-3000:] if result.stdout else "")

    details_path = os.path.join(phase_dir, "execution_details.json")
    if os.path.isfile(details_path):
        with open(details_path) as f:
            return json.load(f)

    print("  WARNING: execution_details.json not generated")
    return []


# ── Phase 2: Standalone Analysis ─────────────────────────────────────────

def run_tracelens_analysis(
    trace_path: str,
    output_dir: str,
    tracelens_dir: str,
    label: str = "",
    enable_kernel_summary: bool = True,
) -> bool:
    """Run TraceLens standalone performance analysis on a single trace file.

    Returns True on success.
    """
    report_script = os.path.join(
        tracelens_dir, "TraceLens", "Reporting",
        "generate_perf_report_pytorch_vllm.py",
    )
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        sys.executable, "-u", report_script,
        "--profile_json_path", trace_path,
        "--output_csvs_dir", output_dir,
    ]
    if enable_kernel_summary:
        cmd.append("--enable_kernel_summary")

    tag = f" [{label}]" if label else ""
    print(f"\n  Running TraceLens analysis{tag}...")
    print(f"    Trace: {os.path.basename(trace_path)}")
    print(f"    Output: {output_dir}")

    env = os.environ.copy()
    env["PYTHONPATH"] = tracelens_dir + os.pathsep + env.get("PYTHONPATH", "")

    result = subprocess.run(
        cmd, capture_output=True, text=True, env=env, timeout=3600,
    )

    if result.returncode != 0:
        print(f"    FAILED (exit code {result.returncode}):")
        stderr_tail = result.stderr[-2000:] if result.stderr else ""
        stdout_tail = result.stdout[-2000:] if result.stdout else ""
        print(stderr_tail or stdout_tail)
        return False

    csv_files = glob.glob(os.path.join(output_dir, "*.csv"))
    print(f"    Generated {len(csv_files)} CSV files")
    for f in sorted(csv_files):
        print(f"      {os.path.basename(f)}")
    return True


# ── Phase 3: Result Summary ─────────────────────────────────────────────

def _read_csv_safe(path: str) -> List[Dict]:
    if not os.path.isfile(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _parse_float(val: str, default: float = 0.0) -> float:
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def build_summary(
    output_dir: str,
    execution_details: Any,
    phase_reports: Dict[str, str],
) -> Dict:
    """Build a machine-readable analysis_summary.json from TraceLens outputs.

    Reads the unified_perf_summary.csv and ops_summary_by_category.csv from
    each phase report directory and consolidates them.
    """
    summary: Dict[str, Any] = {
        "phases": {},
        "execution_details": execution_details,
    }

    for phase_label, report_dir in phase_reports.items():
        phase_data: Dict[str, Any] = {"report_dir": report_dir}

        # GPU timeline
        timeline_path = os.path.join(report_dir, "gpu_timeline.csv")
        timeline_rows = _read_csv_safe(timeline_path)
        timeline = {}
        for row in timeline_rows:
            timeline[row.get("type", "")] = {
                "time_ms": _parse_float(row.get("time ms", "0")),
                "percent": _parse_float(row.get("percent", "0")),
            }
        phase_data["gpu_timeline"] = timeline

        # Category breakdown
        cat_path = os.path.join(report_dir, "ops_summary_by_category.csv")
        cat_rows = _read_csv_safe(cat_path)
        categories = []
        for row in cat_rows:
            categories.append({
                "category": row.get("op category", ""),
                "count": int(_parse_float(row.get("Count", "0"))),
                "total_kernel_time_ms": _parse_float(
                    row.get("total_direct_kernel_time_ms", "0")
                ),
                "percentage": _parse_float(row.get("Percentage (%)", "0")),
            })
        phase_data["categories"] = categories

        # Op summary (top ops by time)
        ops_path = os.path.join(report_dir, "ops_summary.csv")
        ops_rows = _read_csv_safe(ops_path)
        top_ops = []
        for row in ops_rows[:20]:
            top_ops.append({
                "name": row.get("name", ""),
                "count": int(_parse_float(row.get("Count", "0"))),
                "total_kernel_time_ms": _parse_float(
                    row.get("total_direct_kernel_time_ms", "0")
                ),
                "percentage": _parse_float(row.get("Percentage (%)", "0")),
            })
        phase_data["top_ops"] = top_ops

        # Unified perf summary (roofline data — top entries by time %)
        unified_path = os.path.join(report_dir, "unified_perf_summary.csv")
        unified_rows = _read_csv_safe(unified_path)
        _ROOFLINE_FLOAT_FIELDS = {
            "GFLOPS": "gflops",
            "Data Moved (MB)": "data_moved_mb",
            "FLOPS/Byte": "flops_per_byte",
            "TFLOPS/s_mean": "tflops_per_s_mean",
            "TB/s_mean": "tb_per_s_mean",
        }
        roofline_entries = []
        for row in unified_rows[:30]:
            entry = {
                "name": row.get("name", ""),
                "category": row.get("op category", ""),
                "input_dims": row.get("Input Dims", ""),
                "count": int(_parse_float(row.get("operation_count", "0"))),
                "percentage": _parse_float(row.get("Percentage (%)", "0")),
                "has_perf_model": row.get("has_perf_model", "").lower() == "true",
            }
            for csv_col, key in _ROOFLINE_FLOAT_FIELDS.items():
                val = row.get(csv_col, "")
                if val:
                    entry[key] = _parse_float(val)
            spec = row.get("Compute Spec", "")
            if spec:
                entry["compute_spec"] = spec
            roofline_entries.append(entry)
        phase_data["unified_perf_summary"] = roofline_entries

        summary["phases"][phase_label] = phase_data

    return summary


def print_phase_summary(summary: Dict):
    """Pretty-print a comparison of phase results."""
    phases = summary.get("phases", {})
    if not phases:
        print("  No phase data available.")
        return

    print(f"\n{'='*90}")
    print(f"  TRACELENS PERFORMANCE ANALYSIS SUMMARY")
    print(f"{'='*90}")

    for phase_label, phase_data in phases.items():
        timeline = phase_data.get("gpu_timeline", {})
        total_ms = timeline.get("total_time", {}).get("time_ms", 0)
        busy_pct = timeline.get("busy_time", {}).get("percent", 0)
        idle_pct = timeline.get("idle_time", {}).get("percent", 0)

        print(f"\n  Phase: {phase_label}")
        print(f"  {'─'*86}")
        print(f"    Total GPU time: {total_ms:.2f} ms")
        print(f"    GPU busy: {busy_pct:.1f}%  |  GPU idle: {idle_pct:.1f}%")

        categories = phase_data.get("categories", [])
        if categories:
            print(f"\n    {'Category':<20} {'Time(ms)':>10} {'%':>8} {'Count':>8}")
            print(f"    {'─'*50}")
            for cat in categories:
                print(
                    f"    {cat['category']:<20} "
                    f"{cat['total_kernel_time_ms']:>10.2f} "
                    f"{cat['percentage']:>7.1f}% "
                    f"{cat['count']:>8}"
                )

        top_ops = phase_data.get("top_ops", [])
        if top_ops:
            print(f"\n    Top Operators:")
            print(f"    {'Op Name':<45} {'Time(ms)':>10} {'%':>8} {'Count':>8}")
            print(f"    {'─'*75}")
            for op in top_ops[:10]:
                name = op["name"][:45]
                print(
                    f"    {name:<45} "
                    f"{op['total_kernel_time_ms']:>10.2f} "
                    f"{op['percentage']:>7.1f}% "
                    f"{op['count']:>8}"
                )

    print(f"\n{'='*90}\n")


# ── Main ─────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Trace validation, splitting, and TraceLens performance analysis",
    )
    ap.add_argument(
        "-i", "--input", required=True,
        help="Torch profiler trace (.json/.json.gz) or trace directory",
    )
    ap.add_argument(
        "-o", "--output-dir",
        help="Output directory for analysis results (default: same as input)",
    )
    ap.add_argument(
        "--validate-only", action="store_true",
        help="Only validate trace (step 2b gate) — no TraceLens needed",
    )
    ap.add_argument(
        "--tracelens-dir",
        help="Path to TraceLens installation (auto-detected if not set)",
    )
    ap.add_argument(
        "--skip-split", action="store_true",
        help="Skip trace splitting; run analysis directly on the full trace",
    )
    ap.add_argument(
        "--skip-validation", action="store_true",
        help="Skip trace validation (use when trace was already validated in a prior step)",
    )
    args = ap.parse_args()

    # ── Step 1: Find worker trace ──
    input_path = find_worker_trace(args.input)
    if not input_path:
        return 1

    if not args.skip_validation:
        print(f"Loading trace for validation: {input_path}")
        events = load_trace(input_path)
        print(f"Loaded {len(events)} events")

        # ── Step 2b: Validate trace ──
        if not validate_trace_for_shapes(events):
            print("\nERROR: Trace is not suitable for analysis. Exiting.")
            return 1

        if args.validate_only:
            print("\nValidation passed. Use without --validate-only for full analysis.")
            return 0

        del events
    else:
        print(f"Skipping validation (--skip-validation). Trace: {input_path}")
        if args.validate_only:
            print("Nothing to do: --validate-only combined with --skip-validation.")
            return 0

    # ── Locate TraceLens ──
    tracelens_dir = find_tracelens(args.tracelens_dir)
    if not tracelens_dir:
        print("\nERROR: TraceLens not found. Provide --tracelens-dir or set TRACELENS_DIR.")
        print("  Searched:")
        for p in _TRACELENS_SEARCH_PATHS:
            print(f"    {p}")
        return 1

    out_dir = args.output_dir or os.path.dirname(input_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    phase_reports: Dict[str, str] = {}
    execution_details: Any = []

    if args.skip_split:
        # ── Run analysis on the full trace directly ──
        report_dir = os.path.join(out_dir, "full_report")
        ok = run_tracelens_analysis(
            input_path, report_dir, tracelens_dir, label="full-trace",
        )
        if ok:
            phase_reports["full"] = report_dir

    else:
        # ── Step 3: Split trace into phases ──
        execution_details = split_trace(
            input_path, out_dir, tracelens_dir,
            find_steady_state=True,
        )

        if not execution_details:
            print("\nWARNING: Trace splitting produced no output.")
            print("  Falling back to full-trace analysis.")
            report_dir = os.path.join(out_dir, "full_report")
            ok = run_tracelens_analysis(
                input_path, report_dir, tracelens_dir, label="full-trace",
            )
            if ok:
                phase_reports["full"] = report_dir
        else:
            # ── Step 4: Run TraceLens analysis on each phase trace ──
            phase_dir = os.path.join(out_dir, "phase_traces")

            for entry in execution_details:
                trace_file = entry.get("output_path", "")
                if not trace_file or not os.path.isfile(trace_file):
                    continue

                basename = os.path.basename(trace_file)
                if "prefilldecode" in basename.lower() and "decode_" not in basename.lower().split("prefilldecode")[0]:
                    label = "prefilldecode"
                elif basename.lower().startswith("decode_"):
                    label = "decode"
                elif "annotation_iteration" in basename.lower():
                    label = "combined"
                else:
                    label = os.path.splitext(basename)[0][:40]

                # Skip if we already have this phase
                if label in phase_reports:
                    continue

                report_dir = os.path.join(out_dir, f"{label}_report")
                ok = run_tracelens_analysis(
                    trace_file, report_dir, tracelens_dir, label=label,
                )
                if ok:
                    phase_reports[label] = report_dir

    if not phase_reports:
        print("\nERROR: No TraceLens reports were generated successfully.")
        return 1

    # ── Step 5: Build and write summary ──
    summary = build_summary(out_dir, execution_details, phase_reports)
    summary_path = os.path.join(out_dir, "analysis_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote analysis summary: {summary_path}")

    print_phase_summary(summary)

    print("Phase report directories:")
    for label, rdir in phase_reports.items():
        print(f"  {label}: {rdir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
