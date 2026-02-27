#!/usr/bin/env python3
"""
vLLM Trace Extractor (ALKA-style)

Extracts GPU kernel events from vLLM torch profiler trace files.
Produces full kernel trace CSV and unique kernel summary CSV.

Based on the ALKA (Automatic LLM Kernel Analyzer) project:
  https://github.com/ROCm/ALKA

Usage:
    python vllm_trace_extractor.py -i trace.pt.trace.json
    python vllm_trace_extractor.py -i trace.pt.trace.json.gz --full-csv full.csv --unique-csv summary.csv

Part of the model-optimize pipeline. Can be used standalone.
"""
import argparse
import csv
import gzip
import json
import os
import sys
from typing import Any, Dict, List, Tuple
from statistics import mean, median


def open_maybe_gzip(path: str):
    """Open a file, auto-detecting gzip compression."""
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "r")


def load_trace_events(path: str) -> Tuple[List[Dict[str, Any]], str]:
    """Load trace events from a torch profiler JSON file."""
    with open_maybe_gzip(path) as f:
        data = json.load(f)

    display_unit = "us"
    if isinstance(data, dict):
        display_unit = data.get("displayTimeUnit", "us")
        trace = data.get("traceEvents") or data.get("events")
    else:
        trace = None

    if not isinstance(trace, list):
        raise ValueError("Trace JSON missing 'traceEvents' list")

    return trace, display_unit


def extract_kernel_events(trace: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract GPU kernel events (cat=='kernel') from trace."""
    kernels = []
    for ev in trace:
        if ev.get("cat") != "kernel":
            continue
        args = ev.get("args") or {}
        kernels.append({
            "name": ev.get("name", ""),
            "ts": ev.get("ts"),
            "dur": ev.get("dur"),
            "pid": ev.get("pid"),
            "tid": ev.get("tid"),
            "device": args.get("device"),
            "stream": args.get("stream"),
            "correlation": args.get("correlation"),
            "kind": args.get("kind"),
            "grid": "x".join(str(v) for v in args.get("grid", [])) if args.get("grid") else "",
            "block": "x".join(str(v) for v in args.get("block", [])) if args.get("block") else "",
            "external_id": args.get("External id"),
        })
    # Sort by timestamp
    kernels.sort(key=lambda e: (e["ts"] if e["ts"] is not None else float("inf")))
    return kernels


def summarize_kernels(kernels: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate kernel events by name, sorted by total duration (descending)."""
    summary = {}
    for k in kernels:
        name = k["name"]
        dur = k.get("dur")
        if dur is None:
            continue
        if name not in summary:
            summary[name] = []
        summary[name].append(dur)

    rows = []
    for name, durs in summary.items():
        rows.append({
            "name": name,
            "count": len(durs),
            "total_dur": sum(durs),
            "avg_dur": mean(durs),
            "median_dur": median(durs),
            "min_dur": min(durs),
            "max_dur": max(durs),
        })
    rows.sort(key=lambda r: r["total_dur"], reverse=True)
    return rows


def write_full_csv(path: str, events: List[Dict[str, Any]]) -> None:
    """Write full kernel trace to CSV."""
    fieldnames = ["index", "name", "ts", "dur", "pid", "tid", "device",
                  "stream", "correlation", "kind", "grid", "block", "external_id"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, ev in enumerate(events):
            writer.writerow({"index": idx, **ev})


def write_unique_csv(path: str, summary: List[Dict[str, Any]]) -> None:
    """Write unique kernel summary to CSV."""
    fieldnames = ["name", "count", "total_dur", "avg_dur", "median_dur", "min_dur", "max_dur"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)


def default_output_paths(input_path: str) -> Tuple[str, str]:
    """Generate default output paths from input trace path."""
    base = os.path.basename(input_path)
    for suffix in [".pt.trace.json.gz", ".trace.json.gz", ".json.gz", ".json"]:
        if base.endswith(suffix):
            base = base[:-len(suffix)]
            break
    dirname = os.path.dirname(input_path) or "."
    full_csv = os.path.join(dirname, f"{base}_kernel_full.csv")
    uniq_csv = os.path.join(dirname, f"{base}_kernel_unique.csv")
    return full_csv, uniq_csv


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract GPU kernel events from a vLLM/torch profiler trace (ALKA-style)")
    parser.add_argument("-i", "--input", required=True,
                        help="Path to .pt.trace.json or .pt.trace.json.gz")
    parser.add_argument("--full-csv", help="Output CSV path for full kernel trace")
    parser.add_argument("--unique-csv", help="Output CSV path for unique kernel summary")
    args = parser.parse_args()

    print(f"Loading trace: {args.input}")
    trace, display_unit = load_trace_events(args.input)
    kernels = extract_kernel_events(trace)
    print(f"Found {len(kernels)} kernel events (time unit: {display_unit})")

    if not kernels:
        print("WARNING: No kernel events found in trace!")
        return 1

    full_csv, uniq_csv = default_output_paths(args.input)
    if args.full_csv:
        full_csv = args.full_csv
    if args.unique_csv:
        uniq_csv = args.unique_csv

    write_full_csv(full_csv, kernels)
    summary = summarize_kernels(kernels)
    write_unique_csv(uniq_csv, summary)

    print(f"\nWrote full kernel trace: {full_csv} ({len(kernels)} events)")
    print(f"Wrote unique kernels: {uniq_csv} ({len(summary)} unique kernels)")

    # Print top 10 summary
    total_time = sum(k["total_dur"] for k in summary)
    print(f"\nTotal GPU kernel time: {total_time/1000:.2f}ms")
    print(f"\n{'Rank':>4} {'Kernel':<50} {'%':>6} {'Total':>10} {'Count':>6}")
    print("-" * 80)
    for i, row in enumerate(summary[:10], 1):
        pct = row["total_dur"] / total_time * 100 if total_time > 0 else 0
        print(f"{i:4d} {row['name'][:50]:<50} {pct:5.1f}% {row['total_dur']/1000:9.2f}ms {row['count']:6d}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

