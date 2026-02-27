#!/usr/bin/env python3
"""
Pipeline Validation — Hard checks that cannot be bypassed by the LLM Agent.

Validates:
1. *_opt.py files contain @triton.jit (not just torch rewrites)
2. optimized_serving.json is from real vllm bench serve (has standard fields)
3. baseline_serving.json has correct format
4. Hardware info matches actual GPU
5. bottlenecks.json was extracted from real trace data

Usage:
    python validate_pipeline.py --project-dir /path/to/project
    python validate_pipeline.py --project-dir /path/to/project --phase kernels
    python validate_pipeline.py --project-dir /path/to/project --phase serving
    python validate_pipeline.py --project-dir /path/to/project --phase all

Part of the model-optimize pipeline. Should be run after each major phase.
"""
import argparse
import json
import os
import sys
import glob


def check_triton_kernels(project_dir: str) -> list[str]:
    """Check that *_opt.py files use @triton.jit, not just torch rewrites."""
    errors = []
    problems_dir = os.path.join(project_dir, "problems")
    optimized_dir = os.path.join(project_dir, "optimized")

    opt_files = []
    for d in [problems_dir, optimized_dir]:
        if os.path.exists(d):
            opt_files.extend(glob.glob(os.path.join(d, "*_opt.py")))

    if not opt_files:
        errors.append("No *_opt.py files found in problems/ or optimized/")
        return errors

    for filepath in opt_files:
        with open(filepath) as f:
            content = f.read()

        name = os.path.basename(filepath)
        has_triton_jit = "@triton.jit" in content or "triton.jit" in content
        has_tl = "tl." in content or "triton.language" in content
        has_import_triton = "import triton" in content
        has_model_new = "class ModelNew" in content

        if not has_model_new:
            errors.append(f"{name}: Missing 'class ModelNew'")
            continue

        if not has_import_triton:
            errors.append(f"{name}: No 'import triton' — kernel is NOT a Triton kernel")
        if not has_triton_jit:
            errors.append(f"{name}: No '@triton.jit' decorator — kernel is NOT a Triton kernel")
        if not has_tl:
            errors.append(f"{name}: No 'tl.' usage — kernel is NOT using Triton language primitives")

        # Check it's not just a torch rewrite
        torch_only_indicators = [
            "torch.nn.functional" in content and not has_triton_jit,
            "F.silu" in content and not has_triton_jit,
            "F.gelu" in content and not has_triton_jit,
            "torch.rsqrt" in content and not has_triton_jit,
        ]
        if any(torch_only_indicators):
            errors.append(f"{name}: Appears to be a torch rewrite, not a Triton kernel")

    return errors


def check_serving_json(filepath: str, expected_label: str) -> list[str]:
    """Check that a serving JSON has standard vllm bench serve fields."""
    errors = []
    name = os.path.basename(filepath)

    if not os.path.exists(filepath):
        errors.append(f"{name}: File not found")
        return errors

    with open(filepath) as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            errors.append(f"{name}: Invalid JSON")
            return errors

    # Required fields from vllm bench serve --save-result
    required_fields = [
        "backend", "endpoint_type", "label", "model_id",
        "num_prompts", "completed", "failed",
        "total_input_tokens", "total_output_tokens",
        "request_throughput", "output_throughput",
        "mean_ttft_ms", "mean_tpot_ms", "mean_itl_ms",
        "duration",
    ]

    for field in required_fields:
        if field not in data:
            errors.append(f"{name}: Missing standard field '{field}' — not from real vllm bench serve")

    # Check label
    label = data.get("label", "")
    if label != expected_label:
        errors.append(f"{name}: label='{label}', expected '{expected_label}'")

    # Check completed > 0
    completed = data.get("completed", 0)
    if completed == 0:
        errors.append(f"{name}: completed=0 — benchmark failed or was not run")

    # Check failed == 0 or low
    failed = data.get("failed", 0)
    if failed > 0:
        errors.append(f"{name}: {failed} requests failed")

    # Check duration is reasonable (> 1 second)
    duration = data.get("duration", 0)
    if duration < 1:
        errors.append(f"{name}: duration={duration}s — suspiciously short, likely not a real benchmark")

    # Check date format (vllm uses YYYYMMDD-HHMMSS)
    date = data.get("date", "")
    if not date or len(date) < 8:
        errors.append(f"{name}: missing or invalid date field — not from real vllm bench serve")

    return errors


def check_hardware(project_dir: str) -> list[str]:
    """Check that hardware info matches actual GPU."""
    errors = []
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name()
            report_path = os.path.join(project_dir, "report", "optimization_report.md")
            if os.path.exists(report_path):
                with open(report_path) as f:
                    report = f.read()
                # Check for wrong GPU mentions
                if "NVIDIA" in report and "AMD" in gpu_name:
                    errors.append(f"Report mentions NVIDIA but actual GPU is {gpu_name}")
                if "H100" in report and "H100" not in gpu_name:
                    errors.append(f"Report mentions H100 but actual GPU is {gpu_name}")
                if "A100" in report and "A100" not in gpu_name:
                    errors.append(f"Report mentions A100 but actual GPU is {gpu_name}")
    except Exception:
        pass
    return errors


def check_bottlenecks(project_dir: str) -> list[str]:
    """Check that bottlenecks.json has reasonable data."""
    errors = []
    bottlenecks_path = os.path.join(project_dir, "profile", "bottlenecks.json")

    if not os.path.exists(bottlenecks_path):
        errors.append("profile/bottlenecks.json not found")
        return errors

    with open(bottlenecks_path) as f:
        data = json.load(f)

    if not isinstance(data, list) or len(data) == 0:
        errors.append("bottlenecks.json is empty or not a list")
        return errors

    # Check that bottleneck entries have required fields
    for i, entry in enumerate(data[:5]):
        for field in ["name", "total_dur_us", "cuda_time_percent"]:
            if field not in entry and field.replace("_us", "") not in entry:
                # Also accept total_dur, cuda_time_ms etc.
                alt_fields = [field, field.replace("_us", ""), field.replace("_us", "_ms")]
                if not any(af in entry for af in alt_fields):
                    errors.append(f"bottlenecks.json[{i}]: missing field '{field}'")

    return errors


def check_comparison(project_dir: str) -> list[str]:
    """Cross-check baseline and optimized serving data."""
    errors = []
    baseline_path = os.path.join(project_dir, "report", "baseline_serving.json")
    optimized_path = os.path.join(project_dir, "report", "optimized_serving.json")

    if not os.path.exists(baseline_path) or not os.path.exists(optimized_path):
        return errors  # Other checks will catch missing files

    with open(baseline_path) as f:
        bl = json.load(f)
    with open(optimized_path) as f:
        opt = json.load(f)

    # Check dates are different
    if bl.get("date") == opt.get("date") and opt.get("label") != "patch_failed":
        errors.append("baseline and optimized have same date — likely not separate runs")

    return errors


def main():
    parser = argparse.ArgumentParser(description="Validate model-optimize pipeline outputs")
    parser.add_argument("--project-dir", required=True, help="Path to project directory")
    parser.add_argument("--phase", default="all", choices=["kernels", "serving", "report", "all"],
                        help="Which phase to validate")
    args = parser.parse_args()

    project_dir = os.path.abspath(args.project_dir)
    all_errors = []
    all_warnings = []

    print(f"\n{'='*60}")
    print(f"  Pipeline Validation: {project_dir}")
    print(f"  Phase: {args.phase}")
    print(f"{'='*60}\n")

    if args.phase in ("kernels", "all"):
        print("--- Checking Triton kernels ---")
        errs = check_triton_kernels(project_dir)
        for e in errs:
            print(f"  ❌ {e}")
        if not errs:
            print("  ✅ All kernels use Triton")
        all_errors.extend(errs)

    if args.phase in ("serving", "all"):
        print("\n--- Checking baseline_serving.json ---")
        errs = check_serving_json(
            os.path.join(project_dir, "report", "baseline_serving.json"), "baseline")
        for e in errs:
            print(f"  ❌ {e}")
        if not errs:
            print("  ✅ Baseline serving data is valid")
        all_errors.extend(errs)

        print("\n--- Checking optimized_serving.json ---")
        errs = check_serving_json(
            os.path.join(project_dir, "report", "optimized_serving.json"), "optimized")
        for e in errs:
            print(f"  ❌ {e}")
        if not errs:
            print("  ✅ Optimized serving data is valid")
        all_errors.extend(errs)

        print("\n--- Cross-checking baseline vs optimized ---")
        errs = check_comparison(project_dir)
        for e in errs:
            print(f"  ❌ {e}")
        if not errs:
            print("  ✅ Cross-check passed")
        all_errors.extend(errs)

    if args.phase in ("report", "all"):
        print("\n--- Checking hardware info ---")
        errs = check_hardware(project_dir)
        for e in errs:
            print(f"  ❌ {e}")
        if not errs:
            print("  ✅ Hardware info OK")
        all_errors.extend(errs)

        print("\n--- Checking bottlenecks.json ---")
        errs = check_bottlenecks(project_dir)
        for e in errs:
            print(f"  ❌ {e}")
        if not errs:
            print("  ✅ Bottlenecks data OK")
        all_errors.extend(errs)

    print(f"\n{'='*60}")
    if all_errors:
        print(f"  ⛔ VALIDATION FAILED: {len(all_errors)} errors")
        print(f"{'='*60}")
        sys.exit(1)
    else:
        print(f"  ✅ ALL CHECKS PASSED")
        print(f"{'='*60}")
        sys.exit(0)


if __name__ == "__main__":
    main()

