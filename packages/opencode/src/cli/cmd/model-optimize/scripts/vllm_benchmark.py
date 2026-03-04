#!/usr/bin/env python3
"""
vLLM Serving Benchmark & Trace Collector

Orchestrates vLLM serve + bench serve for:
1. Performance benchmarking (throughput, TTFT, TPOT, ITL)
2. Torch profiler trace collection with shape recording for kernel analysis

Usage:
    # Benchmark only (no trace)
    python vllm_benchmark.py --model Qwen/Qwen3-8B --mode benchmark

    # Collect trace for kernel analysis (records shapes)
    python vllm_benchmark.py --model Qwen/Qwen3-8B --mode trace

    # Both benchmark and trace
    python vllm_benchmark.py --model Qwen/Qwen3-8B --mode all

    # With custom concurrency and lengths
    python vllm_benchmark.py --model Qwen/Qwen3-8B --concurrency 32 --input-len 1024 --output-len 1024

Part of the model-optimize pipeline. Can be used standalone.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
import urllib.error


def wait_for_server(port: int, timeout: int = 300) -> bool:
    """Wait for vLLM server to be ready."""
    import urllib.request
    start = time.time()
    while time.time() - start < timeout:
        try:
            req = urllib.request.Request(f"http://localhost:{port}/health")
            urllib.request.urlopen(req, timeout=5)
            return True
        except Exception:
            time.sleep(5)
    return False


def start_vllm_serve(model: str, port: int, extra_args: list[str] = None,
                     trace_dir: str = None, profiler_config: str = None) -> subprocess.Popen:
    """Start vLLM serve as a background process."""
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--port", str(port),
        "--dtype", "auto",
        "--disable-log-requests",
    ]
    if extra_args:
        cmd.extend(extra_args)

    if profiler_config:
        cmd.extend(["--profiler-config", profiler_config])

    env = os.environ.copy()
    if trace_dir:
        os.makedirs(trace_dir, exist_ok=True)
        # VLLM_RPC_TIMEOUT: trace flush after /stop_profile can take minutes for
        # large models (vLLM docs recommend 30min for 100 reqs on 70B).
        # Default is only 10s which causes timeouts.
        env.setdefault("VLLM_RPC_TIMEOUT", "1800000")

    print(f"Starting vLLM serve: {' '.join(cmd)}")
    if trace_dir:
        print(f"  Trace output: {trace_dir}")

    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return proc


def run_bench_serve(model: str, port: int, num_prompts: int, concurrency: int,
                    input_len: int, output_len: int, result_dir: str,
                    result_filename: str = "benchmark_results.json",
                    label: str = "") -> dict:
    """Run vllm bench serve and return results."""
    os.makedirs(result_dir, exist_ok=True)

    cmd = [
        "vllm", "bench", "serve",
        "--model", model,
        "--port", str(port),
        "--dataset-name", "random",
        "--input-len", str(input_len),
        "--output-len", str(output_len),
        "--num-prompts", str(num_prompts),
        "--max-concurrency", str(concurrency),
        "--request-rate", "inf",
        "--save-result",
        "--result-dir", result_dir,
        "--result-filename", result_filename,
    ]
    if label:
        cmd.extend(["--label", label])

    print(f"\nRunning benchmark: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    if result.returncode != 0:
        print(f"Benchmark stderr:\n{result.stderr[-2000:]}")
        # Try to parse any results that were saved
    
    # Print stdout (contains the summary)
    if result.stdout:
        print(result.stdout[-3000:])

    # Load results
    result_path = os.path.join(result_dir, result_filename)
    if os.path.exists(result_path):
        with open(result_path) as f:
            return json.load(f)

    print(f"WARNING: Result file not found: {result_path}")
    return {}


def stop_server(proc: subprocess.Popen):
    """Gracefully stop a vLLM server."""
    if proc.poll() is None:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def profiler_api_call(port: int, action: str) -> bool:
    """Call vLLM profiler API (/start_profile or /stop_profile). Returns True on success."""
    url = f"http://localhost:{port}/{action}"
    try:
        req = urllib.request.Request(url, method="POST")
        resp = urllib.request.urlopen(req, timeout=60)
        print(f"  /{action}: {resp.status} {resp.read().decode()[:200]}")
        return resp.status == 200
    except urllib.error.HTTPError as e:
        print(f"  /{action}: HTTP {e.code} — {e.read().decode()[:200]}")
        return False
    except Exception as e:
        print(f"  /{action}: failed — {e}")
        return False


def build_profiler_config_arg(trace_dir: str) -> str:
    """Build --profiler-config JSON string with record_shapes enabled."""
    config = {
        "profiler": "torch",
        "torch_profiler_dir": trace_dir,
        "torch_profiler_record_shapes": True,
        "torch_profiler_with_stack": True,
        "torch_profiler_with_flops": True,
        "torch_profiler_with_memory": False,
        "torch_profiler_use_gzip": True,
        "ignore_frontend": True,
    }
    return json.dumps(config)


def find_worker_trace(trace_dir: str) -> str:
    """Select the worker trace (rank-0) from trace_dir, rejecting async_llm traces.

    vLLM writes two trace files per profiling session:
      - *async_llm* — frontend-only (CPU python_function events, NO GPU kernels)
      - *rank-0*    — worker trace (CPU ops + CUDA kernels with shapes)

    Returns the path to the worker trace, or empty string if not found.
    """
    if not os.path.isdir(trace_dir):
        return ""

    all_traces = sorted(
        [os.path.join(trace_dir, f) for f in os.listdir(trace_dir)
         if f.endswith(".json") or f.endswith(".json.gz")],
        key=os.path.getmtime, reverse=True,
    )
    if not all_traces:
        return ""

    # Prefer rank-0 worker traces
    for t in all_traces:
        basename = os.path.basename(t)
        if "rank" in basename and "async_llm" not in basename:
            return t

    # Fallback: any non-async_llm trace
    for t in all_traces:
        if "async_llm" not in os.path.basename(t):
            return t

    print(f"WARNING: Only async_llm frontend traces found in {trace_dir}")
    print("  These contain only Python function calls — no GPU kernels or shapes.")
    return ""


def verify_shapes_in_trace(trace_path: str) -> bool:
    """Validate that a trace file has GPU kernels with External id and CPU ops with Input Dims.

    Returns True if the trace is valid for shape analysis, False otherwise.
    """
    import gzip
    opener = gzip.open if trace_path.endswith(".gz") else open
    try:
        with opener(trace_path, "rt") as f:
            data = json.load(f)

        events = data if isinstance(data, list) else data.get("traceEvents", [])

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

        print(f"\n  Trace verification: {trace_path}")
        print(f"    Size:                 {os.path.getsize(trace_path) / 1_000_000:.1f} MB")
        print(f"    Total events:         {len(events)}")
        print(f"    CPU ops (cpu_op):     {n_cpu_ops}")
        print(f"    CPU ops with shapes:  {n_with_shapes}")
        print(f"    GPU kernels:          {n_gpu_kernels}")
        print(f"    GPU kernels w/ ext_id:{n_with_ext_id}")

        ok = True
        if n_gpu_kernels == 0:
            if n_frontend > 0:
                print(f"  FAIL: This is an async_llm frontend trace ({n_frontend} python_function events)")
                print(f"    This trace has NO GPU kernels. Select the *rank-0* worker trace instead.")
            else:
                print(f"  FAIL: No GPU kernel events in trace")
            ok = False
        if n_cpu_ops == 0:
            print(f"  FAIL: No cpu_op events — --enforce-eager was likely missing")
            ok = False
        if n_with_shapes == 0 and n_cpu_ops > 0:
            print(f"  FAIL: No Input Dims on cpu_ops — torch_profiler_record_shapes was likely not set")
            ok = False
        if n_with_ext_id == 0 and n_gpu_kernels > 0:
            print(f"  FAIL: No External id on GPU kernels — --enforce-eager was likely missing")
            ok = False

        if ok:
            pct = n_with_shapes / n_cpu_ops * 100 if n_cpu_ops else 0
            print(f"  PASSED — {pct:.0f}% of CPU ops have shape data, "
                  f"{n_with_ext_id} GPU kernels have External id")
        else:
            print(f"\n  Trace verification FAILED. Shape analysis will not produce useful results.")
            print(f"  Required profiling setup:")
            print(f"    1. --enforce-eager on the vllm serve command")
            print(f"    2. --profiler-config with torch_profiler_record_shapes: true, ignore_frontend: true")
            print(f"    3. /start_profile API call BEFORE requests, /stop_profile AFTER")
            print(f"    4. Select the *rank-0* worker trace, not the *async_llm* trace")

        return ok
    except Exception as e:
        print(f"\n  Could not verify trace: {e}")
        return False


def benchmark_mode(args) -> dict:
    """Run benchmark only (no trace collection)."""
    port = args.port
    proc = start_vllm_serve(args.model, port, args.serve_args)

    try:
        print(f"Waiting for vLLM to be ready on port {port}...")
        if not wait_for_server(port, timeout=args.timeout):
            print("ERROR: vLLM server did not start in time")
            return {}

        print(f"Server ready! Running benchmark...")
        results = run_bench_serve(
            model=args.model, port=port,
            num_prompts=args.num_prompts,
            concurrency=args.concurrency,
            input_len=args.input_len,
            output_len=args.output_len,
            result_dir=args.result_dir,
            result_filename=args.result_filename,
            label=args.label,
        )
        return results
    finally:
        stop_server(proc)


def trace_mode(args) -> str:
    """Collect torch profiler trace with shape recording during serving.

    Flow:
      1. Start vLLM with --profiler-config (record_shapes=True)
      2. Wait for server readiness
      3. POST /start_profile  to begin profiling
      4. Send benchmark requests to generate representative GPU work
      5. POST /stop_profile   to flush the trace
      6. Verify trace contains shape data
    """
    port = args.port + 1  # Use different port for trace collection
    # vLLM requires torch_profiler_dir to be an absolute path.
    trace_dir = os.path.abspath(os.path.join(args.result_dir, "traces"))

    # Build profiler config (JSON string) with record_shapes=True
    profiler_cfg = build_profiler_config_arg(trace_dir)
    print(f"Profiler config: record_shapes=True, with_stack=True, with_flops=True")

    # Force eager mode during tracing so each GPU kernel correctly correlates
    # to its parent CPU op.  With CUDA Graphs the correlation IDs are lost.
    trace_serve_args = list(args.serve_args or [])
    if "--enforce-eager" not in trace_serve_args:
        trace_serve_args.append("--enforce-eager")
        print("  Adding --enforce-eager for accurate kernel↔op correlation")

    proc = start_vllm_serve(args.model, port, trace_serve_args,
                            trace_dir=trace_dir, profiler_config=profiler_cfg)

    try:
        print(f"Waiting for vLLM (with profiler) to be ready on port {port}...")
        if not wait_for_server(port, timeout=args.timeout):
            print("ERROR: vLLM server did not start in time")
            return ""

        # Explicitly start profiling via API (ensures record_shapes is active)
        print("\nStarting profiler via /start_profile ...")
        api_ok = profiler_api_call(port, "start_profile")
        if not api_ok:
            print("  /start_profile not available; relying on env-var auto-profiling")

        # Send fewer requests for trace (to keep trace size manageable)
        trace_prompts = min(args.num_prompts, 30)
        print(f"Sending {trace_prompts} requests for trace collection (concurrency={args.concurrency})...")

        run_bench_serve(
            model=args.model, port=port,
            num_prompts=trace_prompts,
            concurrency=args.concurrency,
            input_len=args.input_len,
            output_len=args.output_len,
            result_dir=args.result_dir,
            result_filename="trace_benchmark.json",
            label="trace",
        )

        # Stop profiling and flush trace
        print("\nStopping profiler via /stop_profile ...")
        profiler_api_call(port, "stop_profile")

        # Wait for profiler to flush
        time.sleep(10)
    finally:
        stop_server(proc)

    # Find the worker trace (rank-0), not the async_llm frontend trace
    print(f"\nLooking for worker trace in: {trace_dir}")
    all_traces = [os.path.join(trace_dir, f) for f in os.listdir(trace_dir)
                  if f.endswith(".json") or f.endswith(".json.gz")]
    if all_traces:
        print(f"  All trace files: {[os.path.basename(t) for t in all_traces]}")

    trace_file = find_worker_trace(trace_dir)
    if trace_file:
        print(f"  Selected worker trace: {os.path.basename(trace_file)}")
        valid = verify_shapes_in_trace(trace_file)
        if not valid:
            print("\nERROR: Trace verification failed — shape analysis will not work.")
            print("Re-run with correct flags (see verification output above).")
            return ""
        return trace_file
    else:
        print("ERROR: No valid worker trace files found!")
        if all_traces:
            print(f"  Found {len(all_traces)} trace files, but all appear to be async_llm frontend traces.")
            print("  Ensure ignore_frontend: true is set in --profiler-config.")
        return ""


def main():
    parser = argparse.ArgumentParser(description="vLLM Benchmark & Trace Collector")
    parser.add_argument("--model", required=True, help="HuggingFace model name")
    parser.add_argument("--mode", choices=["benchmark", "trace", "all"], default="all",
                        help="Mode: benchmark only, trace only, or both")
    parser.add_argument("--concurrency", type=int, default=16,
                        help="Max concurrent requests (default: 16)")
    parser.add_argument("--input-len", type=int, default=1024,
                        help="Input sequence length (default: 1024)")
    parser.add_argument("--output-len", type=int, default=1024,
                        help="Output sequence length (default: 1024)")
    parser.add_argument("--num-prompts", type=int, default=100,
                        help="Number of prompts for benchmark (default: 100)")
    parser.add_argument("--port", type=int, default=8192,
                        help="Base port for vLLM serve (default: 8192)")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Server startup timeout in seconds (default: 300)")
    parser.add_argument("--result-dir", default="./benchmark_results",
                        help="Directory for results (default: ./benchmark_results)")
    parser.add_argument("--result-filename", default="benchmark_results.json",
                        help="Result JSON filename (default: benchmark_results.json)")
    parser.add_argument("--label", default="", help="Label prefix for results")
    parser.add_argument("--serve-args", nargs="*", default=[],
                        help="Extra args for vllm serve (e.g., --tensor-parallel-size 2)")
    parser.add_argument("--config", default=None, help="Path to config.json (for pipeline mode)")
    args = parser.parse_args()

    # Load config if provided
    if args.config:
        with open(args.config) as f:
            config = json.load(f)
        if not args.model:
            args.model = config.get("hf_model", args.model)
        if args.result_dir == "./benchmark_results":
            args.result_dir = config.get("dirs", {}).get("profile", args.result_dir)

    results = {}

    if args.mode in ("benchmark", "all"):
        print("=" * 60)
        print("  BENCHMARK MODE")
        print("=" * 60)
        results["benchmark"] = benchmark_mode(args)

    if args.mode in ("trace", "all"):
        print("\n" + "=" * 60)
        print("  TRACE COLLECTION MODE (record_shapes=True)")
        print("=" * 60)
        trace_file = trace_mode(args)
        results["trace_file"] = trace_file

    # Save combined results
    combined_path = os.path.join(args.result_dir, "combined_results.json")
    with open(combined_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nCombined results saved to: {combined_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
