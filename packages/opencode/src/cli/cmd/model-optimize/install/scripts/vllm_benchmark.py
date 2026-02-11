#!/usr/bin/env python3
"""
vLLM Serving Benchmark & Trace Collector

Orchestrates vLLM serve + bench serve for:
1. Performance benchmarking (throughput, TTFT, TPOT, ITL)
2. Torch profiler trace collection for kernel analysis

Usage:
    # Benchmark only (no trace)
    python vllm_benchmark.py --model Qwen/Qwen3-8B --mode benchmark

    # Collect trace for kernel analysis
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
                     trace_dir: str = None) -> subprocess.Popen:
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

    env = os.environ.copy()
    if trace_dir:
        os.makedirs(trace_dir, exist_ok=True)
        env["VLLM_TORCH_PROFILER_DIR"] = trace_dir

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
        sys.executable, "-m", "vllm.entrypoints.openai.run_batch_benchmark" 
    ]
    # Use vllm bench serve CLI
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
    """Collect torch profiler trace during serving."""
    port = args.port + 1  # Use different port for trace collection
    trace_dir = os.path.join(args.result_dir, "traces")

    proc = start_vllm_serve(args.model, port, args.serve_args, trace_dir=trace_dir)

    try:
        print(f"Waiting for vLLM (with profiler) to be ready on port {port}...")
        if not wait_for_server(port, timeout=args.timeout):
            print("ERROR: vLLM server did not start in time")
            return ""

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

        # Wait for profiler to flush
        time.sleep(10)
    finally:
        stop_server(proc)

    # Find the generated trace file
    trace_files = []
    for f in os.listdir(trace_dir):
        if f.endswith(".json") or f.endswith(".json.gz"):
            trace_files.append(os.path.join(trace_dir, f))

    if trace_files:
        trace_files.sort(key=os.path.getmtime, reverse=True)
        print(f"Trace files collected: {trace_files}")
        return trace_files[0]
    else:
        print("WARNING: No trace files found!")
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
        print("  TRACE COLLECTION MODE")
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

