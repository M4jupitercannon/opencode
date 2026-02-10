# Phase 4: Performance Profiling {{SKIP_LABEL}}

## Goal
Profile the model to identify bottleneck operators/kernels.

## ⚠️ CRITICAL: Keep Profiling Lightweight
- **For image/video generation models**: Use ONLY 2-3 diffusion steps (NOT the full 20-50 steps)
- **For LLMs**: Generate only 10-20 tokens
- **Trace file should be < 100MB** - if larger, reduce steps/tokens

## ⚠️ CRITICAL: Use rocprof for GPU profiling on ROCm
**PyTorch's profiler.key_averages().table() does NOT show CUDA/GPU times on ROCm!**
You MUST use rocprof for accurate GPU kernel timing.

## Profiling Script

Create `{{PROFILE_DIR}}/profile_model.py` using rocprof for GPU kernel timing:

```bash
source {{OUTPUT_DIR}}/venv/bin/activate
cd {{PROFILE_DIR}}
rocprof --stats -o rocprof_results.csv python {{DEMO_DIR}}/demo.py
```

Parse the `rocprof_results.stats.csv` to identify top bottleneck kernels. Save results to `{{PROFILE_DIR}}/bottlenecks.json`.

## Shape Capture

A standalone `shape_capture.py` script is provided at `{{OUTPUT_DIR}}/scripts/shape_capture.py`.
Copy and use it to capture dynamic shape ranges during inference:

```bash
cp {{OUTPUT_DIR}}/scripts/shape_capture.py {{PROFILE_DIR}}/
cd {{PROFILE_DIR}}
python shape_capture.py
```

This generates `{{PROFILE_DIR}}/shape_ranges.json` with min/typical/max for each dimension, which is critical for creating problem files with correct shapes.

## Steps
1. Create and run the profiling script (use rocprof)
2. Run shape capture to get dynamic shape ranges
3. Analyze bottlenecks.json to identify top time-consuming operators
4. Focus on operators that take > 5% of total CUDA time
5. Update progress.json with bottleneck list

