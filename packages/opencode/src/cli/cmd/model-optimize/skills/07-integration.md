# Phase 7: Integration & Final Testing {{SKIP_LABEL}}

## Goal
Integrate optimized kernels into the model using monkey-patching and **MEASURE ACTUAL end-to-end performance**.

## ⚠️ CRITICAL REQUIREMENTS
1. **MEASURE ACTUAL end-to-end speedup** - NOT estimated
2. **Generate BOTH original AND optimized outputs** with same random seed
3. **Run the SAME inference with and without optimizations**

## Use Project venv
```bash
source {{OUTPUT_DIR}}/venv/bin/activate
```

### What You CAN Do:
- Edit/install packages in project venv
- Create files in `{{OUTPUT_DIR}}/`
- Use monkey-patching to override behavior
- Install AITER in project venv if needed

### What You CANNOT Do:
- **NEVER edit /opt/, /usr/, or system site-packages**
- **NEVER report only "estimated" speedup**

## Create Integration Script: `{{OPTIMIZED_DIR}}/integrate.py`

Write an `integrate.py` that:
1. Imports all available optimized kernels (ModelNew classes from `*_opt.py`)
2. Provides `apply_all_patches(model_or_pipe)` function
3. Patches RMSNorm/LayerNorm, activations, attention layers
4. Returns (patched_model, stats_dict)

Key pattern for patching normalization layers:
```python
for name, module in model.named_modules():
    if "RMSNorm" in module.__class__.__name__:
        opt_module = OptKernel(hidden_size)
        opt_module.weight.data = module.weight.data.clone()
        module.forward = lambda x, m=opt_module: m(x)
```

## MANDATORY: End-to-End Performance Measurement

Create `{{OPTIMIZED_DIR}}/measure_e2e.py` that:
1. Loads model WITHOUT optimizations → benchmark (warmup + N runs)
2. Applies patches via `integrate.apply_all_patches()`
3. Benchmarks WITH optimizations (warmup + N runs)
4. Generates comparison outputs with fixed seed (SEED=42)
5. Saves results to `{{REPORT_DIR}}/comparison_outputs/comparison_results.json`

### For Text Generation:
```python
SEED = 42
torch.manual_seed(SEED); torch.cuda.manual_seed(SEED)
output_original = model_original.generate(**inputs, max_new_tokens=100, do_sample=True)
# ... save to comparison_outputs/original_output.txt

torch.manual_seed(SEED); torch.cuda.manual_seed(SEED)
output_optimized = model_optimized.generate(**inputs, max_new_tokens=100, do_sample=True)
# ... save to comparison_outputs/optimized_output.txt
```

### For Image Generation:
```python
SEED = 42
generator = torch.Generator(device="cuda").manual_seed(SEED)
image_original = pipe_original(prompt, generator=generator).images[0]
# ... save original_output.png, optimized_output.png, comparison.png
```

## Steps
1. Create integrate.py with monkey-patches
2. Create measure_e2e.py
3. Run measurement to get ACTUAL end-to-end timing
4. Generate comparison outputs with same seed
5. If correctness fails, debug and fix
6. Record ACTUAL speedup in progress.json

