# Phase 3: Fix Compatibility Issues {{SKIP_LABEL}}

## Goal
If demo.py fails, diagnose and fix issues using monkey-patching.

## IMPORTANT: Always Activate venv
```bash
source {{OUTPUT_DIR}}/venv/bin/activate
```

## ⚠️ Fixing Dependencies
Since we have our OWN venv, we CAN safely modify packages within it:
```bash
pip install some-missing-package
pip install --upgrade diffusers
# STILL NEVER modify /opt/ or /usr/ system directories!
```

## Common Issues & Fixes

### Issue: Missing custom operators
```python
# Create patches/custom_ops.py
import torch
def my_custom_op(x):
    return x
import transformers.models.xxx as module
module.custom_op = my_custom_op
```

### Issue: Unsupported attention implementation
```python
# patches/attention_fix.py
def patched_attention(self, query, key, value, **kwargs):
    return torch.nn.functional.scaled_dot_product_attention(query, key, value)
ModelClass._attention = patched_attention
```

### Issue: RoPE/positional encoding errors
```python
# patches/rope_fix.py
def fixed_rope(x, seq_len):
    ...
```

## Steps
1. Run demo.py and capture the error
2. Analyze the error traceback
3. Create fix in `{{DEMO_DIR}}/patches/` directory
4. Update demo.py to import patches first
5. Re-run demo.py until it works
6. Update progress.json

