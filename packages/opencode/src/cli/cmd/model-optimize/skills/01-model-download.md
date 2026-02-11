# Phase 1: Model Serving with vLLM {{SKIP_LABEL}}

## Goal
Start the model using `vllm serve` and verify it works. vLLM handles model download automatically.

## ⚠️ vLLM Mode
In vLLM mode, there is NO need to:
- Manually download the model (vLLM auto-downloads from HuggingFace)
- Write a demo inference script
- Fix compatibility issues manually

## Steps

### 1. Test vLLM serve
```bash
source {{OUTPUT_DIR}}/venv/bin/activate

# Quick test: start vllm serve and send a test request
# Adjust --tensor-parallel-size based on model size and GPU count
vllm serve {{HF_MODEL}} \
  --dtype auto \
  --max-model-len 2048 \
  --port 8192 \
  --disable-log-requests &

VLLM_PID=$!
sleep 30  # Wait for model to load

# Test with a simple request
curl -s http://localhost:8192/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "{{HF_MODEL}}", "prompt": "Hello, I am", "max_tokens": 20}' | python3 -m json.tool

# Kill the test server
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null
```

### 2. Record model config
```bash
source {{OUTPUT_DIR}}/venv/bin/activate
python3 -c "
from transformers import AutoConfig
config = AutoConfig.from_pretrained('{{HF_MODEL}}', trust_remote_code=True)
import json
info = {
    'model_type': getattr(config, 'model_type', 'unknown'),
    'num_hidden_layers': getattr(config, 'num_hidden_layers', None),
    'hidden_size': getattr(config, 'hidden_size', None),
    'num_attention_heads': getattr(config, 'num_attention_heads', None),
    'num_key_value_heads': getattr(config, 'num_key_value_heads', None),
    'intermediate_size': getattr(config, 'intermediate_size', None),
    'vocab_size': getattr(config, 'vocab_size', None),
}
print(json.dumps(info, indent=2))
with open('{{OUTPUT_DIR}}/model_config.json', 'w') as f:
    json.dump(info, f, indent=2)
"
```

### 3. Update progress.json
Update progress.json: phases_completed.append("download"), phases_completed.append("demo"), phases_completed.append("compatibility")

> **Note**: In vLLM mode, Phase 1 covers download + demo + compatibility in one step.
