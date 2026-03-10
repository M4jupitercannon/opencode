# Phase 1: Model Serving with vLLM {{SKIP_LABEL}}

## Goal

Start the model using `vllm serve` and verify it works. vLLM handles model download automatically.

## ⚠️ Docker vs venv

If Phase 0 created a Docker container (`env_type: "docker"` in `env_info.json`), prefix all commands with `docker exec $CONTAINER_NAME bash -c "..."` and use `HIP_VISIBLE_DEVICES=$BEST_GPU` to target the free GPU. Skip `source venv/bin/activate` inside the container (packages are pre-installed).

Before running this phase, set execution mode once:

```bash
ENV_TYPE=$(python3 -c "import json; print(json.load(open('{{OUTPUT_DIR}}/env_info.json')).get('env_type','venv'))" 2>/dev/null || echo "venv")
CONTAINER_NAME=$(python3 -c "import json; print(json.load(open('{{OUTPUT_DIR}}/env_info.json')).get('container','vllm_model_opt'))" 2>/dev/null || echo "vllm_model_opt")
BEST_GPU=$(python3 -c "import json; print(json.load(open('{{OUTPUT_DIR}}/env_info.json')).get('best_gpu',0))" 2>/dev/null || echo 0)
```

## ⚠️ vLLM Mode

In vLLM mode, there is NO need to:

- Manually download the model (vLLM auto-downloads from HuggingFace)
- Write a demo inference script
- Fix compatibility issues manually

## ⚠️ CRITICAL: Never dump vLLM logs into bash output

**ALL vLLM commands MUST redirect output to log files.** vLLM logs are thousands of lines and will break the session context.

## Steps

### 1. Test vLLM serve

```bash
# venv mode only:
# source {{OUTPUT_DIR}}/venv/bin/activate

# Start vLLM — ALL output to log file, NEVER to stdout
vllm serve {{HF_MODEL}} \
  --dtype auto \
  --max-model-len 2048 \
  --port 8192 \
  --no-enable-log-requests &> {{OUTPUT_DIR}}/vllm_serve.log &
VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"

# Wait for server (silent polling)
for i in $(seq 1 60); do
  curl -s http://localhost:8192/health > /dev/null 2>&1 && break
  sleep 5
done
curl -s http://localhost:8192/health > /dev/null 2>&1 && echo "Server ready" || echo "FAILED — check {{OUTPUT_DIR}}/vllm_serve.log"

# Quick inference test (only show the result, not vllm internals)
curl -s http://localhost:8192/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "{{HF_MODEL}}", "prompt": "Hello, I am", "max_tokens": 20}' \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print('Inference OK' if 'choices' in d else f'Error: {d}')"

# Kill the test server
kill $VLLM_PID 2>/dev/null; wait $VLLM_PID 2>/dev/null
```

### 2. Record model config

```bash
# venv mode only:
# source {{OUTPUT_DIR}}/venv/bin/activate
python3 -c "
import json
try:
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained('{{HF_MODEL}}', trust_remote_code=True)
    d = config.to_dict()
except Exception:
    from huggingface_hub import hf_hub_download
    cfg_path = hf_hub_download('{{HF_MODEL}}', 'config.json')
    with open(cfg_path) as f: d = json.load(f)

tc = d.get('text_config', d)
info = {
    'model_type': d.get('model_type', 'unknown'),
    'architectures': d.get('architectures', []),
    'num_hidden_layers': tc.get('num_hidden_layers', None),
    'hidden_size': tc.get('hidden_size', None),
    'num_attention_heads': tc.get('num_attention_heads', None),
    'num_key_value_heads': tc.get('num_key_value_heads', None),
    'intermediate_size': tc.get('intermediate_size', None),
    'vocab_size': tc.get('vocab_size', None),
    'layer_types': tc.get('layer_types', None),
    'full_attention_interval': tc.get('full_attention_interval', None),
}
print(json.dumps(info, indent=2))
with open('{{OUTPUT_DIR}}/model_config.json', 'w') as f:
    json.dump(info, f, indent=2)
"
```

### 3. Update progress.json

Update progress.json: phases_completed.append("download"), phases_completed.append("demo"), phases_completed.append("compatibility")

> **Note**: In vLLM mode, Phase 1 covers download + demo + compatibility in one step.
