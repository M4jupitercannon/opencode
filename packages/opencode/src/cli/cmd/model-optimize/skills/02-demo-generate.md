# Phase 2: Generate Demo Script {{SKIP_LABEL}}

## Goal
Create a working demo script that runs inference on the model.

## IMPORTANT: Always Activate venv
```bash
source {{OUTPUT_DIR}}/venv/bin/activate
```

## Steps
1. **Detect model type** by reading model's config.json:
   - Check `model_type` field (e.g., "llama", "qwen2", "stable-diffusion", etc.)
   - Check `architectures` field
   - Check task in model card (text-generation, text2image, etc.)

2. **Generate appropriate demo script** in `{{DEMO_DIR}}/demo.py`:

### For Text Generation Models (Qwen, Llama, etc.)
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "{{MODEL_DIR}}"

def run_inference():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16,
        device_map="cuda", trust_remote_code=True
    )
    prompt = "Hello, I am"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50)
    print(tokenizer.decode(outputs[0]))

if __name__ == "__main__":
    run_inference()
```

### For Image Generation Models
```python
import torch
from diffusers import DiffusionPipeline

MODEL_PATH = "{{MODEL_DIR}}"

def run_inference():
    pipe = DiffusionPipeline.from_pretrained(MODEL_PATH, torch_dtype=torch.float16).to("cuda")
    image = pipe("A cat sitting on a couch").images[0]
    image.save("output.png")

if __name__ == "__main__":
    run_inference()
```

3. **Test the demo script**:
```bash
cd {{DEMO_DIR}}
python demo.py
```

4. Update progress.json

