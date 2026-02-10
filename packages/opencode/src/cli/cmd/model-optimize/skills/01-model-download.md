# Phase 1: Model Download {{SKIP_DOWNLOAD_LABEL}} {{SKIP_LABEL}}

## Goal
Download the HuggingFace model to `{{MODEL_DIR}}`

## Steps
1. Check if model already exists (look for config.json in model dir)
2. Download using huggingface_hub or transformers

```python
from huggingface_hub import snapshot_download
# OR
from transformers import AutoModel, AutoTokenizer
```

3. Update progress.json: phase="download", phases_completed.append("download")

