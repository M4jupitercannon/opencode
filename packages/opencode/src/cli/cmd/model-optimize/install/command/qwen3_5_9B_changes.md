# Pipeline Changes: qwen/qwen3.5-9B on MI350X (gfx950)

Changes made relative to `model-analyzer.md` when running the analysis pipeline on `qwen/qwen3.5-9B` with `rocm/vllm-dev:nightly_main_20260308` (vLLM 0.17.0rc1, PyTorch 2.9.1, ROCm 7.3.0, 8x MI350X).

---

## 1. vLLM BlockSize patch

Qwen3.5-9B uses a hybrid `linear_attention` / `full_attention` architecture (24 linear + 8 full attention layers). vLLM aligns the attention block size to the mamba/linear-attention page size, computing `block_size=528`. This value is not in vLLM's `BlockSize` Literal type (`1, 8, 16, 32, 64, 128, 256`), causing all attention backends to reject it:

```
ValueError: No valid attention backend found for rocm with
  AttentionSelectorConfig(head_size=256, block_size=528, ...).
  Reasons: {TRITON_ATTN: [block_size not supported]}.
```

**Fix:** Patched `vllm/config/cache.py` inside the container:

```python
# Before
BlockSize = Literal[1, 8, 16, 32, 64, 128, 256]

# After
BlockSize = Literal[1, 8, 16, 32, 64, 128, 256, 528]
```

This is safe because 528 % 16 == 0, satisfying the Triton attention kernel's `MultipleOf(16)` alignment requirement. The `--block-size` CLI flag does not help because the mamba page alignment overrides user-specified values.

---

## 2. Reduced trace collection (30 prompts -> 3 prompts, 1024 -> 256 tokens)

The hybrid architecture has roughly 2x the operator count per layer (both linear attention SSM ops and full attention ops). With the original settings (`--num-prompts 30 --input-len 1024 --output-len 1024`), the trace exceeded 4GB compressed and took >42 minutes to flush, ultimately truncating on server kill.

**Fix:** Reduced to `--num-prompts 3 --input-len 256 --output-len 256`. This produced a 1.5GB trace with 94% shape coverage and 487K GPU kernels with External IDs -- sufficient for bottleneck analysis.

The `stop_profile` API call still takes ~15 minutes to serialize and compress the trace for this model. The server must not be killed until the trace file size stabilizes.

---

## 3. Missing co-located split script (trace splitting bug, now fixed)

The original `model-analyzer.md` Step 3 only copied `analyze_kernels.py` to `profile/`, but `analyze_kernels.py` resolves its bundled split script via `os.path.join(os.path.dirname(__file__), "split_vllm_trace_annotation.py")`. Since `split_vllm_trace_annotation.py` was only in `scripts/`, not `profile/`, the split subprocess failed and `analyze_kernels.py` silently fell back to full-trace-only analysis.

This was **not** a vLLM V1 annotation issue -- the V1 engine does emit `execute_context_X(Y)_generation_Z(W)` user_annotation events (510 found in the trace), and the regex patterns in the split script match them correctly.

**Fix:** Copy both scripts together:

```bash
cp <output_dir>/scripts/analyze_kernels.py <output_dir>/scripts/split_vllm_trace_annotation.py .
```

This fix has been applied to `model-analyzer.md`. After the fix, trace splitting produces:
- `prefilldecode_report/` -- prefill+decode phase TraceLens CSVs
- `decode_report/` -- decode-only phase TraceLens CSVs
- `bottlenecks.json` now generated from the decode phase (preferred for optimization targeting)

---

## 4. Helper scripts source path

The pipeline specifies copying scripts from `~/.config/opencode/scripts/`. This path does not exist on the host.

**Fix:** Copied from the source repository instead:

```
/home/ziwei/opencode/packages/opencode/src/cli/cmd/model-optimize/install/scripts/*.py
```

---

## 5. Extended model_config.json and model_shapes.json

The original pipeline records standard transformer fields. Qwen3.5-9B's hybrid architecture requires additional fields to capture the linear attention configuration.

**Added to model_config.json:**
- `layer_types` -- array of 32 entries (`"linear_attention"` or `"full_attention"`)
- `full_attention_interval` -- every 4th layer is full attention
- `architectures` -- `["Qwen3_5ForConditionalGeneration"]`

**Added to model_shapes.json:**
- `linear_key_head_dim: 128`
- `linear_value_head_dim: 128`
- `linear_num_key_heads: 16`
- `linear_num_value_heads: 32`

---

## 6. Bottleneck classifier additions

The original bottleneck classifier in Step 4 does not handle linear attention operators. Two additions:

- **"Linear Attention" category** for `ChunkGatedDeltaRuleFunction` (21.1% GPU time) and `FusedRecurrentFunction` (1.3% GPU time) -- these are the chunked delta-rule and fused recurrent kernels used by the linear attention layers.
- **"Memory op" for `aten::fill_`** (44.5% GPU time) -- dominates the profile; likely zeroing mamba/delta-rule state buffers between chunks. Classified as non-optimizable since it is a memory-bound buffer initialization.

---

## 7. CSV field size limit fix

TraceLens `ops_unique_args.csv` contains kernel detail strings that exceed Python's default CSV field size limit (131072 bytes). The hybrid architecture produces especially long entries due to the variety of sub-kernels per operator.

**Fix:** Added before CSV reading in the bottleneck generation script:

```python
import sys
csv.field_size_limit(sys.maxsize)
```

---

## Key Findings

| Metric | Value |
|--------|-------|
| Output throughput (baseline) | 1254 tok/s |
| Mean TPOT | 7.9ms |
| Mean TTFT | 4.3s |
| GPU utilization | 21.8% busy / 78.2% idle |
| Top bottleneck | `aten::fill_` at 44.5% GPU time |
| GEMM share | 0.3% (vs. 40-60% typical for standard transformers) |
| Trace size (3 prompts, 256 tokens) | 1.5GB compressed |

The model is dominated by linear attention kernels and buffer management, not GEMMs. The 78% idle time suggests significant scheduling overhead from the hybrid architecture.
