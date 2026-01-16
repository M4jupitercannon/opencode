import { cmd } from "./cmd"
import { bootstrap } from "../bootstrap"
import { Server } from "../../server/server"
import { createOpencodeClient } from "@opencode-ai/sdk/v2"
import * as fs from "fs"
import * as path from "path"
import { UI } from "../ui"
import { Provider } from "../../provider/provider"

export const ModelOptimizeCommand = cmd({
  command: "model-optimize",
  describe: "end-to-end HuggingFace model optimization pipeline",
  builder: (yargs) =>
    yargs
      .option("model", {
        type: "string",
        alias: "m",
        describe: "HuggingFace model name (e.g., Qwen/Qwen3-8B)",
        demandOption: true,
      })
      .option("output", {
        type: "string",
        alias: "o",
        describe: "output directory for optimized model (default: ./model_opt_<model_name>)",
      })
      .option("llm", {
        type: "string",
        describe: "LLM model to use for optimization (e.g., opencode/glm-4.7-free)",
      })
      .option("skip-download", {
        type: "boolean",
        describe: "skip model download if already exists",
        default: false,
      }),
  async handler(args) {
    const llmArg = args.llm as string | undefined

    // Check if LLM_GATEWAY_KEY is set for amd providers
    const gatewayKey = process.env.LLM_GATEWAY_KEY
    const needsGatewayKey = !llmArg || llmArg.startsWith("amd-")
    if (needsGatewayKey && !gatewayKey) {
      UI.error("LLM_GATEWAY_KEY environment variable is not set")
      UI.println("Please set it with: export LLM_GATEWAY_KEY=your-api-key")
      UI.println("")
      UI.println("Or use a free model that doesn't require a key:")
      UI.println("  opencode model-optimize -m Qwen/Qwen3-8B --llm opencode/glm-4.7-free")
      process.exit(1)
    }

    const hfModel = args.model as string
    const modelName = hfModel.split("/").pop() || hfModel
    const modelNameSafe = modelName.replace(/[^a-zA-Z0-9_-]/g, "_")

    // Output directory
    const outputDir =
      (args.output as string) || path.resolve(`./model_opt_${modelNameSafe}`)

    UI.println("============================================")
    UI.println("Model Optimization Pipeline")
    UI.println("============================================")
    UI.println(`HuggingFace Model: ${hfModel}`)
    UI.println(`Output Directory:  ${outputDir}`)
    if (llmArg) {
      UI.println(`LLM Model:         ${llmArg}`)
    }
    UI.println("============================================")
    UI.println("")

    // Create output directory
    fs.mkdirSync(outputDir, { recursive: true })

    // Create subdirectories
    const dirs = {
      model: path.join(outputDir, "model"),
      demo: path.join(outputDir, "demo"),
      profile: path.join(outputDir, "profile"),
      problems: path.join(outputDir, "problems"),
      optimized: path.join(outputDir, "optimized"),
      report: path.join(outputDir, "report"),
    }
    Object.values(dirs).forEach((d) => fs.mkdirSync(d, { recursive: true }))

    // Create config file for the agent
    const configFile = path.join(outputDir, "config.json")
    const config = {
      hf_model: hfModel,
      model_name: modelName,
      dirs: dirs,
      created: new Date().toISOString(),
      skip_download: args["skip-download"],
    }
    fs.writeFileSync(configFile, JSON.stringify(config, null, 2))

    // Create progress tracker
    const progressFile = path.join(outputDir, "progress.json")
    const progress = {
      phase: "init",
      phases_completed: [] as string[],
      current_step: "",
      errors: [] as string[],
      optimizations: [] as { kernel: string; speedup: number }[],
      final_speedup: 0,
    }
    fs.writeFileSync(progressFile, JSON.stringify(progress, null, 2))

    // Create the main prompt for the agent
    const prompt = buildAgentPrompt(hfModel, modelName, outputDir, dirs, args["skip-download"] as boolean)

    // Create temporary .opencode config
    const opencodeDir = path.join(outputDir, ".opencode")
    const agentDir = path.join(opencodeDir, "agent")
    fs.mkdirSync(agentDir, { recursive: true })

    // Create agent config
    const agentConfig = buildAgentConfig(hfModel)
    fs.writeFileSync(path.join(agentDir, "model-opt.md"), agentConfig)

    // Create opencode.jsonc - use claude-opus-4-5 which is available
    const opencodeConfig = `{
  "$schema": "https://opencode.ai/config.json",
  "model": "amd-anthropic/claude-opus-4-5",
  "default_agent": "model-opt",
  "provider": {
    "amd-anthropic": {
      "options": {
        "timeout": 1200000
      }
    }
  },
  "permission": {
    "bash": "allow",
    "edit": "allow",
    "read": "allow",
    "write": "allow",
    "external_directory": "allow"
  }
}
`
    fs.writeFileSync(path.join(opencodeDir, "opencode.jsonc"), opencodeConfig)

    // Run opencode
    await bootstrap(outputDir, async () => {
      const server = Server.listen({ port: 0, hostname: "127.0.0.1" })
      const sdk = createOpencodeClient({ baseUrl: `http://${server.hostname}:${server.port}` })

      try {
        const sessionResult = await sdk.session.create()
        const sessionID = sessionResult.data?.id
        if (!sessionID) {
          UI.error("Failed to create session")
          process.exit(1)
        }

        // Subscribe to events
        const events = await sdk.event.subscribe()
        UI.println("Session created, sending prompt...")

        // Event processor
        const eventProcessor = (async () => {
          for await (const event of events.stream) {
            // Debug: show event types
            if (event.type !== "message.part.updated") {
              UI.println(UI.Style.TEXT_DIM + `[Event: ${event.type}]`)
            }
            if (event.type === "message.part.updated") {
              const part = event.properties.part
              if (part.sessionID !== sessionID) continue
              if (part.type === "tool" && part.state.status === "completed") {
                const title = part.state.title || JSON.stringify(part.state.input)
                UI.println(UI.Style.TEXT_INFO_BOLD + `|`, UI.Style.TEXT_DIM + ` ${part.tool.padEnd(7)}`, title)
                if (part.tool === "bash" && part.state.output?.trim()) {
                  UI.println()
                  UI.println(part.state.output)
                }
              }
              if (part.type === "text") {
                const textPart = part as any
                if (textPart.state?.done) {
                  UI.println(textPart.state.content)
                }
              }
            }
            if (event.type === "session.error") {
              UI.error(`Session error: ${JSON.stringify(event.properties || event)}`)
              break
            }
            if (event.type === "session.idle") {
              break
            }
          }
        })()

        // Send the prompt (uses model from opencode.jsonc if not specified)
        const modelParam = llmArg ? Provider.parseModel(llmArg) : undefined
        UI.println(`Sending prompt to LLM...`)
        await sdk.session.prompt({
          sessionID,
          model: modelParam,
          parts: [{ type: "text", text: prompt }],
        })
        UI.println("Prompt sent, waiting for completion...")

        // Wait for completion
        await eventProcessor
        UI.println("Event processor completed")
      } finally {
        // Cleanup opencode config (keep other files)
        try {
          fs.rmSync(opencodeDir, { recursive: true })
        } catch {
          // Ignore cleanup errors
        }
        server.stop()
      }
    })

    UI.println("")
    UI.println("============================================")
    UI.println("Model Optimization Pipeline Complete")
    UI.println("============================================")
    UI.println(`Output directory: ${outputDir}`)
    UI.println(`Report: ${path.join(dirs.report, "optimization_report.md")}`)
    UI.println("============================================")
  },
})

function buildAgentPrompt(
  hfModel: string,
  modelName: string,
  outputDir: string,
  dirs: Record<string, string>,
  skipDownload: boolean
): string {
  return `# End-to-End Model Optimization Pipeline

## Target Model
- **HuggingFace Model**: ${hfModel}
- **Model Name**: ${modelName}

## Output Directory Structure
\`\`\`
${outputDir}/
├── model/          # Downloaded model files
├── demo/           # Demo scripts for running the model
├── profile/        # Profiling results
├── problems/       # Kernel problems for optimization
├── optimized/      # Optimized kernels
├── report/         # Final optimization report
├── config.json     # Configuration
└── progress.json   # Progress tracking
\`\`\`

## IMPORTANT FILES
- **Config**: ${path.join(outputDir, "config.json")}
- **Progress**: ${path.join(outputDir, "progress.json")}

Update progress.json after completing each phase!

## YOUR TASK: Complete all 7 phases sequentially

---

# Phase 1: Model Download ${skipDownload ? "(SKIP if exists)" : ""}

## Goal
Download the HuggingFace model to \`${dirs.model}\`

## Steps
1. Check if model already exists (look for config.json in model dir)
2. ${skipDownload ? "If --skip-download is set AND model exists, skip to Phase 2" : "Download using huggingface_hub or transformers"}

\`\`\`python
from huggingface_hub import snapshot_download
# OR
from transformers import AutoModel, AutoTokenizer
\`\`\`

3. Update progress.json: phase="download", phases_completed.append("download")

---

# Phase 2: Generate Demo Script

## Goal
Create a working demo script that runs inference on the model.

## Steps
1. **Detect model type** by reading model's config.json:
   - Check \`model_type\` field (e.g., "llama", "qwen2", "stable-diffusion", etc.)
   - Check \`architectures\` field
   - Check task in model card (text-generation, text2image, etc.)

2. **Generate appropriate demo script** in \`${dirs.demo}/demo.py\`:

### For Text Generation Models (Qwen, Llama, etc.)
\`\`\`python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "${dirs.model}"

def run_inference():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=True
    )
    
    prompt = "Hello, I am"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50)
    
    print(tokenizer.decode(outputs[0]))

if __name__ == "__main__":
    run_inference()
\`\`\`

### For Image Generation Models
\`\`\`python
import torch
from diffusers import DiffusionPipeline

MODEL_PATH = "${dirs.model}"

def run_inference():
    pipe = DiffusionPipeline.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16
    ).to("cuda")
    
    prompt = "A cat sitting on a couch"
    image = pipe(prompt).images[0]
    image.save("output.png")

if __name__ == "__main__":
    run_inference()
\`\`\`

3. **Test the demo script**:
\`\`\`bash
cd ${dirs.demo}
python demo.py
\`\`\`

4. Update progress.json

---

# Phase 3: Fix Compatibility Issues

## Goal
If demo.py fails, diagnose and fix issues using monkey-patching (NO system library modifications).

## Common Issues & Fixes

### Issue: Missing custom operators
\`\`\`python
# Create patches/custom_ops.py
import torch

def my_custom_op(x):
    # Implement missing op
    return x

# Monkey patch
import transformers.models.xxx as module
module.custom_op = my_custom_op
\`\`\`

### Issue: Unsupported attention implementation
\`\`\`python
# patches/attention_fix.py
def patched_attention(self, query, key, value, **kwargs):
    # Fallback implementation
    return torch.nn.functional.scaled_dot_product_attention(query, key, value)

ModelClass._attention = patched_attention
\`\`\`

### Issue: RoPE/positional encoding errors
\`\`\`python
# patches/rope_fix.py
def fixed_rope(x, seq_len):
    # Fixed implementation
    ...
\`\`\`

## Steps
1. Run demo.py and capture the error
2. Analyze the error traceback
3. Create fix in \`${dirs.demo}/patches/\` directory
4. Update demo.py to import patches first
5. Re-run demo.py until it works
6. Update progress.json

---

# Phase 4: Performance Profiling

## Goal
Profile the model to identify bottleneck operators/kernels.

## Create Profiling Script: \`${dirs.profile}/profile_model.py\`

\`\`\`python
import torch
from torch.profiler import profile, ProfilerActivity, schedule
import json
import sys
sys.path.insert(0, "${dirs.demo}")

# Import the demo's model loading code
from demo import MODEL_PATH  # Adjust based on actual demo.py structure

def profile_model():
    # Load model (reuse demo.py logic)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, device_map="cuda", trust_remote_code=True
    )
    
    prompt = "Hello, I am a language model"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    # Warmup
    for _ in range(3):
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=10)
    
    # Profile
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=20)
    
    # Export results
    prof.export_chrome_trace("${dirs.profile}/trace.json")
    
    # Print top operators
    print("\\n=== Top CUDA Operators by Time ===")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    
    # Save summary
    summary = prof.key_averages().table(sort_by="cuda_time_total", row_limit=50)
    with open("${dirs.profile}/operator_summary.txt", "w") as f:
        f.write(summary)
    
    # Extract bottleneck operators
    bottlenecks = []
    for evt in prof.key_averages():
        if evt.cuda_time_total > 0:
            bottlenecks.append({
                "name": evt.key,
                "cuda_time_ms": evt.cuda_time_total / 1000,
                "cuda_time_percent": 0,  # Calculate later
                "count": evt.count,
                "input_shapes": str(evt.input_shapes) if evt.input_shapes else ""
            })
    
    # Sort by time and calculate percentages
    total_cuda_time = sum(b["cuda_time_ms"] for b in bottlenecks)
    bottlenecks = sorted(bottlenecks, key=lambda x: x["cuda_time_ms"], reverse=True)[:20]
    for b in bottlenecks:
        b["cuda_time_percent"] = (b["cuda_time_ms"] / total_cuda_time * 100) if total_cuda_time > 0 else 0
    
    with open("${dirs.profile}/bottlenecks.json", "w") as f:
        json.dump(bottlenecks, f, indent=2)
    
    print("\\n=== Top 10 Bottleneck Operators ===")
    for i, b in enumerate(bottlenecks[:10], 1):
        print(f"{i}. {b['name']}: {b['cuda_time_ms']:.2f}ms ({b['cuda_time_percent']:.1f}%)")
    
    return bottlenecks

if __name__ == "__main__":
    profile_model()
\`\`\`

## Additional: AMD ROCm Profiling (if available)
\`\`\`bash
# Check if rocprof is available
which rocprof && rocprof --stats python ${dirs.demo}/demo.py
\`\`\`

## Steps
1. Create and run the profiling script
2. Analyze bottlenecks.json to identify top time-consuming operators
3. Focus on operators that take > 5% of total CUDA time
4. Update progress.json with bottleneck list

---

# Phase 5: Generate Problem Files for Kernel Optimization

## Goal
Convert bottleneck operators into Problem files for kernel-optimize.

## Problem File Format
Each problem file in \`${dirs.problems}/\` must have:

\`\`\`python
# problem_<operator_name>.py
import torch
import torch.nn as nn

class Model(nn.Module):
    """Reference implementation using PyTorch."""
    def __init__(self):
        super().__init__()
        # Initialize any needed parameters
    
    def forward(self, *inputs):
        # PyTorch implementation of the operator
        return output

# Define typical input shapes from profiling
batch_size = 1
seq_len = 512
hidden_size = 4096

def get_inputs():
    """Return list of input tensors with typical shapes."""
    return [torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float16, device='cuda')]

def get_init_inputs():
    """Return list of arguments for Model.__init__"""
    return []
\`\`\`

## Common Operators to Optimize

### 1. Linear/GEMM
\`\`\`python
# problem_linear.py
class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features, dtype=torch.float16))
    
    def forward(self, x):
        return x @ self.weight.T
\`\`\`

### 2. Attention (Self-Attention)
\`\`\`python
# problem_attention.py
class Model(nn.Module):
    def forward(self, q, k, v):
        return torch.nn.functional.scaled_dot_product_attention(q, k, v)
\`\`\`

### 3. RMSNorm / LayerNorm
\`\`\`python
# problem_rmsnorm.py
class Model(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.float16))
        self.eps = 1e-6
    
    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x
\`\`\`

### 4. SiLU/SwiGLU/GELU
\`\`\`python
# problem_activation.py
class Model(nn.Module):
    def forward(self, x, gate):
        return torch.nn.functional.silu(gate) * x
\`\`\`

### 5. Rotary Position Embedding (RoPE)
\`\`\`python
# problem_rope.py
class Model(nn.Module):
    def forward(self, x, cos, sin):
        # RoPE implementation
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
\`\`\`

## Steps
1. Read bottlenecks.json
2. For each major bottleneck (>5% time), create a Problem file
3. Use shapes from profiling data
4. Ensure get_inputs() returns realistic input tensors
5. Update progress.json with list of created problems

---

# Phase 6: Run Kernel Optimization

## Goal
Optimize each bottleneck kernel using kernel-optimize.

## Steps
For each problem file in \`${dirs.problems}/\`:

\`\`\`bash
cd ${dirs.problems}

# Run kernel-optimize for each problem
opencode kernel-optimize --src problem_linear.py --goal 1.5
opencode kernel-optimize --src problem_attention.py --goal 1.2
# ... etc
\`\`\`

The optimized kernels will be saved as \`problem_<name>_opt.py\`.

## After Optimization
1. Copy optimized kernels to \`${dirs.optimized}/\`
2. Record speedup for each kernel in progress.json
3. Note which kernels failed to optimize

---

# Phase 7: Integration & Final Testing

## Goal
Integrate optimized kernels into the model using monkey-patching.

## Create Integration Script: \`${dirs.optimized}/integrate.py\`

\`\`\`python
"""
Monkey-patch optimized Triton kernels into the model.
"""
import torch
import sys

# Import optimized kernels
from problem_linear_opt import ModelNew as OptimizedLinear
from problem_attention_opt import ModelNew as OptimizedAttention
# ... import other optimized kernels

# Create instances
_opt_linear = OptimizedLinear()
_opt_attention = OptimizedAttention()

def patch_model(model):
    """Apply monkey-patches to replace slow operators with optimized versions."""
    
    # Example: Replace all Linear layers
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # Create a patched forward method
            original_forward = module.forward
            def make_optimized_forward(mod):
                def optimized_forward(x):
                    # Use optimized kernel
                    weight = mod.weight
                    if mod.bias is not None:
                        return _opt_linear.forward(x, weight) + mod.bias
                    return _opt_linear.forward(x, weight)
                return optimized_forward
            module.forward = make_optimized_forward(module)
    
    return model
\`\`\`

## Create Test Script: \`${dirs.optimized}/test_integration.py\`

\`\`\`python
"""
Test optimized model for correctness and performance.
"""
import torch
import time
import sys
sys.path.insert(0, "${dirs.demo}")
sys.path.insert(0, "${dirs.optimized}")

from transformers import AutoModelForCausalLM, AutoTokenizer
from integrate import patch_model

MODEL_PATH = "${dirs.model}"

def test_integration():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    
    # Load original model
    model_original = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, device_map="cuda", trust_remote_code=True
    )
    
    # Load and patch optimized model
    model_optimized = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, device_map="cuda", trust_remote_code=True
    )
    model_optimized = patch_model(model_optimized)
    
    prompt = "The future of artificial intelligence is"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    # Test correctness
    print("=== Testing Correctness ===")
    with torch.no_grad():
        out_original = model_original(**inputs)
        out_optimized = model_optimized(**inputs)
    
    logits_diff = (out_original.logits - out_optimized.logits).abs().max().item()
    print(f"Max logits difference: {logits_diff:.6f}")
    print(f"Correctness: {'PASSED' if logits_diff < 0.1 else 'FAILED'}")
    
    # Test performance
    print("\\n=== Testing Performance ===")
    
    # Warmup
    for _ in range(5):
        with torch.no_grad():
            model_original.generate(**inputs, max_new_tokens=10)
            model_optimized.generate(**inputs, max_new_tokens=10)
    torch.cuda.synchronize()
    
    # Benchmark original
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        with torch.no_grad():
            model_original.generate(**inputs, max_new_tokens=20)
    torch.cuda.synchronize()
    t_original = (time.perf_counter() - t0) / 10
    
    # Benchmark optimized
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        with torch.no_grad():
            model_optimized.generate(**inputs, max_new_tokens=20)
    torch.cuda.synchronize()
    t_optimized = (time.perf_counter() - t0) / 10
    
    speedup = t_original / t_optimized
    
    print(f"Original:  {t_original*1000:.2f} ms")
    print(f"Optimized: {t_optimized*1000:.2f} ms")
    print(f"Speedup:   {speedup:.2f}x")
    
    return {
        "correctness": logits_diff < 0.1,
        "original_ms": t_original * 1000,
        "optimized_ms": t_optimized * 1000,
        "speedup": speedup
    }

if __name__ == "__main__":
    results = test_integration()
    
    # Save results
    import json
    with open("${dirs.report}/integration_results.json", "w") as f:
        json.dump(results, f, indent=2)
\`\`\`

## Steps
1. Create integrate.py with monkey-patches for optimized kernels
2. Create test_integration.py
3. Run integration tests
4. If correctness fails, debug and fix
5. Record final speedup
6. Update progress.json

---

# Phase 8: Generate Final Report

## Goal
Create a comprehensive optimization report.

## Create Report: \`${dirs.report}/optimization_report.md\`

\`\`\`markdown
# Model Optimization Report

## Model Information
- **Model**: ${hfModel}
- **Optimization Date**: [DATE]

## Summary
- **Total Optimization Time**: X hours
- **Final Speedup**: X.Xx
- **Kernels Optimized**: N

## Bottleneck Analysis

| Operator | Original Time (ms) | % of Total | Optimized | Speedup |
|----------|-------------------|------------|-----------|---------|
| Linear   | XX.X              | XX%        | Yes       | X.Xx    |
| Attention| XX.X              | XX%        | Yes       | X.Xx    |
| ...      | ...               | ...        | ...       | ...     |

## Optimized Kernels

### 1. Linear (GEMM)
- **Problem File**: problem_linear.py
- **Optimization Techniques**: [list techniques]
- **Speedup**: X.Xx

### 2. Attention
- **Problem File**: problem_attention.py
- **Optimization Techniques**: [list techniques]
- **Speedup**: X.Xx

## Integration Details

### Monkey Patches Applied
1. Replaced Linear layers with Triton GEMM
2. Replaced attention with Flash Attention implementation
3. ...

### Correctness Verification
- Max logits difference: X.XXXXXX
- Status: PASSED/FAILED

## Performance Results

| Metric | Original | Optimized | Improvement |
|--------|----------|-----------|-------------|
| Inference Time (ms) | XX.X | XX.X | X.Xx |
| Memory Usage (GB) | X.X | X.X | X.Xx |

## Files Generated

\\\`\\\`\\\`
${outputDir}/
├── model/              # Downloaded model
├── demo/
│   ├── demo.py         # Working demo script
│   └── patches/        # Compatibility patches
├── profile/
│   ├── trace.json      # Chrome trace
│   ├── bottlenecks.json
│   └── operator_summary.txt
├── problems/
│   ├── problem_linear.py
│   ├── problem_linear_opt.py
│   └── ...
├── optimized/
│   ├── integrate.py    # Integration script
│   └── test_integration.py
└── report/
    ├── optimization_report.md
    └── integration_results.json
\\\`\\\`\\\`

## Recommendations for Further Optimization
1. ...
2. ...

## Known Issues
1. ...
\`\`\`

## Steps
1. Gather all results from previous phases
2. Generate the comprehensive report
3. Update progress.json: phase="complete", phases_completed.append("report")

---

# EXECUTION INSTRUCTIONS

1. **Execute phases in order**: 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8
2. **Update progress.json after each phase**
3. **If a phase fails, debug and fix before proceeding**
4. **For kernel-optimize, use the existing opencode command**
5. **All monkey patches go in ${dirs.demo}/patches/ or ${dirs.optimized}/**
6. **NEVER modify system libraries - only use monkey patching**

## Start Now
Begin with Phase 1: Model Download
`
}

function buildAgentConfig(hfModel: string): string {
  return `---
model: amd-anthropic/claude-opus-4-5
temperature: 0.2
steps: 100
---

# Model Optimization Expert

You are an expert in end-to-end deep learning model optimization. Your task is to optimize the HuggingFace model "${hfModel}" for maximum inference performance.

## Core Competencies

1. **Model Understanding**: Analyze transformer architectures, identify compute patterns
2. **Profiling**: Use PyTorch Profiler and AMD ROCm tools to identify bottlenecks
3. **Kernel Development**: Write high-performance Triton kernels
4. **Integration**: Safely integrate optimizations via monkey-patching

## Key Principles

1. **Correctness First**: Never sacrifice correctness for speed
2. **Systematic Approach**: Follow the phases in order, update progress
3. **No System Modifications**: Use monkey-patching, never modify installed packages
4. **Document Everything**: Keep detailed notes in progress.json and final report

## Workflow

Execute each phase completely before moving to the next:

1. Download model
2. Generate and test demo script
3. Fix compatibility issues (if any)
4. Profile and identify bottlenecks
5. Generate Problem files for top bottlenecks
6. Run kernel-optimize on each Problem
7. Integrate optimized kernels via monkey-patching
8. Generate final report

## Tools You'll Use

- \`huggingface_hub\` / \`transformers\`: Model download and loading
- \`torch.profiler\`: Performance profiling
- \`opencode kernel-optimize\`: Kernel optimization
- Python: Scripting, monkey-patching, testing

## Error Handling

- If model download fails: Check model name, authentication
- If demo fails: Analyze error, create monkey-patch fix
- If profiling fails: Simplify the test case
- If kernel optimization fails: Document and skip that kernel
- If integration fails: Debug monkey-patch, test incrementally

Start by reading the task prompt and executing Phase 1.
`
}

