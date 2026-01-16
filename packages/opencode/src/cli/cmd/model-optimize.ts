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
      })
      .option("resume", {
        type: "boolean",
        describe: "resume from last completed phase in existing project",
        default: false,
      })
      .option("from-phase", {
        type: "string",
        alias: "f",
        describe: "start from specific phase (demo, profile, problems, optimize, integrate, report)",
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

    // Handle resume / from-phase options
    const resumeMode = args.resume as boolean
    const fromPhase = args["from-phase"] as string | undefined
    const progressFile = path.join(outputDir, "progress.json")
    
    let existingProgress: any = null
    let startPhase = "download"
    
    if (resumeMode || fromPhase) {
      // Check for existing progress
      if (fs.existsSync(progressFile)) {
        try {
          existingProgress = JSON.parse(fs.readFileSync(progressFile, "utf-8"))
          UI.println(UI.Style.TEXT_INFO_BOLD + "Found existing project progress")
          
          if (fromPhase) {
            // Validate phase name
            const validPhases = ["download", "demo", "compatibility", "profile", "problems", "optimize", "integrate", "report"]
            if (!validPhases.includes(fromPhase)) {
              UI.error(`Invalid phase: ${fromPhase}. Valid phases: ${validPhases.join(", ")}`)
              process.exit(1)
            }
            startPhase = fromPhase
            UI.println(`Starting from phase: ${startPhase}`)
          } else if (resumeMode && existingProgress.phases_completed) {
            // Resume from last completed phase
            const phasesOrder = ["download", "demo", "compatibility", "profile", "problems", "optimize", "integrate", "report"]
            const completed = existingProgress.phases_completed as string[]
            for (let i = phasesOrder.length - 1; i >= 0; i--) {
              if (completed.includes(phasesOrder[i])) {
                startPhase = phasesOrder[i + 1] || "report"
                break
              }
            }
            UI.println(`Resuming from phase: ${startPhase}`)
          }
        } catch (e) {
          UI.println(UI.Style.TEXT_WARNING + "Could not parse existing progress.json, starting fresh")
        }
      } else {
        UI.println(UI.Style.TEXT_WARNING + "No existing progress.json found, starting from beginning")
      }
    }

    // Create config file for the agent
    const configFile = path.join(outputDir, "config.json")
    const config = {
      hf_model: hfModel,
      model_name: modelName,
      dirs: dirs,
      created: new Date().toISOString(),
      skip_download: args["skip-download"],
      start_phase: startPhase,
      resume_mode: resumeMode || !!fromPhase,
    }
    fs.writeFileSync(configFile, JSON.stringify(config, null, 2))

    // Create or update progress tracker
    const progress = existingProgress || {
      phase: "init",
      phases_completed: [] as string[],
      current_step: "",
      errors: [] as string[],
      optimizations: [] as { kernel: string; speedup: number }[],
      final_speedup: 0,
    }
    if (!existingProgress) {
      fs.writeFileSync(progressFile, JSON.stringify(progress, null, 2))
    }

    // Create the main prompt for the agent
    const prompt = buildAgentPrompt(hfModel, modelName, outputDir, dirs, args["skip-download"] as boolean, startPhase, existingProgress)

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

        // Send the prompt
        // If using AMD gateway (LLM_GATEWAY_KEY set) and no --llm specified, default to claude-opus-4-5
        let modelParam
        if (llmArg) {
          modelParam = Provider.parseModel(llmArg)
        } else if (gatewayKey) {
          // Default to claude-opus-4-5 for AMD gateway
          modelParam = Provider.parseModel("amd-anthropic/claude-opus-4-5")
          UI.println(`Using default AMD gateway model: amd-anthropic/claude-opus-4-5`)
        }
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
  skipDownload: boolean,
  startPhase: string = "download",
  existingProgress: any = null
): string {
  // Build resume context if applicable
  const resumeContext = startPhase !== "download" ? `
## RESUME MODE ACTIVE
**Starting from Phase: ${startPhase}**
${existingProgress ? `
### Previous Progress
- Phases completed: ${existingProgress.phases_completed?.join(", ") || "none"}
- Previous optimizations: ${JSON.stringify(existingProgress.details?.optimization?.kernels_optimized || [], null, 2)}
` : ""}

**IMPORTANT**: Skip phases before "${startPhase}" - their artifacts already exist.
Review existing files before proceeding to understand current state.

---
` : ""

  return `# End-to-End Model Optimization Pipeline

## Target Model
- **HuggingFace Model**: ${hfModel}
- **Model Name**: ${modelName}
${resumeContext}
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

## YOUR TASK: Complete phases starting from "${startPhase}"

---

# Phase 1: Model Download ${skipDownload ? "(SKIP if exists)" : ""} ${startPhase !== "download" ? "[SKIP - ALREADY DONE]" : ""}

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

# Phase 2: Generate Demo Script ${startPhase === "download" ? "" : (startPhase === "demo" ? "" : "[SKIP - ALREADY DONE]")}

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

# Phase 3: Fix Compatibility Issues ${["download", "demo"].includes(startPhase) ? "" : (startPhase === "compatibility" ? "" : "[SKIP - ALREADY DONE]")}

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

# Phase 4: Performance Profiling ${["download", "demo", "compatibility"].includes(startPhase) ? "" : (startPhase === "profile" ? "" : "[SKIP - ALREADY DONE]")}

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

# Phase 5: Generate Problem Files for Kernel Optimization ${["download", "demo", "compatibility", "profile"].includes(startPhase) ? "" : "[SKIP - ALREADY DONE]"}

## Goal
Convert bottleneck operators into Problem files for kernel-optimize.
**IMPORTANT**: Analyze operators for fusion opportunities BEFORE creating individual problem files.

## STEP 1: Operator Fusion Analysis (CRITICAL)

Before creating individual problem files, analyze the profiling data for **fusable operator patterns**:

### Common Fusion Opportunities in LLMs

| Pattern | Operators to Fuse | Fused Name | Expected Speedup |
|---------|-------------------|------------|------------------|
| **ResidualNorm** | add + rmsnorm/layernorm | fused_residual_norm | 1.2-1.5x |
| **SwiGLU/GeGLU** | silu/gelu + mul | fused_swiglu | 1.3-1.8x |
| **BiasAdd** | matmul + add (bias) | fused_linear_bias | 1.1-1.3x |
| **RotaryEmbed** | rope_cos + rope_sin + cat | fused_rope | 1.2-1.5x |
| **QKV Projection** | 3x linear (q,k,v) | fused_qkv_proj | 1.2-1.4x |
| **MLP Block** | linear + activation + linear | fused_mlp | 1.3-2.0x |

### Fusion Detection Script: \`${dirs.profile}/analyze_fusion.py\`

\`\`\`python
"""Analyze operator patterns for fusion opportunities."""
import json

def analyze_fusion_opportunities(bottlenecks_file):
    with open(bottlenecks_file) as f:
        bottlenecks = json.load(f)
    
    # Extract operator names and percentages
    ops = [(b["name"], b["cuda_time_percent"]) for b in bottlenecks]
    
    fusion_opportunities = []
    
    # Check for residual + norm pattern
    has_add = any("add" in op[0].lower() for op in ops)
    has_norm = any("norm" in op[0].lower() or "mean" in op[0].lower() for op in ops)
    if has_add and has_norm:
        add_pct = sum(op[1] for op in ops if "add" in op[0].lower())
        norm_pct = sum(op[1] for op in ops if "norm" in op[0].lower() or "mean" in op[0].lower())
        fusion_opportunities.append({
            "name": "fused_residual_rmsnorm",
            "operators": ["aten::add", "RMSNorm (aten::mean, aten::rsqrt, aten::mul)"],
            "combined_percent": add_pct + norm_pct,
            "expected_speedup": "1.3-1.5x",
            "priority": "HIGH" if add_pct + norm_pct > 10 else "MEDIUM"
        })
    
    # Check for SiLU + mul pattern (SwiGLU)
    has_silu = any("silu" in op[0].lower() for op in ops)
    has_mul = any("mul" in op[0].lower() and "norm" not in op[0].lower() for op in ops)
    if has_silu and has_mul:
        fusion_opportunities.append({
            "name": "fused_swiglu",
            "operators": ["aten::silu", "aten::mul"],
            "combined_percent": sum(op[1] for op in ops if "silu" in op[0].lower() or ("mul" in op[0].lower() and "norm" not in op[0].lower())),
            "expected_speedup": "1.3-1.8x",
            "priority": "MEDIUM"
        })
    
    # Check for consecutive linear layers (QKV projection)
    mm_count = sum(1 for op in ops if "mm" in op[0].lower() or "linear" in op[0].lower())
    if mm_count >= 3:
        fusion_opportunities.append({
            "name": "fused_qkv_proj",
            "operators": ["3x aten::mm for Q, K, V"],
            "combined_percent": sum(op[1] for op in ops if "mm" in op[0].lower()) / 3 * 1.5,
            "expected_speedup": "1.2-1.4x (batch the projections)",
            "priority": "LOW"  # rocBLAS already fast
        })
    
    print("\\n=== Fusion Opportunities ===")
    for f in sorted(fusion_opportunities, key=lambda x: x["combined_percent"], reverse=True):
        print(f"\\n{f['name']} [{f['priority']}]")
        print(f"  Operators: {', '.join(f['operators'])}")
        print(f"  Combined time: {f['combined_percent']:.1f}%")
        print(f"  Expected speedup: {f['expected_speedup']}")
    
    # Save to file
    with open("${dirs.profile}/fusion_opportunities.json", "w") as f:
        json.dump(fusion_opportunities, f, indent=2)
    
    return fusion_opportunities

if __name__ == "__main__":
    analyze_fusion_opportunities("${dirs.profile}/bottlenecks.json")
\`\`\`

### Run Fusion Analysis FIRST
\`\`\`bash
cd ${dirs.profile}
python analyze_fusion.py
cat fusion_opportunities.json
\`\`\`

## STEP 2: Create FUSED Problem Files (Priority)

**Create fused kernels BEFORE individual kernels!**

### Example: Fused Residual + RMSNorm
\`\`\`python
# problem_fused_residual_rmsnorm.py
import torch
import torch.nn as nn

class Model(nn.Module):
    """Fused residual add + RMSNorm for LLM transformer layers."""
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.float16))
        self.eps = eps
    
    def forward(self, hidden_states, residual):
        # Fused: hidden = RMSNorm(hidden_states + residual)
        hidden_states = hidden_states + residual
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states, hidden_states  # Return both normalized and pre-norm for next residual

# Typical shapes for Qwen3-8B
batch_size = 1
seq_len = 512
hidden_size = 4096

def get_inputs():
    return [
        torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float16, device='cuda'),
        torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float16, device='cuda'),
    ]

def get_init_inputs():
    return [hidden_size]
\`\`\`

### Example: Fused SwiGLU
\`\`\`python
# problem_fused_swiglu.py
import torch
import torch.nn as nn

class Model(nn.Module):
    """Fused SiLU(gate) * up for SwiGLU MLP."""
    def forward(self, gate, up):
        # Fused: silu(gate) * up
        return torch.nn.functional.silu(gate) * up

batch_size = 1
seq_len = 512
intermediate_size = 11008  # Qwen3-8B intermediate

def get_inputs():
    return [
        torch.randn(batch_size, seq_len, intermediate_size, dtype=torch.float16, device='cuda'),
        torch.randn(batch_size, seq_len, intermediate_size, dtype=torch.float16, device='cuda'),
    ]

def get_init_inputs():
    return []
\`\`\`

## STEP 3: Create Individual Problem Files (Lower Priority)

Only create individual problem files for operators that:
1. Cannot be fused with neighbors
2. Take > 5% of total time individually
3. Are not already optimized by vendor libraries (e.g., rocBLAS GEMM)

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

# Phase 6: Run Kernel Optimization ${["download", "demo", "compatibility", "profile", "problems"].includes(startPhase) ? "" : "[SKIP - ALREADY DONE]"}

## Goal
Optimize each bottleneck kernel using kernel-optimize.

## IMPORTANT: Prioritize Fused Kernels
Optimize fused kernels FIRST as they provide the highest speedup potential:

\`\`\`bash
cd ${dirs.problems}

# 1. FIRST: Optimize fused kernels (highest priority)
opencode kernel-optimize --src problem_fused_residual_rmsnorm.py --goal 1.5
opencode kernel-optimize --src problem_fused_swiglu.py --goal 1.5

# 2. THEN: Optimize remaining individual kernels (if not already done)
opencode kernel-optimize --src problem_rope.py --goal 1.3
# Skip GEMM/Linear if rocBLAS is already fast
# Skip simple elementwise ops (add, mul) - fusion handles these
\`\`\`

## Decision: When to Skip Individual Kernel Optimization
- **SKIP** if operator is part of a fused kernel you already optimized
- **SKIP** GEMM/Linear if profiling shows rocBLAS is already near-optimal (speedup < 1.1x)
- **SKIP** simple elementwise (add, copy) - overhead of custom kernel exceeds benefit

The optimized kernels will be saved as \`problem_<name>_opt.py\`.

## After Optimization
1. Copy **successfully optimized** kernels (speedup > 1.0x) to \`${dirs.optimized}/\`
2. Record speedup for each kernel in progress.json
3. Note which kernels failed or were skipped

---

# Phase 7: Integration & Final Testing ${["download", "demo", "compatibility", "profile", "problems", "optimize"].includes(startPhase) ? "" : "[SKIP - ALREADY DONE]"}

## Goal
Integrate optimized kernels into the model using monkey-patching.

## IMPORTANT: Integration Strategy for Fused Kernels

Fused kernels require careful integration as they replace MULTIPLE operations:

### Fused Residual + RMSNorm Integration Pattern
\`\`\`python
# Locate the RMSNorm class in transformers
# Original pattern in forward():
#   residual = hidden_states
#   hidden_states = self.input_layernorm(hidden_states)
#
# Replace with fused version:
#   hidden_states, residual = fused_residual_rmsnorm(hidden_states, residual)
\`\`\`

## Create Integration Script: \`${dirs.optimized}/integrate.py\`

\`\`\`python
"""
Monkey-patch optimized Triton kernels into the model.
Supports both individual and fused kernels.
"""
import torch
import sys
import os

# Add paths
sys.path.insert(0, "${dirs.optimized}")
sys.path.insert(0, "${dirs.problems}")

# Import optimized kernels (check which ones exist)
_optimized_kernels = {}

def try_import(name, module_name):
    try:
        mod = __import__(module_name)
        _optimized_kernels[name] = mod.ModelNew()
        print(f"Loaded optimized kernel: {name}")
        return True
    except ImportError as e:
        print(f"Skipping {name}: {e}")
        return False

# Try to import fused kernels first
try_import("fused_residual_rmsnorm", "problem_fused_residual_rmsnorm_opt")
try_import("fused_swiglu", "problem_fused_swiglu_opt")

# Then individual kernels
try_import("rmsnorm", "problem_rmsnorm_opt")
try_import("rope", "problem_rope_opt")

def patch_rmsnorm_layers(model):
    """Patch RMSNorm layers with optimized version."""
    if "rmsnorm" not in _optimized_kernels and "fused_residual_rmsnorm" not in _optimized_kernels:
        return 0
    
    patched = 0
    opt_kernel = _optimized_kernels.get("rmsnorm")
    
    for name, module in model.named_modules():
        # Match various RMSNorm implementations
        class_name = module.__class__.__name__
        if "RMSNorm" in class_name or "Qwen3RMSNorm" in class_name:
            original_forward = module.forward
            weight = module.weight
            eps = getattr(module, 'variance_epsilon', getattr(module, 'eps', 1e-6))
            
            def make_opt_forward(w, e):
                def opt_forward(hidden_states):
                    return opt_kernel.forward(hidden_states)
                return opt_forward
            
            if opt_kernel:
                module.forward = make_opt_forward(weight, eps)
                patched += 1
    
    return patched

def patch_rope(model):
    """Patch RoPE implementation with optimized version."""
    if "rope" not in _optimized_kernels:
        return False
    
    # Find and patch the rotary embedding function
    # This varies by model architecture
    return True

def patch_model(model):
    """Apply all monkey-patches to the model."""
    stats = {
        "rmsnorm_layers": patch_rmsnorm_layers(model),
        "rope": patch_rope(model),
    }
    print(f"Patching complete: {stats}")
    return model, stats
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

# Phase 8: Generate Final Report ${startPhase === "report" ? "" : (["download", "demo", "compatibility", "profile", "problems", "optimize", "integrate"].includes(startPhase) ? "" : "[SKIP - ALREADY DONE]")}

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

${startPhase === "download" ? `
## Fresh Start Mode
1. **Execute phases in order**: 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8
2. Begin with Phase 1: Model Download
` : `
## Resume Mode Active - Starting from "${startPhase}"
1. **Skip phases before "${startPhase}"** - their artifacts already exist
2. **Review existing files first** to understand current state
3. **Continue from Phase: ${startPhase}**

### Quick Start Checklist
- [ ] Read existing progress.json
- [ ] Verify artifacts from previous phases exist
- [ ] Start working on Phase: ${startPhase}
`}

## General Rules
1. **Update progress.json after each phase**
2. **If a phase fails, debug and fix before proceeding**
3. **For kernel-optimize, use the existing opencode command** (it inherits LLM_GATEWAY_KEY from the environment)
4. **All monkey patches go in ${dirs.demo}/patches/ or ${dirs.optimized}/**
5. **NEVER modify system libraries - only use monkey patching**

## Optimization Priority (Phase 5-6)
1. **FIRST**: Create and optimize FUSED kernels (residual+norm, swiglu)
2. **THEN**: Optimize remaining individual kernels
3. **SKIP**: Operators already optimized by vendor libs (rocBLAS GEMM)
4. **SKIP**: Simple elementwise ops covered by fused kernels

## Start Now
Begin with Phase: ${startPhase.charAt(0).toUpperCase() + startPhase.slice(1)}
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

