import { test, expect } from "bun:test"
import path from "path"
import { tmpdir } from "../fixture/fixture"
import { Instance } from "../../src/project/instance"
import { Provider } from "../../src/provider/provider"
import { Env } from "../../src/env"

test("amd-anthropic provider loaded from LLM_GATEWAY_KEY env variable", async () => {
  await using tmp = await tmpdir({
    init: async (dir) => {
      await Bun.write(
        path.join(dir, "opencode.json"),
        JSON.stringify({
          $schema: "https://opencode.ai/config.json",
        }),
      )
    },
  })
  await Instance.provide({
    directory: tmp.path,
    init: async () => {
      Env.set("LLM_GATEWAY_KEY", "test-gateway-key")
    },
    fn: async () => {
      const providers = await Provider.list()
      expect(providers["amd-anthropic"]).toBeDefined()
      expect(providers["amd-anthropic"].name).toBe("AMD Gateway (Anthropic)")
      expect(providers["amd-anthropic"].env).toContain("LLM_GATEWAY_KEY")
      // Check that models are inherited from anthropic
      expect(Object.keys(providers["amd-anthropic"].models).length).toBeGreaterThan(0)
      // Check that the first model has the correct API URL
      const firstModel = Object.values(providers["amd-anthropic"].models)[0]
      expect(firstModel.api.url).toBe("https://llm-api.amd.com/Anthropic")
      expect(firstModel.api.npm).toBe("@ai-sdk/anthropic")
    },
  })
})

test("amd-openai provider loaded from LLM_GATEWAY_KEY env variable", async () => {
  await using tmp = await tmpdir({
    init: async (dir) => {
      await Bun.write(
        path.join(dir, "opencode.json"),
        JSON.stringify({
          $schema: "https://opencode.ai/config.json",
        }),
      )
    },
  })
  await Instance.provide({
    directory: tmp.path,
    init: async () => {
      Env.set("LLM_GATEWAY_KEY", "test-gateway-key")
    },
    fn: async () => {
      const providers = await Provider.list()
      expect(providers["amd-openai"]).toBeDefined()
      expect(providers["amd-openai"].name).toBe("AMD Gateway (OpenAI)")
      expect(providers["amd-openai"].env).toContain("LLM_GATEWAY_KEY")
      // Check that models are inherited from openai
      expect(Object.keys(providers["amd-openai"].models).length).toBeGreaterThan(0)
      // Check that the first model has the correct API URL
      const firstModel = Object.values(providers["amd-openai"].models)[0]
      expect(firstModel.api.url).toBe("https://llm-api.amd.com/OpenAI")
      expect(firstModel.api.npm).toBe("@ai-sdk/openai")
    },
  })
})

test("amd-anthropic provider options include correct headers", async () => {
  await using tmp = await tmpdir({
    init: async (dir) => {
      await Bun.write(
        path.join(dir, "opencode.json"),
        JSON.stringify({
          $schema: "https://opencode.ai/config.json",
        }),
      )
    },
  })
  await Instance.provide({
    directory: tmp.path,
    init: async () => {
      Env.set("LLM_GATEWAY_KEY", "test-gateway-key")
    },
    fn: async () => {
      const providers = await Provider.list()
      expect(providers["amd-anthropic"]).toBeDefined()
      expect(providers["amd-anthropic"].options.baseURL).toBe("https://llm-api.amd.com/Anthropic")
      expect(providers["amd-anthropic"].options.apiKey).toBe("dummy")
      expect(providers["amd-anthropic"].options.headers["Ocp-Apim-Subscription-Key"]).toBe("test-gateway-key")
    },
  })
})

test("amd-openai provider options include correct headers", async () => {
  await using tmp = await tmpdir({
    init: async (dir) => {
      await Bun.write(
        path.join(dir, "opencode.json"),
        JSON.stringify({
          $schema: "https://opencode.ai/config.json",
        }),
      )
    },
  })
  await Instance.provide({
    directory: tmp.path,
    init: async () => {
      Env.set("LLM_GATEWAY_KEY", "test-gateway-key")
    },
    fn: async () => {
      const providers = await Provider.list()
      expect(providers["amd-openai"]).toBeDefined()
      expect(providers["amd-openai"].options.baseURL).toBe("https://llm-api.amd.com/OpenAI")
      expect(providers["amd-openai"].options.apiKey).toBe("dummy")
      expect(providers["amd-openai"].options.headers["Ocp-Apim-Subscription-Key"]).toBe("test-gateway-key")
    },
  })
})

test("amd gateway providers not loaded without LLM_GATEWAY_KEY", async () => {
  await using tmp = await tmpdir({
    init: async (dir) => {
      await Bun.write(
        path.join(dir, "opencode.json"),
        JSON.stringify({
          $schema: "https://opencode.ai/config.json",
        }),
      )
    },
  })
  await Instance.provide({
    directory: tmp.path,
    init: async () => {
      // Explicitly remove the key to ensure it's not set
      Env.remove("LLM_GATEWAY_KEY")
    },
    fn: async () => {
      const providers = await Provider.list()
      expect(providers["amd-anthropic"]).toBeUndefined()
      expect(providers["amd-openai"]).toBeUndefined()
    },
  })
})

test("amd-anthropic can be disabled via disabled_providers", async () => {
  await using tmp = await tmpdir({
    init: async (dir) => {
      await Bun.write(
        path.join(dir, "opencode.json"),
        JSON.stringify({
          $schema: "https://opencode.ai/config.json",
          disabled_providers: ["amd-anthropic"],
        }),
      )
    },
  })
  await Instance.provide({
    directory: tmp.path,
    init: async () => {
      Env.set("LLM_GATEWAY_KEY", "test-gateway-key")
    },
    fn: async () => {
      const providers = await Provider.list()
      expect(providers["amd-anthropic"]).toBeUndefined()
      expect(providers["amd-openai"]).toBeDefined()
    },
  })
})

test("amd-openai can be used as default model", async () => {
  await using tmp = await tmpdir({
    init: async (dir) => {
      await Bun.write(
        path.join(dir, "opencode.json"),
        JSON.stringify({
          $schema: "https://opencode.ai/config.json",
          model: "amd-openai/gpt-5",
        }),
      )
    },
  })
  await Instance.provide({
    directory: tmp.path,
    init: async () => {
      Env.set("LLM_GATEWAY_KEY", "test-gateway-key")
    },
    fn: async () => {
      const model = await Provider.defaultModel()
      expect(model.providerID).toBe("amd-openai")
      expect(model.modelID).toBe("gpt-5")
    },
  })
})

test("amd-anthropic models inherit capabilities from anthropic", async () => {
  await using tmp = await tmpdir({
    init: async (dir) => {
      await Bun.write(
        path.join(dir, "opencode.json"),
        JSON.stringify({
          $schema: "https://opencode.ai/config.json",
        }),
      )
    },
  })
  await Instance.provide({
    directory: tmp.path,
    init: async () => {
      Env.set("LLM_GATEWAY_KEY", "test-gateway-key")
    },
    fn: async () => {
      const providers = await Provider.list()
      const amdAnthropic = providers["amd-anthropic"]
      expect(amdAnthropic).toBeDefined()
      
      // Check that Claude models are available
      const models = Object.keys(amdAnthropic.models)
      const hasClaudeModel = models.some(m => m.includes("claude"))
      expect(hasClaudeModel).toBe(true)
      
      // Check that a Claude model has expected capabilities
      const claudeModel = Object.values(amdAnthropic.models).find(m => m.id.includes("claude"))
      if (claudeModel) {
        expect(claudeModel.capabilities.toolcall).toBe(true)
        expect(claudeModel.providerID).toBe("amd-anthropic")
      }
    },
  })
})

