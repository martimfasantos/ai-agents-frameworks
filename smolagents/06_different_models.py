from smolagents import CodeAgent, OpenAIModel, LiteLLMModel

from settings import settings

"""
-------------------------------------------------------
In this example, we explore smolagents with different model providers:

- OpenAIModel for OpenAI API (and compatible endpoints)
- LiteLLMModel for 100+ providers via LiteLLM proxy
- Swapping models without changing agent code
- Comparing responses from different providers

smolagents supports multiple model backends including
OpenAI, Hugging Face Inference, LiteLLM, Anthropic (via
LiteLLM), local models (Transformers, vLLM, MLX), and more.

For more details, visit:
https://huggingface.co/docs/smolagents/reference/models
-------------------------------------------------------
"""

# --- 1. OpenAI Model (direct) ---
print("=== Different Models Demo ===\n")

openai_model = OpenAIModel(
    model_id=settings.OPENAI_MODEL_NAME,
    api_key=settings.OPENAI_API_KEY.get_secret_value(),
)

agent_openai = CodeAgent(
    tools=[],
    model=openai_model,
    max_steps=2,
)

print("--- Model 1: OpenAIModel ---")
print(f"Using model: {settings.OPENAI_MODEL_NAME}")
result1 = agent_openai.run(
    "What is the capital of Portugal? Reply in exactly one sentence."
)
print(f"Response: {result1}\n")


# --- 2. LiteLLM Model (same provider, via LiteLLM proxy) ---
# LiteLLM adds a unified interface and supports 100+ providers.
# Here we use it with the same OpenAI model for comparison.
litellm_model = LiteLLMModel(
    model_id=settings.OPENAI_MODEL_NAME,
    api_key=settings.OPENAI_API_KEY.get_secret_value(),
)

agent_litellm = CodeAgent(
    tools=[],
    model=litellm_model,
    max_steps=2,
)

print("--- Model 2: LiteLLMModel ---")
print(f"Using model: {settings.OPENAI_MODEL_NAME} (via LiteLLM)")
result2 = agent_litellm.run(
    "What is the capital of Portugal? Reply in exactly one sentence."
)
print(f"Response: {result2}\n")

# --- 3. Summary ---
print("--- Summary ---")
print("Both models answered the same question. smolagents makes it easy")
print("to swap between providers by changing the model class.")
print("Available model classes: OpenAIModel, LiteLLMModel,")
print("InferenceClientModel, TransformersModel, AzureOpenAIModel,")
print("AmazonBedrockModel, MLXModel, VLLMModel")
