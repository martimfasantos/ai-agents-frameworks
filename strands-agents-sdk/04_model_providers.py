import os

from strands import Agent

from settings import settings

"""
-------------------------------------------------------
In this example, we explore Strands Agents SDK with the following features:
- Amazon Bedrock provider (default)
- OpenAI provider configuration
- Anthropic provider configuration
- Ollama local provider configuration

Strands is model-agnostic. The default provider is Amazon Bedrock with
Claude Sonnet, but you can swap in OpenAI, Anthropic, Ollama, and many
more by passing a model object to the Agent constructor.

For more details, visit:
https://strandsagents.com/docs/user-guide/concepts/model-providers/
-------------------------------------------------------
"""

PROMPT = "What are the three laws of robotics? Answer concisely in 2-3 sentences."


# --- 1. Amazon Bedrock (default) ---
def bedrock_example():
    """Use Amazon Bedrock with a specific model ID."""
    from strands.models import BedrockModel

    bedrock_model = BedrockModel(
        model_id="us.anthropic.claude-sonnet-4-20250514-v1:0",
        region_name=settings.AWS_DEFAULT_REGION,
        temperature=0.3,
    )
    agent = Agent(model=bedrock_model)
    result = agent(PROMPT)
    print(f"Bedrock: {result.message}")


# --- 2. OpenAI ---
def openai_example():
    """Use OpenAI as the model provider."""
    from strands.models.openai import OpenAIModel

    openai_model = OpenAIModel(
        client_args={
            "api_key": settings.OPENAI_API_KEY.get_secret_value()
            if settings.OPENAI_API_KEY
            else os.getenv("OPENAI_API_KEY")
        },
        model_id=settings.OPENAI_MODEL_NAME,
        params={"temperature": 0.3},
    )
    agent = Agent(model=openai_model)
    result = agent(PROMPT)
    print(f"OpenAI: {result.message}")


# --- 3. Anthropic ---
def anthropic_example():
    """Use Anthropic directly (not via Bedrock)."""
    from strands.models.anthropic import AnthropicModel

    anthropic_model = AnthropicModel(
        client_args={
            "api_key": settings.ANTHROPIC_API_KEY.get_secret_value()
            if settings.ANTHROPIC_API_KEY
            else os.getenv("ANTHROPIC_API_KEY")
        },
        model_id="claude-sonnet-4-20250514",
        max_tokens=1024,
    )
    agent = Agent(model=anthropic_model)
    result = agent(PROMPT)
    print(f"Anthropic: {result.message}")


# --- 4. Ollama (local) ---
def ollama_example():
    """Use Ollama for local model inference."""
    from strands.models.ollama import OllamaModel

    ollama_model = OllamaModel(
        host="http://localhost:11434",
        model_id="llama3",
    )
    agent = Agent(model=ollama_model)
    result = agent(PROMPT)
    print(f"Ollama: {result.message}")


# --- 5. Run examples ---
if __name__ == "__main__":
    print("=== Model Provider Examples ===\n")
    print("Uncomment the provider you want to test:\n")

    # Uncomment the example you want to run (requires appropriate credentials):

    # bedrock_example()
    openai_example()
    # anthropic_example()
    # ollama_example()

    print("(Edit this file to uncomment a provider example)")
