import os
import asyncio

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm

from settings import settings
from utils import call_agent_async, print_new_section

# Both the native Gemini model and LiteLLM routing need their keys in env
os.environ["GOOGLE_API_KEY"] = settings.GOOGLE_API_KEY.get_secret_value()
os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore Google ADK with the following features:
- LiteLlm integration: use non-Google models inside ADK agents
- OpenAI-compatible models (e.g. gpt-4o-mini) via LiteLLM routing
- Side-by-side comparison of Google Gemini vs an OpenAI model

ADK's LiteLlm adapter lets you swap any LiteLLM-supported model
(OpenAI, Anthropic, Cohere, Mistral, and more) into an LlmAgent
without changing the agent code. This makes it easy to benchmark
different providers or use specialised models for different tasks.

For more details, visit:
https://google.github.io/adk-docs/agents/models/litellm/
-------------------------------------------------------
"""


async def main() -> None:
    query = (
        "Explain the concept of recursion in programming. "
        "Answer in exactly 2 sentences."
    )

    # --- 2. Agent using Google Gemini (native ADK model) ---
    gemini_agent = LlmAgent(
        name="GeminiAgent",
        model=settings.GOOGLE_MODEL_NAME,  # e.g. "gemini-2.0-flash"
        instruction="You are a helpful programming tutor. Be concise.",
    )

    # --- 3. Agent using OpenAI via LiteLLM ---
    # LiteLlm model strings follow the format "<provider>/<model>"
    openai_agent = LlmAgent(
        name="OpenAIAgent",
        model=LiteLlm(model=settings.OPENAI_MODEL_NAME),  # e.g. "gpt-4o-mini"
        instruction="You are a helpful programming tutor. Be concise.",
    )

    # --------------------------------------------------------------
    # Example 1: Gemini answers the question
    # --------------------------------------------------------------
    print_new_section("1. Google Gemini (native ADK model)")
    print(f"  Model : {settings.GOOGLE_MODEL_NAME}")
    print(f"  Query : {query}\n")

    await call_agent_async(gemini_agent, query)

    # --------------------------------------------------------------
    # Example 2: OpenAI (via LiteLLM) answers the same question
    # --------------------------------------------------------------
    print_new_section("2. OpenAI via LiteLLM")
    print(f"  Model : {settings.OPENAI_MODEL_NAME}")
    print(f"  Query : {query}\n")

    await call_agent_async(openai_agent, query)

    print("\n" + "-" * 65)
    print("  Both agents used the same ADK interface — only the model differed.")
    print("-" * 65)


if __name__ == "__main__":
    asyncio.run(main())
