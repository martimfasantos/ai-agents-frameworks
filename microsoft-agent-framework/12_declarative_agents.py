import asyncio

from dotenv import load_dotenv

from agent_framework_declarative import AgentFactory

from settings import settings

load_dotenv()

"""
-------------------------------------------------------
In this example, we explore Microsoft Agent Framework
with the following features:
- Declarative agent definition using YAML
- AgentFactory for creating agents from configuration
- Separating agent logic from agent definition

Declarative agents let you define agents in YAML or JSON
configuration files rather than code. This enables
non-developers to modify agent behavior, supports
version-controlled agent definitions, and simplifies
agent management at scale.

For more details, visit:
https://learn.microsoft.com/en-us/agent-framework/agents/declarative?pivots=programming-language-python
-------------------------------------------------------
"""


async def main() -> None:
    # --- 1. Define an agent in YAML ---
    # The YAML top-level uses `kind: Prompt` (a PromptAgent), with
    # `instructions` and a `model` block that specifies the provider,
    # apiType, and a `connection` for the API key.
    agent_yaml = f"""
kind: Prompt
name: travel-advisor
description: A helpful travel advisor agent
instructions: |
  You are a friendly travel advisor. Provide concise
  recommendations about destinations, focusing on
  practical tips. Keep responses to 2-3 sentences.
model:
  id: {settings.OPENAI_MODEL_NAME}
  provider: OpenAI
  apiType: Chat
  connection:
    kind: key
    apiKey: {settings.OPENAI_API_KEY.get_secret_value()}
"""

    # --- 2. Create agent from YAML using AgentFactory ---
    factory = AgentFactory()
    agent = await factory.create_agent_from_yaml_async(agent_yaml)

    # --- 3. Run the declarative agent ---
    print("=== Declarative Agent ===")
    result = await agent.run("What's the best time to visit Japan?")
    print(f"Answer: {result.text}\n")

    # --- 4. Run another query to show it works like any other agent ---
    result = await agent.run("What should I pack for a trip to Iceland in winter?")
    print(f"Answer: {result.text}")


if __name__ == "__main__":
    asyncio.run(main())
