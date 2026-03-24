from strands import Agent
from strands.models.openai import OpenAIModel

from settings import settings

"""
-------------------------------------------------------
In this example, we explore Strands Agents SDK with the following features:
- Minimal agent creation with default settings
- Agent invocation with a simple prompt
- Accessing the agent result (message and stop reason)

The simplest possible Strands agent — just import Agent and call it.
Strands defaults to Amazon Bedrock with Claude Sonnet as the model provider,
so no explicit model configuration is required. Here we use OpenAI as an
alternative provider configured via settings.

For more details, visit:
https://strandsagents.com/docs/user-guide/quickstart/python/
-------------------------------------------------------
"""

# --- 1. Configure the model ---
# Default: Agent() uses Amazon Bedrock with Claude Sonnet (requires AWS credentials)
# Alternative: Use OpenAI as shown below
openai_model = OpenAIModel(
    client_args={
        "api_key": settings.OPENAI_API_KEY.get_secret_value()
        if settings.OPENAI_API_KEY
        else ""
    },
    model_id=settings.OPENAI_MODEL_NAME,
)

# --- 2. Create a basic agent ---
agent = Agent(model=openai_model)

# --- 3. Invoke the agent with a simple prompt ---
result = agent("What is the capital of France? Answer in one sentence.")

# --- 4. Access the result ---
print("\n--- Agent Result ---")
print(f"Message: {result.message}")
print(f"Stop reason: {result.stop_reason}")
