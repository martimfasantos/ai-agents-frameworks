import os

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool

from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore LangChain with the following features:
- Streaming agent responses with agent.stream()
- Using stream_mode="updates" for step-by-step updates
- Using stream_mode="messages" for token-by-token LLM output

Streaming lets you display agent progress in real time instead
of waiting for the full response. The "updates" mode emits each
agent step (tool calls, model responses), while "messages" mode
streams individual LLM tokens as they are generated.

For more details, visit:
https://docs.langchain.com/oss/python/langchain/streaming
-------------------------------------------------------
"""


# --- 1. Define a tool ---
@tool
def get_population(country: str) -> str:
    """Get the approximate population of a country."""
    populations = {
        "portugal": "10.3 million",
        "japan": "125 million",
        "brazil": "215 million",
    }
    return populations.get(
        country.lower(), f"Population data not available for {country}"
    )


# --- 2. Create the agent ---
agent = create_agent(
    model=init_chat_model(f"openai:{settings.OPENAI_MODEL_NAME}"),
    tools=[get_population],
    system_prompt="You are a helpful assistant. Be concise.",
)

# --------------------------------------------------------------
# Example 1: Stream with "updates" mode
# --------------------------------------------------------------
print("=== Example 1: Stream Updates ===")

# --- 3. Stream step-by-step updates ---
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "What's the population of Portugal?"}]},
    stream_mode="updates",
):
    # Each chunk is a dict with node name -> output
    for node_name, update in chunk.items():
        print(f"[{node_name}] {update}")
    print()

# --------------------------------------------------------------
# Example 2: Stream with "messages" mode (token-by-token)
# --------------------------------------------------------------
print("=== Example 2: Stream Messages (token-by-token) ===")

# --- 4. Stream individual tokens ---
for chunk in agent.stream(
    {
        "messages": [
            {"role": "user", "content": "Tell me a fun fact about Japan's population."}
        ]
    },
    stream_mode="messages",
):
    message, metadata = chunk
    # Only print content tokens from the model node (skip tool call chunks)
    if message.content and metadata.get("langgraph_node") == "model":
        print(message.content, end="", flush=True)

print()  # Final newline
