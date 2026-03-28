from dotenv import load_dotenv

from agno.agent import Agent, RunEvent, RunOutputEvent
from agno.models.openai import OpenAIChat

from settings import settings

load_dotenv()

"""
-------------------------------------------------------
In this example, we explore Agno with the following features:
- Streaming agent responses token-by-token
- Handling RunOutputEvent and RunEvent for stream processing
- Filtering for content events vs. other event types

Streaming lets you display partial responses as they arrive,
improving perceived latency for the user. Agno's run(stream=True)
yields RunOutputEvent objects. Each event has a .event field
you can check against RunEvent.run_content to extract the
generated text chunks.

For more details, visit:
https://docs.agno.com/agents/run#streaming
-------------------------------------------------------
"""

# --- 1. Create the agent ---
agent = Agent(
    model=OpenAIChat(id=settings.OPENAI_MODEL_NAME),
    instructions="You are a storyteller. Write creative, engaging stories.",
    markdown=True,
)

# --- 2. Run the agent with streaming ---
print("=== Streaming response ===\n")

stream = agent.run(
    "Tell me a very short story about a robot learning to paint.", stream=True
)

for chunk in stream:
    # chunk is a RunOutputEvent — filter for content events
    if isinstance(chunk, RunOutputEvent) and chunk.event == RunEvent.run_content:
        print(chunk.content, end="", flush=True)

print("\n\n=== Stream complete ===")
