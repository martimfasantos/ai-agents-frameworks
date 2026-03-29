import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_core.memory import ListMemory, MemoryContent, MemoryMimeType
from autogen_ext.models.openai import OpenAIChatCompletionClient

from settings import settings

"""
------------------------------------------------------------------------
In this example, we explore Autogen agents with the following features:
- Memory stores with ListMemory
- User preferences persisted across interactions
- Memory-aware tool usage

This example shows a primitive memory store that maintains user
preferences across interactions. The agent automatically queries its
memory before each response, enriching its context. You can build on
the Memory protocol to implement more complex stores using vector
databases, ML models, or other advanced storage mechanisms.

For more details, visit:
https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/memory.html
------------------------------------------------------------------------
"""


# --- 1. Define a tool function ---
async def get_weather(city: str, units: str = "imperial") -> str:
    """Get weather information for a city."""
    if units == "imperial":
        return f"The weather in {city} is 73 °F and Sunny."
    elif units == "metric":
        return f"The weather in {city} is 23 °C and Sunny."
    else:
        return f"Sorry, I don't know the weather in {city}."


async def main() -> None:
    # --- 2. Define the model client ---
    model_client = OpenAIChatCompletionClient(
        model=settings.OPENAI_MODEL_NAME,
        api_key=settings.OPENAI_API_KEY.get_secret_value(),
    )

    # --- 3. Initialize user memory and add preferences ---
    user_memory = ListMemory()

    await user_memory.add(
        MemoryContent(
            content="The weather should be in metric units",
            mime_type=MemoryMimeType.TEXT,
        )
    )

    await user_memory.add(
        MemoryContent(
            content="Meal recipes must be vegan",
            mime_type=MemoryMimeType.TEXT,
        )
    )

    # --- 4. Define the agent with memory ---
    assistant_agent = AssistantAgent(
        name="assistant_agent",
        model_client=model_client,
        tools=[get_weather],
        memory=[user_memory],
        system_message="You are a helpful assistant that remembers user preferences.",
    )

    # --- 5. Run the agent - weather query (should use metric units) ---
    print("=" * 50)
    print("Weather Query (expects metric units from memory):")
    print("=" * 50)
    await Console(
        assistant_agent.run_stream(task="What's the weather like in New York?")
    )

    # --- 6. Run the agent - recipe query (should be vegan) ---
    print("\n" + "=" * 50)
    print("Recipe Query (expects vegan from memory):")
    print("=" * 50)
    await Console(
        assistant_agent.run_stream(task="Write a brief meal recipe with broth.")
    )

    # --- 7. Close the model client ---
    await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())
