import os
import asyncio

from agents import Agent, ItemHelpers, MessageOutputItem, Runner, trace

from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore OpenAI's Agents SDK with the following features:
- Agents as tools (Agent.as_tool)
- Multi-agent orchestration
- Tracing

The orchestrator agent receives a translation request and delegates to
language-specific sub-agents exposed as tools.  A synthesizer agent
then reviews and combines the translations into a final response.

For more details, visit:
https://openai.github.io/openai-agents-python/agents/#agents-as-tools
-------------------------------------------------------
"""

# --- 1. Define the agents that will be used as tools ---
spanish_agent = Agent(
    name="spanish_agent",
    instructions="You translate the user's message to Spanish",
    handoff_description="An english to spanish translator",
    model=settings.OPENAI_MODEL_NAME,
)

french_agent = Agent(
    name="french_agent",
    instructions="You translate the user's message to French",
    handoff_description="An english to french translator",
    model=settings.OPENAI_MODEL_NAME,
)

italian_agent = Agent(
    name="italian_agent",
    instructions="You translate the user's message to Italian",
    handoff_description="An english to italian translator",
    model=settings.OPENAI_MODEL_NAME,
)

# --- 2. Define the orchestrator agent that uses translation agents as tools ---
orchestrator_agent = Agent(
    name="orchestrator_agent",
    instructions=(
        "You are a translation agent. You use the tools given to you to translate. "
        "If asked for multiple translations, you call the relevant tools in order. "
        "You never translate on your own, you always use the provided tools."
    ),
    tools=[
        spanish_agent.as_tool(
            tool_name="translate_to_spanish",
            tool_description="Translate the user's message to Spanish",
        ),
        french_agent.as_tool(
            tool_name="translate_to_french",
            tool_description="Translate the user's message to French",
        ),
        italian_agent.as_tool(
            tool_name="translate_to_italian",
            tool_description="Translate the user's message to Italian",
        ),
    ],
    model=settings.OPENAI_MODEL_NAME,
)

# --- 3. Define the synthesizer agent ---
synthesizer_agent = Agent(
    name="synthesizer_agent",
    instructions=(
        "You inspect translations, correct them if needed, and "
        "produce a final concatenated response."
    ),
    model=settings.OPENAI_MODEL_NAME,
)


async def main() -> None:
    msg = "Translate 'Hello, how are you?' to Spanish, French, and Italian."

    # --- 4. Run the agents with tracing ---
    with trace("05_agents_as_tools"):
        # 4a. Run the orchestrator to get translations
        orchestrator_result = await Runner.run(orchestrator_agent, msg)

        for item in orchestrator_result.new_items:
            if isinstance(item, MessageOutputItem):
                text = ItemHelpers.text_message_output(item)
                if text:
                    print(f"  - Translation step: {text}")

        # 4b. Run the synthesizer to combine the translations
        synthesizer_result = await Runner.run(
            synthesizer_agent, orchestrator_result.to_input_list()
        )

    print(f"\n\nFinal response:\n{synthesizer_result.final_output}")


if __name__ == "__main__":
    asyncio.run(main())
