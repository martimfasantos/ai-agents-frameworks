import asyncio
import os

from agents import Agent, Runner, function_tool
from agents.testing import (
    ModelStep,
    ScriptedModel,
    UnconsumedModelSteps,
    assistant_message,
    function_call,
)

from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore the OpenAI Agents SDK with the following features:
- ScriptedModel, the provider-neutral test model added in v0.21.0
- Scripting a tool call and a final answer with function_call/assistant_message
- Asserting the agent's behaviour with no network calls at all
- UnconsumedModelSteps catching a workflow that stopped early

Testing an agent normally means paying for, and waiting on, a real model whose
output changes run to run. ScriptedModel replaces the model with a fixed list
of steps, so a test drives the exact tool calls and replies it wants to
exercise and asserts on what the agent did. The steps are the SDK's normalized
output items, so the same script works regardless of provider.

For more details, visit:
https://openai.github.io/openai-agents-python/running_agents/
-------------------------------------------------------
"""

CALLS: list[str] = []


# --- 1. A tool whose invocation we want to assert on ---
@function_tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    CALLS.append(city)
    return f"{city}: sunny, 25C"


async def main() -> None:
    print("=== Scripted model testing ===\n")

    # --- 2. Script the two model turns the agent should make ---
    # Turn 1 asks for the tool; turn 2 answers using its result.
    scripted = ScriptedModel(
        [
            ModelStep(
                output=[
                    function_call(
                        name="get_weather",
                        arguments={"city": "Lisbon"},
                        call_id="call-1",
                    )
                ]
            ),
            ModelStep(output=[assistant_message("It is sunny and 25C in Lisbon.")]),
        ]
    )

    agent = Agent(
        name="weather-agent",
        instructions="Answer weather questions.",
        model=scripted,
        tools=[get_weather],
    )

    result = await Runner.run(agent, "What's the weather in Lisbon?")

    # --- 3. Assert on behaviour, not on wording ---
    print(f"Final output: {result.final_output}")
    print(f"Tool called with: {CALLS}")
    assert CALLS == ["Lisbon"], CALLS
    assert "25C" in result.final_output
    print("Assertions passed — and not one network call was made.\n")

    # --- 4. A script the agent never finishes is an error, not a silent pass ---
    # The agent stops after the first assistant message, so the third step is
    # never consumed. Without this check a workflow that quietly stopped early
    # would look identical to one that did all its work.
    over_scripted = ScriptedModel(
        [
            ModelStep(output=[assistant_message("Done immediately.")]),
            ModelStep(output=[assistant_message("Never reached.")]),
        ]
    )
    short_agent = Agent(
        name="short-agent", instructions="Answer.", model=over_scripted
    )
    await Runner.run(short_agent, "Anything")

    try:
        over_scripted.assert_complete()
    except UnconsumedModelSteps as exc:
        print(f"UnconsumedModelSteps: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
