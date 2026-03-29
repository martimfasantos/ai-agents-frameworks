import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import (
    MaxMessageTermination,
    TextMentionTermination,
    TimeoutTermination,
)
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

from settings import settings
from utils import print_new_section

"""
------------------------------------------------------------------------
In this example, we explore Autogen agents with the following features:
- Termination conditions: MaxMessage, TextMention, Timeout
- Combining conditions with OR (|) and AND (&) operators
- Inspecting termination reason from task results

This example shows how to control when agent teams stop execution using
built-in termination conditions. You can limit by message count, watch
for a keyword, set a wall-clock timeout, or combine multiple conditions
with logical operators so that the team stops when any (OR) or all (AND)
conditions are met.

For more details, visit:
https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/termination.html
------------------------------------------------------------------------
"""


async def main() -> None:
    # --- Setup: Define the model client ---
    model_client = OpenAIChatCompletionClient(
        model=settings.OPENAI_MODEL_NAME,
        api_key=settings.OPENAI_API_KEY.get_secret_value(),
    )

    agent = AssistantAgent(
        "assistant",
        model_client=model_client,
        system_message="You are a helpful assistant.",
    )

    # ============================================================
    # --- 1. MaxMessageTermination ---
    # ============================================================
    # Stop after a fixed number of messages (including the user message).

    max_msg = MaxMessageTermination(max_messages=3)
    team = RoundRobinGroupChat([agent], termination_condition=max_msg)

    print_new_section("1. MaxMessageTermination (3 msgs)")
    result = await Console(team.run_stream(task="Count from 1 to 10."))
    print(f"Stop reason: {result.stop_reason}")

    # ============================================================
    # --- 2. TextMentionTermination ---
    # ============================================================
    # Stop when the agent produces a message containing a keyword.

    text_term = TextMentionTermination("DONE")
    team2 = RoundRobinGroupChat([agent], termination_condition=text_term)

    print_new_section("2. TextMentionTermination ('DONE')")
    result2 = await Console(team2.run_stream(task="Say hello and then say DONE."))
    print(f"Stop reason: {result2.stop_reason}")

    # ============================================================
    # --- 3. TimeoutTermination ---
    # ============================================================
    # Stop after a wall-clock timeout in seconds.

    timeout_term = TimeoutTermination(timeout_seconds=5)
    team3 = RoundRobinGroupChat([agent], termination_condition=timeout_term)

    print_new_section("3. TimeoutTermination (5 seconds)")
    result3 = await Console(team3.run_stream(task="Tell me a very short joke."))
    print(f"Stop reason: {result3.stop_reason}")

    # ============================================================
    # --- 4. Combining with OR (|) ---
    # ============================================================
    # Stop when EITHER the keyword appears OR max messages is reached.

    combined_or = TextMentionTermination("FINISHED") | MaxMessageTermination(10)
    team4 = RoundRobinGroupChat([agent], termination_condition=combined_or)

    print_new_section("4. Combined OR: TextMention | MaxMessage")
    result4 = await Console(
        team4.run_stream(task="Write one sentence about the moon, then say FINISHED.")
    )
    print(f"Stop reason: {result4.stop_reason}")

    # ============================================================
    # --- 5. Combining with AND (&) ---
    # ============================================================
    # Stop only when BOTH conditions are satisfied.
    # Here we require at least 3 messages AND the keyword "COMPLETE".

    combined_and = MaxMessageTermination(3) & TextMentionTermination("COMPLETE")
    team5 = RoundRobinGroupChat([agent], termination_condition=combined_and)

    print_new_section("5. Combined AND: MaxMessage & TextMention")
    result5 = await Console(
        team5.run_stream(
            task="Say one word per message. After two messages say COMPLETE."
        )
    )
    print(f"Stop reason: {result5.stop_reason}")

    # --- Close the model client ---
    await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())
