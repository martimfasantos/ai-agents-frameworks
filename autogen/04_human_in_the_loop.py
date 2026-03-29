import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import Handoff
from autogen_agentchat.conditions import HandoffTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

from settings import settings

"""
------------------------------------------------------------------------
In this example, we explore Autogen agents with the following features:
- Human-in-the-loop via HandoffTermination
- Agent handoff to user for input
- Resuming a team run with user feedback

This example shows how to integrate human feedback into agent workflows
using the handoff pattern. When the agent cannot complete a task alone,
it hands off to the user. The team pauses, and the application provides
the needed information in the next run() call to resume execution.

For more details, visit:
https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/human-in-the-loop.html
------------------------------------------------------------------------
"""


async def main() -> None:
    # --- 1. Define the model client ---
    model_client = OpenAIChatCompletionClient(
        model=settings.OPENAI_MODEL_NAME,
        api_key=settings.OPENAI_API_KEY.get_secret_value(),
    )

    # --- 2. Create an agent that hands off to the user when it cannot proceed ---
    lazy_agent = AssistantAgent(
        "lazy_assistant",
        model_client=model_client,
        handoffs=[Handoff(target="user", message="Transfer to user.")],
        system_message=(
            "If you cannot complete the task, transfer to user. "
            "Otherwise, when finished, respond with 'TERMINATE'."
        ),
    )

    # --- 3. Define termination conditions ---
    handoff_termination = HandoffTermination(target="user")
    text_termination = TextMentionTermination("TERMINATE")

    # --- 4. Create a single-agent team ---
    team = RoundRobinGroupChat(
        [lazy_agent],
        termination_condition=handoff_termination | text_termination,
    )

    # --- 5. First run: the agent hands off to the user ---
    print("=" * 50)
    print("FIRST RUN: Agent will hand off to user")
    print("=" * 50)
    await Console(team.run_stream(task="What is the weather in New York?"))

    # --- 6. Second run: provide the information the agent needs ---
    print("\n" + "=" * 50)
    print("SECOND RUN: User provides the answer")
    print("=" * 50)
    await Console(
        team.run_stream(task="The weather in New York is sunny and 72 degrees.")
    )

    # --- 7. Close the model client ---
    await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())
