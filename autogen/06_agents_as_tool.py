import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import SourceMatchTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.tools import AgentTool, TeamTool
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

from settings import settings
from utils import print_new_section

"""
------------------------------------------------------------------------
In this example, we explore Autogen agents with the following features:
- Agents as tools (AgentTool)
- Teams as tools (TeamTool)
- Disabling parallel tool calls for agent/team tools

This example shows how to use both individual agents and teams of agents
as tools that can be invoked by other agents. When using AgentTool or
TeamTool, parallel tool calls must be disabled to avoid concurrency
issues since agents and teams maintain internal state.

For more details, visit:
https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/agents.html
------------------------------------------------------------------------
"""


async def main() -> None:
    # --- 1. Define helper agents ---
    model_client = OpenAIChatCompletionClient(
        model=settings.OPENAI_MODEL_NAME,
        api_key=settings.OPENAI_API_KEY.get_secret_value(),
    )

    writer = AssistantAgent(
        name="writer",
        description="A writer agent for generating text.",
        model_client=model_client,
        system_message="Write well.",
    )
    summarizer = AssistantAgent(
        name="summarizer",
        model_client=model_client,
        system_message="You summarize the text.",
    )

    # --- 2. Create model client with parallel tool calls disabled ---
    # IMPORTANT: When using AgentTool or TeamTool, you must disable
    # parallel tool calls to avoid concurrency issues.
    main_model_client = OpenAIChatCompletionClient(
        model=settings.OPENAI_MODEL_NAME,
        api_key=settings.OPENAI_API_KEY.get_secret_value(),
        parallel_tool_calls=False,
    )

    # ============================================================
    # --- 3. Agent as Tool ---
    # ============================================================
    agent_tool = AgentTool(agent=writer)

    assistant = AssistantAgent(
        name="assistant",
        model_client=main_model_client,
        tools=[agent_tool],
        system_message="You are a helpful assistant.",
    )

    print_new_section("1. Agent as Tool")
    await Console(assistant.run_stream(task="Write a poem about the sea."))

    # ============================================================
    # --- 4. Team of Agents as Tool ---
    # ============================================================
    team = RoundRobinGroupChat(
        [writer, summarizer],
        termination_condition=SourceMatchTermination(sources=["summarizer"]),
    )

    team_tool = TeamTool(
        team=team,
        name="writing_team",
        description="A tool for writing tasks.",
        return_value_as_last_message=True,
    )

    assistant = AssistantAgent(
        name="assistant",
        model_client=main_model_client,
        tools=[team_tool],
        system_message="You are a helpful assistant.",
    )

    print_new_section("2. Team of Agents as Tool")
    await Console(assistant.run_stream(task="Write a poem about the sea."))

    # --- 5. Close model clients ---
    await model_client.close()
    await main_model_client.close()


if __name__ == "__main__":
    asyncio.run(main())
