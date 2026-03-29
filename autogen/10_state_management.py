import asyncio
import json

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

from settings import settings
from utils import print_new_section

"""
------------------------------------------------------------------------
In this example, we explore Autogen agents with the following features:
- Saving and loading agent state
- Saving and loading team state
- Resuming a conversation from a checkpoint

This example shows how to persist the internal state of agents and teams
using save_state() and load_state(). The state is a JSON-serializable
dictionary that captures model context, message history, and termination
status. This enables checkpointing long-running workflows, resuming
after interruptions, and transferring state between processes.

For more details, visit:
https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/state.html
------------------------------------------------------------------------
"""


async def main() -> None:
    # --- 1. Define the model client ---
    model_client = OpenAIChatCompletionClient(
        model=settings.OPENAI_MODEL_NAME,
        api_key=settings.OPENAI_API_KEY.get_secret_value(),
    )

    # ============================================================
    # --- 2. Agent State: save and restore ---
    # ============================================================

    agent = AssistantAgent(
        "assistant",
        model_client=model_client,
        system_message="You are a helpful assistant. Be concise.",
    )

    print_new_section("1. Agent State Management")

    # Run the agent once so it has conversation history
    await Console(agent.run_stream(task="What is the capital of France?"))

    # Save the agent's state
    agent_state = await agent.save_state()
    print("\n--- Agent state saved ---")
    print(f"State keys: {list(agent_state.keys())}")

    # Create a new agent and load the saved state
    agent2 = AssistantAgent(
        "assistant",
        model_client=model_client,
        system_message="You are a helpful assistant. Be concise.",
    )
    await agent2.load_state(agent_state)
    print("--- Agent state loaded into new agent ---")

    # The new agent remembers the previous conversation
    await Console(agent2.run_stream(task="What was my previous question?"))

    # ============================================================
    # --- 3. Team State: save, reset, and restore ---
    # ============================================================

    print_new_section("2. Team State Management")

    writer = AssistantAgent(
        "writer",
        model_client=model_client,
        system_message="Write a brief 1-sentence story continuation.",
    )
    critic = AssistantAgent(
        "critic",
        model_client=model_client,
        system_message="Provide a one-sentence critique, then say APPROVE.",
    )

    team = RoundRobinGroupChat(
        [writer, critic],
        termination_condition=MaxMessageTermination(4),
    )

    # First run
    print("--- First team run ---")
    await Console(team.run_stream(task="Start a story about a dragon."))

    # Save the full team state
    team_state = await team.save_state()
    print("\n--- Team state saved ---")
    print(f"State JSON preview: {json.dumps(team_state, indent=2)[:200]}...")

    # Reset the team (clears all internal state)
    await team.reset()
    print("--- Team reset ---")

    # Load the saved state back
    await team.load_state(team_state)
    print("--- Team state restored ---\n")

    # Continue the conversation from where we left off
    print("--- Resumed team run ---")
    await Console(team.run_stream())

    # --- 4. Close the model client ---
    await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())
