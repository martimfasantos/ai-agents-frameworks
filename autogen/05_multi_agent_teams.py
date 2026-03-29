import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import (
    MaxMessageTermination,
    TextMentionTermination,
)
from autogen_agentchat.teams import (
    MagenticOneGroupChat,
    RoundRobinGroupChat,
    SelectorGroupChat,
    Swarm,
)
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

from settings import settings
from utils import print_new_section

"""
------------------------------------------------------------------------
In this example, we explore Autogen agents with the following features:
- Multi-agent collaboration
- Four team presets: RoundRobin, Selector, MagenticOne, Swarm
- Termination conditions
- Agent handoffs

This example shows the 4 different team presets of the AgentChat API:
1. RoundRobinGroupChat - Agents take turns in a round-robin fashion.
2. SelectorGroupChat - A model selects the next speaker after each turn.
3. MagenticOneGroupChat - A generalist multi-agent orchestrator.
4. Swarm - Agents use HandoffMessage to signal transitions.

For more details, visit:
https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/teams.html
------------------------------------------------------------------------
"""


async def main() -> None:
    # --- Setup: Define the model client ---
    model_client = OpenAIChatCompletionClient(
        model=settings.OPENAI_MODEL_NAME,
        api_key=settings.OPENAI_API_KEY.get_secret_value(),
    )

    # ============================================================
    # --- 1. RoundRobinGroupChat ---
    # ============================================================
    # Agents take turns publishing messages in round-robin order.

    primary_agent = AssistantAgent(
        "primary",
        model_client=model_client,
        system_message="You are a helpful AI assistant.",
    )

    critic_agent = AssistantAgent(
        "critic",
        model_client=model_client,
        system_message=(
            "Provide constructive feedback. Respond with 'APPROVE' "
            "when your feedbacks are addressed."
        ),
    )

    text_termination = TextMentionTermination("APPROVE")
    team = RoundRobinGroupChat(
        participants=[primary_agent, critic_agent],
        termination_condition=text_termination,
    )

    print_new_section("1. RoundRobinGroupChat")
    await Console(team.run_stream(task="Write a short poem about the fall season."))

    # ============================================================
    # --- 2. SelectorGroupChat ---
    # ============================================================
    # A model selects the most appropriate agent to speak next.

    async def lookup_hotel(location: str) -> str:
        return f"Here are some hotels in {location}: hotel1, hotel2, hotel3."

    async def lookup_flight(origin: str, destination: str) -> str:
        return f"Here are some flights from {origin} to {destination}: flight1, flight2, flight3."

    async def book_trip(hotel: str, flight: str) -> str:
        return "Your trip is booked!"

    travel_advisor = AssistantAgent(
        "travel_advisor",
        model_client,
        tools=[book_trip],
        description="Helps with travel planning.",
    )
    hotel_agent = AssistantAgent(
        "hotel_agent",
        model_client,
        tools=[lookup_hotel],
        description="Helps with hotel booking.",
    )
    flight_agent = AssistantAgent(
        "flight_agent",
        model_client,
        tools=[lookup_flight],
        description="Helps with flight booking.",
    )

    termination = TextMentionTermination("TERMINATE")
    team = SelectorGroupChat(
        [travel_advisor, hotel_agent, flight_agent],
        model_client=model_client,
        termination_condition=termination,
        allow_repeated_speaker=False,
        max_selector_attempts=3,
    )

    print_new_section("2. SelectorGroupChat")
    await Console(
        team.run_stream(
            task="Book a 3-day trip from Lisbon to NY with flight and hotel."
        )
    )

    # ============================================================
    # --- 3. MagenticOneGroupChat ---
    # ============================================================
    # Agents are managed by the MagenticOneOrchestrator.

    assistant = AssistantAgent(
        "assistant",
        model_client=model_client,
    )

    team = MagenticOneGroupChat(
        [assistant],
        model_client=model_client,
        max_turns=2,
        max_stalls=1,
    )

    print_new_section("3. MagenticOneGroupChat")
    await Console(team.run_stream(task="What is the capital of France?"))

    # ============================================================
    # --- 4. Swarm ---
    # ============================================================
    # Agents use handoff messages to signal transitions.

    agent1 = AssistantAgent(
        "Alice",
        model_client=model_client,
        handoffs=["Bob"],
        system_message=(
            "You are Alice and you only answer questions about yourself. "
            "If the question is about Bob, please hand off to Bob."
        ),
    )
    agent2 = AssistantAgent(
        "Bob",
        model_client=model_client,
        system_message="You are Bob and your birthday is on 1st January.",
    )

    termination = MaxMessageTermination(3)
    team = Swarm([agent1, agent2], termination_condition=termination)

    print_new_section("4. Swarm")
    await Console(team.run_stream(task="What is Bob's birthday?"))

    # --- Close the model client ---
    await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())
