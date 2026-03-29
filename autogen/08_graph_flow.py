import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import DiGraphBuilder, GraphFlow
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

from settings import settings
from utils import print_new_section

"""
------------------------------------------------------------------------
In this example, we explore Autogen agents with the following features:
- Sequential and parallel workflows with GraphFlow
- Directed graph construction with DiGraphBuilder
- Conditional edges based on message content

This example shows how to build structured agent workflows using
GraphFlow and DiGraphBuilder. You can define sequential pipelines
where agents run one after another, parallel fan-out where multiple
agents run concurrently, and conditional routing where the next agent
is selected based on the content of the previous message.

For more details, visit:
https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/graph-flow.html
------------------------------------------------------------------------
"""


async def main() -> None:
    # --- Setup: Define the model client ---
    model_client = OpenAIChatCompletionClient(
        model=settings.OPENAI_MODEL_NAME,
        api_key=settings.OPENAI_API_KEY.get_secret_value(),
    )

    # ============================================================
    # --- 1. Sequential Workflow ---
    # ============================================================
    # Agent A writes a draft, then Agent B reviews it.

    writer = AssistantAgent(
        "writer",
        model_client=model_client,
        system_message="Write a short 2-sentence story about a robot. Be concise.",
    )
    reviewer = AssistantAgent(
        "reviewer",
        model_client=model_client,
        system_message=(
            "Review the story and provide brief feedback. "
            "End your response with 'APPROVE' if the story is acceptable."
        ),
    )

    builder = DiGraphBuilder()
    builder.add_node(writer).add_node(reviewer)
    builder.add_edge(writer, reviewer)
    builder.set_entry_point(writer)

    team = GraphFlow(
        participants=[writer, reviewer],
        graph=builder.build(),
        termination_condition=TextMentionTermination("APPROVE"),
    )

    print_new_section("1. Sequential Workflow")
    await Console(team.run_stream(task="Write a short story about a robot."))

    # ============================================================
    # --- 2. Parallel Fan-Out Workflow ---
    # ============================================================
    # A planner fans out to two specialist agents working in parallel,
    # then a summarizer collects their outputs.

    planner = AssistantAgent(
        "planner",
        model_client=model_client,
        system_message="You plan travel itineraries. List 2 things to do in the destination city.",
    )
    food_expert = AssistantAgent(
        "food_expert",
        model_client=model_client,
        system_message="You recommend exactly one must-try local dish for the destination. Be concise (1 sentence).",
    )
    culture_expert = AssistantAgent(
        "culture_expert",
        model_client=model_client,
        system_message="You recommend exactly one cultural attraction for the destination. Be concise (1 sentence).",
    )
    summarizer = AssistantAgent(
        "summarizer",
        model_client=model_client,
        system_message="Combine all the inputs into a concise travel summary. End with 'DONE'.",
    )

    builder2 = DiGraphBuilder()
    builder2.add_node(planner).add_node(food_expert).add_node(culture_expert).add_node(
        summarizer
    )
    # Fan-out: planner -> food_expert AND planner -> culture_expert
    builder2.add_edge(planner, food_expert)
    builder2.add_edge(planner, culture_expert)
    # Fan-in: both experts -> summarizer
    builder2.add_edge(food_expert, summarizer)
    builder2.add_edge(culture_expert, summarizer)
    builder2.set_entry_point(planner)

    team2 = GraphFlow(
        participants=[planner, food_expert, culture_expert, summarizer],
        graph=builder2.build(),
        termination_condition=TextMentionTermination("DONE"),
    )

    print_new_section("2. Parallel Fan-Out Workflow")
    await Console(team2.run_stream(task="Plan a 1-day trip to Tokyo."))

    # ============================================================
    # --- 3. Conditional Routing ---
    # ============================================================
    # Based on the classifier's output, route to either a positive
    # or negative response agent.

    classifier = AssistantAgent(
        "classifier",
        model_client=model_client,
        system_message=(
            "Classify the sentiment of the user's message as either POSITIVE or NEGATIVE. "
            "Respond with only the single word: POSITIVE or NEGATIVE."
        ),
    )
    positive_responder = AssistantAgent(
        "positive_responder",
        model_client=model_client,
        system_message="The user is happy! Respond warmly and end with 'DONE'.",
    )
    negative_responder = AssistantAgent(
        "negative_responder",
        model_client=model_client,
        system_message="The user is upset. Respond empathetically and end with 'DONE'.",
    )

    builder3 = DiGraphBuilder()
    builder3.add_node(classifier).add_node(positive_responder).add_node(
        negative_responder
    )
    builder3.add_edge(
        classifier,
        positive_responder,
        condition="POSITIVE",
    )
    builder3.add_edge(
        classifier,
        negative_responder,
        condition="NEGATIVE",
    )
    builder3.set_entry_point(classifier)

    team3 = GraphFlow(
        participants=[classifier, positive_responder, negative_responder],
        graph=builder3.build(),
        termination_condition=TextMentionTermination("DONE"),
    )

    print_new_section("3. Conditional Routing (Positive)")
    await Console(team3.run_stream(task="I just got promoted at work!"))

    await team3.reset()

    print_new_section("3. Conditional Routing (Negative)")
    await Console(team3.run_stream(task="I lost my wallet today."))

    # --- Close the model client ---
    await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())
