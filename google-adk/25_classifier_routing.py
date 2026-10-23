import os
import asyncio

from google.adk import Context, Event
from google.adk.labs.classifiers import Binary, Choice, ClassifierNode, LlmClassifier
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.workflow import Workflow, START
from google.genai import types

from settings import settings
from utils import print_new_section

os.environ["GOOGLE_API_KEY"] = settings.GOOGLE_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore Google ADK with the following features:
- ClassifierNode: a workflow node whose route is a classifier's answer
- LlmClassifier: any Gemini model answering typed questions with probabilities
- Choice (pick one label) and Binary (probability of true, cut by threshold=)
- answer_key: the full answer, with every probability, saved to session state

Many graphs branch on a judgment about the input: which team owns a ticket,
or whether it is urgent. A ClassifierNode replaces the "LLM returns a label +
function maps it to a route" pair with one node, and the Workflow checks at
build time that every label has an edge. The node passes its input through
unchanged, so the routed-to node sees the original ticket. Part of the
experimental google.adk.labs package (new in ADK 2.13.0).

For more details, visit:
https://github.com/google/adk-python/blob/v2.13.0/docs/guides/labs/classifiers/classifier_node/index.md
-------------------------------------------------------
"""

APP_NAME = "classifier_demo"
USER_ID = "user"
URGENCY_THRESHOLD = 0.7


# ----------------------------------------------------------------
#                1. Classifier and the two questions
# ----------------------------------------------------------------
# The keys of Choice.criteria are both the labels and the routes, so the edge
# dict below must use the same strings. A Binary routes on True/False.

classifier = LlmClassifier(model=settings.GOOGLE_MODEL_NAME)

triage = ClassifierNode(
    name="triage",
    classifier=classifier,
    question=Choice(
        instructions="Which team should handle this support ticket?",
        criteria={
            "billing": "Invoices, refunds, or subscription charges",
            "technical": "Bugs, errors, outages, or API problems",
        },
    ),
    answer_key="triage_answer",
)

is_urgent = ClassifierNode(
    name="is_urgent",
    classifier=classifier,
    question=Binary(
        instructions="Is this an outage, or does it block the customer's production use?",
    ),
    # A wrong page wakes someone up, so require 70% rather than the 0.5 default.
    threshold=URGENCY_THRESHOLD,
    answer_key="urgency_answer",
)


# ----------------------------------------------------------------
#                 2. Terminal nodes (plain Python)
# ----------------------------------------------------------------


def billing_queue(node_input: str) -> str:
    return "Filed with the billing team."


def page_oncall(node_input: str, ctx: Context) -> str:
    # answer_key makes the classifier's probability readable downstream.
    probability = ctx.state["urgency_answer"]["probability"]
    return f"Paged the on-call engineer (urgency {probability:.0%})."


def eng_queue(node_input: str) -> str:
    return "Filed in the engineering backlog."


# ----------------------------------------------------------------
#                      3. Wire the graph
# ----------------------------------------------------------------
# Construction fails if an edge names a label the question never emits, or a
# label has no edge — e.g. writing "biling" or dropping the False branch.

support_workflow = Workflow(
    name="support_workflow",
    edges=[
        (START, triage),
        (triage, {"billing": billing_queue, "technical": is_urgent}),
        (is_urgent, {True: page_oncall, False: eng_queue}),
    ],
)


# ----------------------------------------------------------------
#                  4. Run tickets and trace each route
# ----------------------------------------------------------------


def describe_answer(answer: dict) -> str:
    if answer["type"] == "choice":
        probs = ", ".join(f"{k}={v:.2f}" for k, v in answer["probabilities"].items())
        return f"P: {probs}"
    return f"P(true)={answer['probability']:.2f}, threshold={URGENCY_THRESHOLD}"


async def run_ticket(ticket: str, session_id: str) -> None:
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=session_id
    )
    runner = Runner(
        node=support_workflow, app_name=APP_NAME, session_service=session_service
    )

    message = types.Content(role="user", parts=[types.Part(text=ticket)])
    path = []
    async for event in runner.run_async(
        user_id=USER_ID, session_id=session_id, new_message=message
    ):
        node_name = getattr(event, "node_name", None)
        if not node_name:
            continue
        path.append(node_name)
        route = event.actions.route
        delta = event.actions.state_delta
        if route is not None:
            answer = delta.get("triage_answer") or delta.get("urgency_answer")
            print(f"    {node_name:<13} route={route!r:<12} {describe_answer(answer)}")
        else:
            print(f"    {node_name:<13} -> {event.output}")
    print(f"    path: {' -> '.join(path)}")


async def main() -> None:
    tickets = [
        "I was charged twice for my March invoice, please refund one of them.",
        "Checkout API returns 500 on every call since 9am, production orders are failing.",
        "The tooltip on the dashboard export button has a typo.",
    ]
    for i, ticket in enumerate(tickets, start=1):
        print_new_section(f"Ticket {i}")
        print(f"  {ticket!r}\n")
        await run_ticket(ticket, session_id=f"ticket-{i}")


if __name__ == "__main__":
    asyncio.run(main())
