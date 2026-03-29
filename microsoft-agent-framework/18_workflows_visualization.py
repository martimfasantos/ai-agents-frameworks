import asyncio
import os
from dataclasses import dataclass

from dotenv import load_dotenv

from agent_framework import (
    Executor,
    WorkflowBuilder,
    WorkflowContext,
    WorkflowViz,
    handler,
    executor,
)

from settings import settings

load_dotenv()

"""
-------------------------------------------------------
In this example, we explore Microsoft Agent Framework
with the following features:
- Workflow visualization with WorkflowViz
- Generating Mermaid diagrams from workflows
- Exporting workflow graphs as PNG images

WorkflowViz introspects a workflow's executor graph and
generates visual representations — Mermaid markdown for
docs, Graphviz DOT for custom rendering, and direct PNG
export for presentations and debugging.

For more details, visit:
https://learn.microsoft.com/en-us/agent-framework/workflows/visualization?pivots=programming-language-python
-------------------------------------------------------
"""


# --- 1. Define data types and executors for a sample workflow ---
@dataclass
class UserInput:
    text: str


@dataclass
class Classification:
    category: str
    text: str


@dataclass
class Response:
    message: str


class Classifier(Executor):
    """Classifies user input into categories."""

    @handler
    async def handle(
        self, message: UserInput, ctx: WorkflowContext[Classification]
    ) -> None:
        text_lower = message.text.lower()
        if "weather" in text_lower:
            category = "weather"
        elif "help" in text_lower or "support" in text_lower:
            category = "support"
        else:
            category = "general"
        await ctx.send_message(Classification(category=category, text=message.text))


@executor(id="weather-handler")
async def handle_weather(
    message: Classification, ctx: WorkflowContext[Response]
) -> None:
    """Handles weather-related queries."""
    await ctx.yield_output(Response(message=f"Weather response for: {message.text}"))


@executor(id="support-handler")
async def handle_support(
    message: Classification, ctx: WorkflowContext[Response]
) -> None:
    """Handles support-related queries."""
    await ctx.yield_output(Response(message=f"Support response for: {message.text}"))


@executor(id="general-handler")
async def handle_general(
    message: Classification, ctx: WorkflowContext[Response]
) -> None:
    """Handles general queries."""
    await ctx.yield_output(Response(message=f"General response for: {message.text}"))


async def main() -> None:
    # --- 2. Build a workflow with branching ---
    classifier = Classifier(id="classifier")

    workflow = (
        WorkflowBuilder(start_executor=classifier)
        .add_edge(
            classifier, handle_weather, condition=lambda msg: msg.category == "weather"
        )
        .add_edge(
            classifier, handle_support, condition=lambda msg: msg.category == "support"
        )
        .add_edge(
            classifier, handle_general, condition=lambda msg: msg.category == "general"
        )
        .build()
    )

    # --- 3. Generate Mermaid diagram ---
    viz = WorkflowViz(workflow)
    mermaid = viz.to_mermaid()
    print("=== Mermaid Diagram ===")
    print(mermaid)
    print()

    # --- 4. Export as PNG (requires graphviz to be installed) ---
    os.makedirs("res", exist_ok=True)
    try:
        viz.save_png("res/workflow_graph.png")
        print("Workflow graph saved to res/workflow_graph.png")
    except Exception as e:
        print(f"PNG export skipped (graphviz may not be installed): {e}")

    # --- 5. Run the workflow to verify it works ---
    print("\n=== Running Workflow ===")
    test_inputs = [
        UserInput(text="What's the weather like?"),
        UserInput(text="I need help with my order"),
        UserInput(text="Tell me a joke"),
    ]

    for user_input in test_inputs:
        result = await workflow.run(user_input)
        outputs = result.get_outputs()
        for output in outputs:
            print(f"Input: '{user_input.text}' -> {output}")


if __name__ == "__main__":
    asyncio.run(main())
