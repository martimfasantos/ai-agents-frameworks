import asyncio

from dotenv import load_dotenv

from agent_framework import WorkflowBuilder
from agent_framework.openai import OpenAIChatClient

from settings import settings

load_dotenv()

"""
-------------------------------------------------------
In this example, we explore Microsoft Agent Framework
with the following features:
- Embedding agents directly into workflows
- Chaining agents using add_chain for sequential flow
- Building agent pipelines with WorkflowBuilder

Agents implement SupportsAgentRun and can be used
directly as workflow executors. This lets you chain
LLM-powered agents in a workflow graph, passing output
from one agent to the next — perfect for multi-step
content generation and processing pipelines.

For more details, visit:
https://learn.microsoft.com/en-us/agent-framework/workflows/agents-in-workflows?pivots=programming-language-python
-------------------------------------------------------
"""


async def main() -> None:
    # --- 1. Create the client and specialist agents ---
    client = OpenAIChatClient(
        model_id=settings.OPENAI_MODEL_NAME,
        api_key=settings.OPENAI_API_KEY.get_secret_value(),
    )

    # Agent that generates a story outline
    outliner = client.as_agent(
        name="outliner",
        instructions=(
            "You are a story outliner. Given a topic, create a brief 3-point "
            "outline for a short story. Output ONLY the outline, numbered 1-3."
        ),
    )

    # Agent that writes a story from an outline
    writer = client.as_agent(
        name="writer",
        instructions=(
            "You are a creative writer. Given a story outline, write a very "
            "short story (3-4 sentences) based on it. Output ONLY the story."
        ),
    )

    # Agent that provides editorial feedback
    editor = client.as_agent(
        name="editor",
        instructions=(
            "You are a story editor. Given a short story, provide a brief "
            "one-sentence editorial comment about its quality and style."
        ),
    )

    # --- 2. Build a sequential workflow — agents are used directly ---
    workflow = (
        WorkflowBuilder(start_executor=outliner)
        .add_chain([outliner, writer, editor])
        .build()
    )

    # --- 3. Run the workflow ---
    print("Running story pipeline: Outliner -> Writer -> Editor")
    print("=" * 50)

    result = await workflow.run("A robot discovering music for the first time")

    # --- 4. Print outputs from each stage ---
    outputs = result.get_outputs()
    for i, output in enumerate(outputs):
        print(f"\nStage {i + 1} output:")
        print(output)
        print("-" * 50)


if __name__ == "__main__":
    asyncio.run(main())
