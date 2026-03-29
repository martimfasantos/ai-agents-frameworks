from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from dotenv import load_dotenv

from pydantic_ai import Agent
from pydantic_graph import BaseNode, End, Graph, GraphRunContext

from settings import settings

load_dotenv()

"""
-----------------------------------------------------------------------
In this example, we explore Pydantic Graph + Pydantic AI with the following features:
- Agents inside graph nodes for LLM-powered decisions
- Feedback loops where the LLM output drives graph transitions
- State accumulation across LLM-powered nodes
- Combining deterministic graph flow with generative AI

This example shows how to embed Pydantic AI agents inside graph nodes,
letting the LLM make decisions that drive the graph's control flow.
A content review pipeline uses one agent to draft content and another
to critique it, looping until quality is acceptable.

For more details, visit:
https://ai.pydantic.dev/graph/#genai-example
-----------------------------------------------------------------------
"""


# --- 1. Define shared state ---
@dataclass
class ReviewState:
    """State persisted across the content review pipeline."""

    topic: str = "the benefits of open source software"
    draft: str = ""
    feedback: str = ""
    revision_count: int = 0
    max_revisions: int = 2
    approved: bool = False
    history: list[str] = field(default_factory=list)


# --- 2. Create the agents ---
writer_agent = Agent(
    model=settings.OPENAI_MODEL_NAME,
    instructions=(
        "You are a concise technical writer. "
        "Write or revise content based on the given topic and feedback. "
        "Keep responses to 2-3 sentences."
    ),
)

reviewer_agent = Agent(
    model=settings.OPENAI_MODEL_NAME,
    instructions=(
        "You are a strict content reviewer. "
        "Evaluate the draft and respond with ONLY one of:\n"
        "- 'APPROVED' if the content is good\n"
        "- 'REVISE: <specific feedback>' if it needs improvement\n"
        "Be concise."
    ),
)


# --- 3. Define graph nodes ---
@dataclass
class WriteDraft(BaseNode[ReviewState]):
    """Node: Use the writer agent to create or revise content."""

    async def run(self, ctx: GraphRunContext[ReviewState]) -> ReviewDraft:
        if ctx.state.feedback:
            prompt = (
                f"Revise this draft about '{ctx.state.topic}':\n"
                f"Draft: {ctx.state.draft}\n"
                f"Feedback: {ctx.state.feedback}"
            )
        else:
            prompt = f"Write a short paragraph about: {ctx.state.topic}"

        result = await writer_agent.run(prompt)
        ctx.state.draft = result.output
        ctx.state.revision_count += 1
        ctx.state.history.append(
            f"Draft v{ctx.state.revision_count}: {ctx.state.draft}"
        )

        print(
            f"  [Writer] Draft v{ctx.state.revision_count}: {ctx.state.draft[:80]}..."
        )
        return ReviewDraft()


@dataclass
class ReviewDraft(BaseNode[ReviewState]):
    """Node: Use the reviewer agent to evaluate the draft."""

    async def run(self, ctx: GraphRunContext[ReviewState]) -> End | WriteDraft:
        result = await reviewer_agent.run(f"Review this draft:\n{ctx.state.draft}")
        review = result.output.strip()

        if (
            review.upper().startswith("APPROVED")
            or ctx.state.revision_count >= ctx.state.max_revisions
        ):
            ctx.state.approved = True
            print(f"  [Reviewer] APPROVED")
            return End(ctx.state.draft)
        else:
            # Extract feedback after "REVISE:"
            ctx.state.feedback = review.replace("REVISE:", "").strip()
            print(f"  [Reviewer] Needs revision: {ctx.state.feedback[:60]}...")
            return WriteDraft()


# --- 4. Create the graph ---
review_pipeline = Graph(nodes=[WriteDraft, ReviewDraft])


# --- 5. Run the pipeline ---
async def main():
    print("=== Graphs with GenAI: Content Review Pipeline ===\n")
    print("=" * 60)

    state = ReviewState()
    result = await review_pipeline.run(WriteDraft(), state=state)

    print()
    print("=" * 60)
    print(f"\nFinal content: {result.output}")
    print(f"Revisions: {state.revision_count}")
    print(f"Approved: {state.approved}")

    print(f"\nRevision history ({len(state.history)} versions):")
    for entry in state.history:
        print(f"  - {entry[:80]}...")

    print("\nMermaid Diagram:")
    print(review_pipeline.mermaid_code(start_node=WriteDraft))


if __name__ == "__main__":
    asyncio.run(main())
