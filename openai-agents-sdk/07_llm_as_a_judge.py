from __future__ import annotations

import os
import asyncio
from dataclasses import dataclass
from typing import Literal

from agents import Agent, ItemHelpers, Runner, TResponseInputItem, trace

from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore OpenAI's Agents SDK with the following features:
- LLM-as-a-judge evaluation loop
- Dataclass output types
- Multi-turn agent interaction

A story-outline generator produces an outline, then an evaluator
agent scores it.  The loop continues, feeding feedback back in,
until the evaluator gives a "pass".

For more details, visit:
https://openai.github.io/openai-agents-python/running_agents/
-------------------------------------------------------
"""

# --- 1. Define the story outline generator agent ---
story_outline_generator = Agent(
    name="story_outline_generator",
    instructions=(
        "You generate a very short story outline based on the user's input. "
        "Keep it to 3-5 bullet points maximum. Be concise — no full paragraphs. "
        "If there is any feedback provided, use it to improve the outline."
    ),
    model=settings.OPENAI_MODEL_NAME,
)


# --- 2. Define the evaluation feedback model ---
@dataclass
class EvaluationFeedback:
    feedback: str
    score: Literal["pass", "needs_improvement", "fail"]


# --- 3. Define the evaluator / LLM judge agent ---
evaluator = Agent[None](
    name="evaluator",
    instructions=(
        "You evaluate a story outline and decide if it's good enough. "
        "If it's not good enough, you provide feedback on what needs to be improved. "
        "Never give it a pass on the first try. "
        "If the outline is already decent on the second try, give it a pass. "
        "Don't be harsh, but be honest and constructive in your feedback."
    ),
    output_type=EvaluationFeedback,
    model=settings.OPENAI_MODEL_NAME,
)


async def main() -> None:
    msg = "A story about a robot learning to paint"
    input_items: list[TResponseInputItem] = [{"content": msg, "role": "user"}]

    latest_outline: str | None = None

    # --- 4. Run the generate-evaluate loop (max 3 iterations) ---
    max_iterations = 3
    with trace("07_llm_as_a_judge"):
        for iteration in range(1, max_iterations + 1):
            print(f"\n--- Iteration {iteration} ---")
            story_outline_result = await Runner.run(
                story_outline_generator,
                input_items,
            )

            input_items = story_outline_result.to_input_list()
            latest_outline = ItemHelpers.text_message_outputs(
                story_outline_result.new_items
            )
            print(f"Story outline generated ({len(latest_outline)} chars)")

            evaluator_result = await Runner.run(evaluator, input_items)
            result: EvaluationFeedback = evaluator_result.final_output

            print(f"Evaluator score: {result.score}")
            print(f"Feedback: {result.feedback[:150]}")

            if result.score == "pass":
                print("\nStory outline passed evaluation!")
                break

            print("Re-running with feedback...")
            input_items.append(
                {"content": f"Feedback: {result.feedback}", "role": "user"}
            )
        else:
            print(f"\nReached max iterations ({max_iterations}), stopping.")

    print(f"\nFinal story outline:\n{latest_outline}")


if __name__ == "__main__":
    asyncio.run(main())
