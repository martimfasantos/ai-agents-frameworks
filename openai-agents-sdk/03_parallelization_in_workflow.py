import os
import asyncio

from agents import Agent, ItemHelpers, Runner, trace

from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore OpenAI's Agents SDK with the following features:
- Parallelization with asyncio.gather
- Predefined workflows
- Single trace wrapping multiple runs

This example runs two translation agents in parallel, collects their
outputs, then asks a picker agent to choose the best translation --
all wrapped in a single trace for unified observability.

For more details, visit:
https://openai.github.io/openai-agents-python/running_agents/
-------------------------------------------------------
"""

# --- 1. Define the agents ---
spanish_agent = Agent(
    name="spanish_agent",
    instructions="You translate the user's message to Spanish",
    model=settings.OPENAI_MODEL_NAME,
)

translation_picker = Agent(
    name="translation_picker",
    instructions="You pick the best Spanish translation from the given options.",
    model=settings.OPENAI_MODEL_NAME,
)


async def main() -> None:
    msg = "Hello, how are you doing today? I hope you're having a great day!"

    # --- 2. Run the agents with parallelization and tracing ---
    with trace("03_parallelization_in_workflow"):
        res_1, res_2 = await asyncio.gather(
            Runner.run(spanish_agent, msg),
            Runner.run(spanish_agent, msg),
        )

        # --- 3. Get the last text messages from the outputs ---
        outputs = [
            ItemHelpers.text_message_outputs(res_1.new_items),
            ItemHelpers.text_message_outputs(res_2.new_items),
        ]

        translations = "\n\n".join(outputs)
        print(f"\n\nTranslations:\n\n{translations}")

        # --- 4. Pick the best translation ---
        best_translation = await Runner.run(
            translation_picker,
            f"Input: {msg}\n\nTranslations:\n{translations}",
        )

    print("\n\n-----")
    print(f"Best translation: {best_translation.final_output}")


if __name__ == "__main__":
    asyncio.run(main())
