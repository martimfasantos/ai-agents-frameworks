import os

from autogen import ConversableAgent, LLMConfig

from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore AG2 with the following features:
- Human-in-the-loop via max_consecutive_auto_reply
- Simulated human approval using a custom reply function
- Controlling conversation flow with termination conditions

Real human-in-the-loop uses human_input_mode="ALWAYS" which
requires interactive input. This example simulates human
oversight by using a reply function that auto-approves after
reviewing the agent's plan, demonstrating the pattern without
blocking execution.

For more details, visit:
https://docs.ag2.ai/latest/docs/user-guide/basic-concepts/human-in-the-loop/
-------------------------------------------------------
"""

# --- 1. Configure LLM ---
llm_config = LLMConfig({"model": settings.OPENAI_MODEL_NAME})

# --- 2. Create the assistant agent ---
assistant = ConversableAgent(
    name="assistant",
    system_message=(
        "You are a travel planner. When given a destination, "
        "propose a brief 3-day itinerary (3-5 bullet points). "
        "After the human approves, say TERMINATE."
    ),
    llm_config=llm_config,
    human_input_mode="NEVER",
)

# --- 3. Create a human proxy that auto-approves ---
# In production, this would use human_input_mode="ALWAYS" for real input.
# Here we simulate a human who reviews and approves the plan.
approval_count = 0


def simulated_human_reply(
    recipient: ConversableAgent,
    messages: list[dict] | None = None,
    sender: ConversableAgent | None = None,
    config: dict | None = None,
) -> tuple[bool, str]:
    """Simulate a human reviewing and approving the assistant's plan."""
    global approval_count
    approval_count += 1
    if approval_count == 1:
        print("\n[Simulated Human] Reviewing the plan...")
        return True, "Looks good! I approve this itinerary. Please finalize it."
    return True, "Thank you!"


human = ConversableAgent(
    name="human",
    llm_config=False,
    human_input_mode="NEVER",
    max_consecutive_auto_reply=2,
    is_termination_msg=lambda x: "TERMINATE" in (x.get("content", "") or ""),
)
human.register_reply([ConversableAgent, None], simulated_human_reply)

# --- 4. Run the conversation ---
print("=== Human-in-the-Loop: Travel Planning ===\n")
human.initiate_chat(
    assistant,
    message="Plan a 3-day trip to Barcelona.",
    max_turns=3,
)

print("\n=== Conversation Complete ===")
print(f"Human reviewed and approved after {approval_count} interaction(s).")
