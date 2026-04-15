import os

from autogen import ConversableAgent, LLMConfig, gather_usage_summary

from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore AG2 with the following features:
- Token usage tracking via get_total_usage() and get_actual_usage()
- Per-model usage breakdown from the agent's client
- Multi-agent usage aggregation with gather_usage_summary()

AG2 tracks token usage internally on each ConversableAgent's
OpenAI client. After a conversation completes, you can retrieve
prompt, completion, and total token counts per model. The
gather_usage_summary() utility aggregates usage across multiple
agents for cost analysis.

For more details, visit:
https://docs.ag2.ai/latest/docs/user-guide/basic-concepts/agents/conversable-agent
-------------------------------------------------------
"""

# --- 1. Configure LLM ---
llm_config = LLMConfig({"model": settings.OPENAI_MODEL_NAME})

# --- 2. Create agents ---
assistant = ConversableAgent(
    name="assistant",
    system_message=(
        "You are a helpful assistant. Answer questions concisely in 1-2 sentences."
    ),
    llm_config=llm_config,
    human_input_mode="NEVER",
)

user = ConversableAgent(
    name="user",
    human_input_mode="NEVER",
    llm_config=False,
    is_termination_msg=lambda x: True,
)

# --- 3. Run conversation 1 ---
print("=== Token Usage Tracking ===\n")
print("--- Conversation 1 ---")
user.initiate_chat(
    assistant,
    message="What is the speed of light?",
    max_turns=1,
)

# --- 4. Check usage after first conversation ---
print("\n--- Usage after conversation 1 ---")
total_usage = assistant.get_total_usage()
actual_usage = assistant.get_actual_usage()

if total_usage:
    for model, usage in total_usage.items():
        if isinstance(usage, dict):
            print(f"  Model: {model}")
            print(f"    Prompt tokens:     {usage.get('prompt_tokens', 0)}")
            print(f"    Completion tokens: {usage.get('completion_tokens', 0)}")
            print(f"    Total tokens:      {usage.get('total_tokens', 0)}")
else:
    print("  No usage data available")

# --- 5. Run conversation 2 ---
print("\n--- Conversation 2 ---")
user.initiate_chat(
    assistant,
    message="What is the largest planet in our solar system?",
    max_turns=1,
)

# --- 6. Check cumulative usage ---
print("\n--- Cumulative usage after 2 conversations ---")
total_usage = assistant.get_total_usage()

if total_usage:
    for model, usage in total_usage.items():
        if isinstance(usage, dict):
            print(f"  Model: {model}")
            print(f"    Prompt tokens:     {usage.get('prompt_tokens', 0)}")
            print(f"    Completion tokens: {usage.get('completion_tokens', 0)}")
            print(f"    Total tokens:      {usage.get('total_tokens', 0)}")

# --- 7. Aggregate usage across agents ---
print("\n--- Aggregated usage (gather_usage_summary) ---")
summary = gather_usage_summary([assistant, user])

for category, data in summary.items():
    if data:
        print(f"\n  {category}:")
        for key, value in data.items():
            print(f"    {key}: {value}")

# --- 8. Print usage summary (built-in pretty print) ---
print("\n--- Built-in print_usage_summary ---")
assistant.print_usage_summary()

print("\n=== Token Usage Demo Complete ===")
