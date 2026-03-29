import os

from autogen import ConversableAgent, LLMConfig

from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore AG2 with the following features:
- Sequential chat pipeline using initiate_chats()
- Automatic carryover of summaries between chats
- Multi-step workflows with different specialist agents

Sequential chat chains multiple two-agent conversations
together, where each chat's summary becomes the carryover
context for the next chat. This enables multi-step
workflows where each agent handles a specific task.

For more details, visit:
https://docs.ag2.ai/latest/docs/user-guide/advanced-concepts/orchestration/sequential-chat/
-------------------------------------------------------
"""

# --- 1. Configure LLM ---
llm_config = LLMConfig({"model": settings.OPENAI_MODEL_NAME})

# --- 2. Create the orchestrator agent ---
orchestrator = ConversableAgent(
    name="orchestrator",
    system_message=(
        "You are a project orchestrator. You coordinate work between "
        "specialists and provide context for each step."
    ),
    llm_config=llm_config,
    human_input_mode="NEVER",
)

# --- 3. Create specialist agents ---
researcher = ConversableAgent(
    name="researcher",
    system_message=(
        "You are a technology researcher. When given a topic, provide "
        "3-4 key facts or trends about it. Be concise and factual."
    ),
    llm_config=llm_config,
    human_input_mode="NEVER",
)

analyst = ConversableAgent(
    name="analyst",
    system_message=(
        "You are a business analyst. Given research findings, identify "
        "2-3 business opportunities or implications. Be specific and brief."
    ),
    llm_config=llm_config,
    human_input_mode="NEVER",
)

writer = ConversableAgent(
    name="writer",
    system_message=(
        "You are an executive summary writer. Given research and analysis, "
        "write a concise executive briefing (3-4 sentences) that synthesizes "
        "the key findings and recommendations."
    ),
    llm_config=llm_config,
    human_input_mode="NEVER",
)

# --- 4. Run sequential chat pipeline ---
print("=== Sequential Chat Pipeline ===\n")

chat_results = orchestrator.initiate_chats(
    [
        {
            "recipient": researcher,
            "message": "Research the current state of quantum computing in 2025.",
            "max_turns": 1,
            "summary_method": "last_msg",
        },
        {
            "recipient": analyst,
            "message": "Analyze the business implications of these findings.",
            "max_turns": 1,
            "summary_method": "last_msg",
        },
        {
            "recipient": writer,
            "message": "Write an executive briefing based on the research and analysis.",
            "max_turns": 1,
            "summary_method": "last_msg",
        },
    ]
)

# --- 5. Display pipeline results ---
print("\n=== Pipeline Results ===")
for i, (label, result) in enumerate(
    zip(["Research", "Analysis", "Executive Briefing"], chat_results)
):
    print(f"\n--- Step {i + 1}: {label} ---")
    print(result.summary[:200] + "..." if len(result.summary) > 200 else result.summary)
