import os

from autogen import ConversableAgent, LLMConfig
from autogen.agentchat import initiate_group_chat
from autogen.agentchat.group.patterns import AutoPattern

from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore AG2 with the following features:
- Multi-agent group chat with pattern-based orchestration
- AutoPattern for automatic speaker selection
- initiate_group_chat() as the modern group chat API

AG2's group chat allows multiple specialized agents to
collaborate on a task. AutoPattern uses LLM-based speaker
selection to route the conversation to the most appropriate
agent at each turn.

For more details, visit:
https://docs.ag2.ai/latest/docs/user-guide/advanced-concepts/orchestration/group-chat/auto-pattern/
-------------------------------------------------------
"""

# --- 1. Configure LLM ---
llm_config = LLMConfig({"model": settings.OPENAI_MODEL_NAME})

# --- 2. Create specialized agents ---
researcher = ConversableAgent(
    name="researcher",
    system_message=(
        "You are a research specialist. Find and present key facts "
        "about the topic. Be concise — 3-4 bullet points max."
    ),
    llm_config=llm_config,
    human_input_mode="NEVER",
)

writer = ConversableAgent(
    name="writer",
    system_message=(
        "You are a content writer. Take research findings and write "
        "a brief, engaging summary paragraph (3-4 sentences). "
        "When the summary is complete, end with TERMINATE."
    ),
    llm_config=llm_config,
    human_input_mode="NEVER",
)

critic = ConversableAgent(
    name="critic",
    system_message=(
        "You are a quality reviewer. Check the writer's summary for "
        "accuracy and clarity. Give brief feedback in 1-2 sentences, "
        "or say 'Approved' if it's good."
    ),
    llm_config=llm_config,
    human_input_mode="NEVER",
)

# --- 3. Create a user proxy for initiating the chat ---
user = ConversableAgent(
    name="user",
    human_input_mode="NEVER",
    llm_config=False,
    is_termination_msg=lambda x: "TERMINATE" in (x.get("content", "") or ""),
)

# --- 4. Set up AutoPattern orchestration ---
pattern = AutoPattern(
    initial_agent=researcher,
    agents=[researcher, writer, critic],
    user_agent=user,
    group_manager_args={"llm_config": llm_config},
)

# --- 5. Run the group chat ---
print("=== Multi-Agent Group Chat ===\n")
result, context, last_agent = initiate_group_chat(
    pattern=pattern,
    messages="Research and write a brief summary about the history of the Python programming language.",
    max_rounds=6,
)

print(f"\n=== Group Chat Complete (last speaker: {last_agent.name}) ===")
