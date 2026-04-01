import os

from autogen import ConversableAgent, GroupChat, GroupChatManager, LLMConfig
from autogen.agentchat.group.safeguards import apply_safeguard_policy

from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore AG2 with the following features:
- Guardrails using AG2's safeguard policy system (Maris)
- Regex-based content filtering between agents
- Inter-agent safeguards with block and warn actions

AG2's safeguard system (Maris) lets you define policies
that filter or block agent messages based on regex patterns
or LLM-based checks. This protects against sensitive data
leakage, harmful content, or policy violations in multi-agent
conversations.

For more details, visit:
https://docs.ag2.ai/latest/docs/use-cases/notebooks/notebooks/agentchat_safeguard_demo/
-------------------------------------------------------
"""

# --- 1. Configure LLM ---
llm_config = LLMConfig({"model": settings.OPENAI_MODEL_NAME})

# --- 2. Create agents for a group chat ---
assistant = ConversableAgent(
    name="assistant",
    system_message=(
        "You are a helpful assistant. When asked about a topic, provide "
        "useful information. Always be helpful and informative."
    ),
    llm_config=llm_config,
    human_input_mode="NEVER",
)

reviewer = ConversableAgent(
    name="reviewer",
    system_message=(
        "You are a content reviewer. After the assistant responds, "
        "summarize the key points in 1-2 sentences. End with APPROVE."
    ),
    llm_config=llm_config,
    human_input_mode="NEVER",
)

user = ConversableAgent(
    name="user",
    human_input_mode="NEVER",
    llm_config=False,
    is_termination_msg=lambda x: "APPROVE" in (x.get("content", "") or ""),
)

# --- 3. Create group chat with GroupChatManager ---
groupchat = GroupChat(
    agents=[user, assistant, reviewer],
    messages=[],
    max_round=4,
    send_introductions=False,
)

manager = GroupChatManager(
    groupchat=groupchat,
    llm_config=llm_config,
)

# --- 4. Define safeguard policy ---
# Inter-agent safeguards monitor messages between agents.
# - Block messages containing SSN-like patterns (e.g., 123-45-6789)
# - Warn on messages containing sensitive keywords
safeguard_policy = {
    "inter_agent_safeguards": {
        "agent_transitions": [
            {
                "message_source": "assistant",
                "message_destination": "reviewer",
                "check_method": "regex",
                "pattern": r"\b\d{3}-\d{2}-\d{4}\b",
                "action": "block",
            },
            {
                "message_source": "assistant",
                "message_destination": "user",
                "check_method": "regex",
                "pattern": r"(?i)\b(password|secret|credential)\b",
                "action": "warning",
            },
        ]
    },
}

# --- 5. Apply safeguard policy ---
print("=== Guardrails: Safeguard Policy Demo ===\n")
print("Safeguard rules applied:")
print("  1. BLOCK: SSN-like patterns (###-##-####) from assistant -> reviewer")
print(
    "  2. WARN: Sensitive keywords (password/secret/credential) from assistant -> user"
)
print()

apply_safeguard_policy(
    groupchat_manager=manager,
    policy=safeguard_policy,
)

# --- 6. Run the group chat ---
result = user.initiate_chat(
    manager,
    message="Explain best practices for protecting personal information like social security numbers online.",
    max_turns=4,
)

print("\n=== Guardrails Demo Complete ===")
print("The safeguard policy was active during the conversation,")
print("monitoring agent messages for SSN patterns and sensitive terms.")
