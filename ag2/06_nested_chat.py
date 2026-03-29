import os

from autogen import ConversableAgent, LLMConfig

from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore AG2 with the following features:
- Nested chats using register_nested_chats()
- Encapsulating multi-agent workflows inside a single agent
- Automatic delegation to sub-agents on incoming messages

Nested chats allow an agent to internally run a sequence
of chats with other agents whenever it receives a message.
The outer caller sees a single agent, but internally a
full workflow executes and produces the final reply.

For more details, visit:
https://docs.ag2.ai/latest/docs/user-guide/advanced-concepts/orchestration/nested-chat/
-------------------------------------------------------
"""

# --- 1. Configure LLM ---
llm_config = LLMConfig({"model": settings.OPENAI_MODEL_NAME})

# --- 2. Create inner workflow agents ---
fact_checker = ConversableAgent(
    name="fact_checker",
    system_message=(
        "You are a fact checker. Given a claim or topic, verify key facts "
        "and list 2-3 verified points. Be concise."
    ),
    llm_config=llm_config,
    human_input_mode="NEVER",
)

editor = ConversableAgent(
    name="editor",
    system_message=(
        "You are an editor. Given fact-checked content, improve clarity "
        "and conciseness. Output a polished 2-3 sentence summary."
    ),
    llm_config=llm_config,
    human_input_mode="NEVER",
)

# --- 3. Create the lead agent with nested chats ---
lead_agent = ConversableAgent(
    name="lead_agent",
    system_message=(
        "You are a lead content agent. You oversee content quality by "
        "coordinating fact-checking and editing workflows."
    ),
    llm_config=llm_config,
    human_input_mode="NEVER",
)

# --- 4. Register nested chats on the lead agent ---
# When lead_agent receives a message from an external sender,
# it internally runs these chats in sequence.
nested_chats = [
    {
        "recipient": fact_checker,
        "message": lambda recipient, messages, sender, config: (
            f"Fact-check the following: {messages[-1]['content']}"
        ),
        "max_turns": 1,
        "summary_method": "last_msg",
    },
    {
        "recipient": editor,
        "message": "Edit and polish the fact-checked content into a final summary.",
        "max_turns": 1,
        "summary_method": "last_msg",
    },
]

lead_agent.register_nested_chats(
    chat_queue=nested_chats,
    trigger=lambda sender: sender not in [fact_checker, editor],
)

# --- 5. Create an external user agent ---
user = ConversableAgent(
    name="user",
    human_input_mode="NEVER",
    llm_config=False,
    max_consecutive_auto_reply=0,
)

# --- 6. Run the conversation ---
print("=== Nested Chat: Content Pipeline ===\n")
result = user.initiate_chat(
    lead_agent,
    message="Write about the discovery of penicillin by Alexander Fleming.",
    max_turns=1,
)

print("\n=== Final Output ===")
print(result.summary)
