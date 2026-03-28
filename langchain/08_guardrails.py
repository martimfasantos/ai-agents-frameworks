import os
from typing import Any

from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import (
    Runtime,
    before_agent,
    after_agent,
    hook_config,
)
from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage
from langchain.tools import tool

from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore LangChain with the following features:
- Guardrails using before_agent and after_agent hooks
- Deterministic input filtering (banned keyword detection)
- Custom content validation on agent output

Guardrails validate and filter content at key points in an agent's
execution. A before_agent hook can block unsafe requests before any
processing, while an after_agent hook can check the final response
for quality or safety before returning it to the user.

For more details, visit:
https://docs.langchain.com/oss/python/langchain/guardrails
-------------------------------------------------------
"""

# --- 1. Define a before-agent guardrail (input filter) ---
BANNED_KEYWORDS = ["hack", "exploit", "malware", "attack"]


@before_agent(can_jump_to=["end"])
def content_filter(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Block requests containing banned keywords."""
    if not state["messages"]:
        return None

    first_message = state["messages"][0]
    if first_message.type != "human":
        return None

    content = first_message.content.lower()
    for keyword in BANNED_KEYWORDS:
        if keyword in content:
            print(f"  [guardrail] BLOCKED: found banned keyword '{keyword}'")
            return {
                "messages": [
                    {
                        "role": "assistant",
                        "content": "I cannot process requests containing inappropriate content. Please rephrase your request.",
                    }
                ],
                "jump_to": "end",
            }
    print("  [guardrail] Input passed content filter")
    return None


# --- 2. Define an after-agent guardrail (output validation) ---
@after_agent(can_jump_to=["end"])
def output_length_check(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Warn if the agent response is suspiciously short."""
    if not state["messages"]:
        return None

    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.content:
        word_count = len(last_message.content.split())
        print(f"  [guardrail] Output has {word_count} words")
    return None


# --- 3. Define a tool ---
@tool
def search_docs(query: str) -> str:
    """Search documentation for an answer."""
    docs = {
        "password reset": "To reset your password, go to Settings > Security > Reset Password.",
        "billing": "Billing information can be found under Account > Billing.",
    }
    for key, value in docs.items():
        if key in query.lower():
            return value
    return f"No documentation found for: {query}"


# --- 4. Create agent with guardrails ---
agent = create_agent(
    model=init_chat_model(f"openai:{settings.OPENAI_MODEL_NAME}"),
    tools=[search_docs],
    system_prompt="You are a helpful support assistant. Use the search tool to find answers.",
    middleware=[content_filter, output_length_check],
)

# --- 5. Test with a safe request ---
print("=== Safe Request ===")
result = agent.invoke(
    {"messages": [{"role": "user", "content": "How do I reset my password?"}]}
)
print(f"Response: {result['messages'][-1].content}\n")

# --- 6. Test with a blocked request ---
print("=== Blocked Request ===")
result = agent.invoke(
    {"messages": [{"role": "user", "content": "How do I hack into the admin panel?"}]}
)
print(f"Response: {result['messages'][-1].content}")
