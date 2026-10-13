from typing import Any

from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler

from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from deepagents.backends.utils import create_file_data
from deepagents.middleware import SkillsMiddleware

from settings import settings

load_dotenv()

"""
-----------------------------------------------------------------------
In this example, we explore Deep Agents with the following features:
- Skill-scoped tools listed under metadata.include_tools in SKILL.md
- SkillsMiddleware(tools=[...]) replacing the default skills middleware
- Tools that stay hidden from the model until their skill is read

A skill can bring its own tools. They are handed to SkillsMiddleware
instead of the agent, so their schemas stay out of every prompt until the
model reads the SKILL.md that lists them; a call before that fails as an
unknown tool. Here an "order-support" skill unlocks a lookup_order tool,
and a callback logs which tools the model was offered on each call.

For more details, visit:
https://docs.langchain.com/oss/python/deepagents/skills#add-tools-to-skills
-----------------------------------------------------------------------
"""

# --- 1. A mock tool that only the order-support skill unlocks ---
ORDERS = {"A-1042": "shipped on Oct 9 via DHL, arriving Oct 13"}


def lookup_order(order_id: str) -> str:
    """Look up the shipping status of a customer order by its ID."""
    return ORDERS.get(order_id, f"No order found with ID {order_id}.")


# include_tools is a space-separated string; a YAML list would match no tool
SKILL_MD = """---
name: order-support
description: Use when the user asks about the status of an order.
metadata:
  include_tools: lookup_order
---
# Order Support Skill
Call lookup_order with the order ID, then answer in one sentence.
"""


# --- 2. Log the tools offered to each model call and every tool call ---
# A callback sees the tools actually sent to the model; custom middleware runs
# outside SkillsMiddleware and would never see the disclosed tool.
class ToolVisibilityLogger(BaseCallbackHandler):
    def __init__(self) -> None:
        self.model_calls = 0

    def on_chat_model_start(self, serialized: dict[str, Any], messages: list, **kwargs: Any) -> None:
        tools = kwargs.get("invocation_params", {}).get("tools", [])
        offered = "lookup_order" in [t["function"]["name"] for t in tools]
        self.model_calls += 1
        print(f"  [model call {self.model_calls}] lookup_order offered: {offered}")

    def on_tool_start(self, serialized: dict[str, Any], input_str: str, **kwargs: Any) -> None:
        print(f"  [tool] {serialized['name']}({kwargs.get('inputs')})")


# --- 3. Pass the tool to SkillsMiddleware, sharing one backend with the agent ---
backend = StateBackend()
agent = create_deep_agent(
    model=f"openai:{settings.OPENAI_MODEL_NAME}",
    backend=backend,
    # Replaces the default SkillsMiddleware by name, so skills= is not needed
    middleware=[SkillsMiddleware(backend=backend, sources=["/skills/"], tools=[lookup_order])],
    system_prompt=(
        "You are a support assistant. Before answering, check the available "
        "skills; if one matches, read its SKILL.md with read_file and follow it."
    ),
)

print("=== Deep Agents Skill Tools ===")
registered = sorted(agent.nodes["tools"].bound.tools_by_name)
print(f"Tools registered on the agent: {registered}")
print(f"lookup_order registered: {'lookup_order' in registered}\n")

# --- 4. Ask something that needs the skill's tool ---
question = "Where is my order A-1042?"
print(f"User: {question}")
result = agent.invoke(
    {
        "messages": [{"role": "user", "content": question}],
        "files": {"/skills/order-support/SKILL.md": create_file_data(SKILL_MD)},
    },
    config={"callbacks": [ToolVisibilityLogger()]},
)

# --- 5. The order of tool calls shows the read unlocked the tool ---
calls = [tc["name"] for msg in result["messages"] if msg.type == "ai" for tc in msg.tool_calls]
print(f"\nTool calls in order: {' -> '.join(calls)}")
print(f"Agent: {result['messages'][-1].text}")
