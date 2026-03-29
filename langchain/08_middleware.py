import os
from dataclasses import dataclass
from typing import Any

from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import (
    Runtime,
    before_model,
    after_model,
    dynamic_prompt,
    wrap_model_call,
    ModelRequest,
    ModelResponse,
)
from langchain.tools import tool
from langchain_openai import ChatOpenAI

from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore LangChain with the following features:
- Middleware hooks: @before_model, @after_model, @dynamic_prompt
- The @wrap_model_call decorator for request/response interception
- Logging, prompt customization, and model call wrapping

Middleware hooks let you intercept and modify the agent's behavior
at each step of its execution loop. Use @dynamic_prompt to change
the system prompt per call, @before_model / @after_model for
logging, and @wrap_model_call for full request/response control.

For more details, visit:
https://docs.langchain.com/oss/python/langchain/middleware
-------------------------------------------------------
"""


# --- 1. Define context for personalization ---
@dataclass
class Context:
    user_name: str


# --- 2. Create the model ---
model = ChatOpenAI(
    model=settings.OPENAI_MODEL_NAME,
    temperature=0.1,
    max_tokens=1000,
    timeout=30,
)


# --- 3. Create middleware hooks ---


# Dynamic prompt: personalize the system prompt based on context
@dynamic_prompt
def personalized_prompt(request: ModelRequest) -> str:
    user_name = request.runtime.context.user_name
    return f"You are a helpful assistant. Address the user as {user_name}. Be concise."


# Before model: log each model call
@before_model
def log_before(state: AgentState, runtime: Runtime[Context]) -> dict | None:
    message_count = len(state["messages"])
    print(
        f"  [before_model] Processing for {runtime.context.user_name}, {message_count} messages"
    )
    return None


# After model: log model response
@after_model
def log_after(state: AgentState, runtime: Runtime[Context]) -> dict | None:
    last = state["messages"][-1]
    content_preview = str(last.content)[:80] if last.content else "(no content)"
    print(f"  [after_model] Response preview: {content_preview}")
    return None


# --- 4. Define a tool ---
@tool
def get_fact(topic: str) -> str:
    """Get an interesting fact about a topic."""
    facts = {
        "python": "Python was named after Monty Python's Flying Circus.",
        "coffee": "Finland consumes the most coffee per capita in the world.",
        "space": "A day on Venus is longer than a year on Venus.",
    }
    return facts.get(topic.lower(), f"No fact available for {topic}")


# --- 5. Create the agent with middleware ---
agent = create_agent(
    model=model,
    tools=[get_fact],
    middleware=[personalized_prompt, log_before, log_after],
    context_schema=Context,
)

# --- 6. Run the agent ---
print("=== Agent with Middleware ===")
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Tell me a fun fact about coffee."}]},
    context=Context(user_name="Alice"),
)
print(f"\nFinal: {result['messages'][-1].content}")
