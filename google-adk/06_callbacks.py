import os
import asyncio
import copy
from typing import Optional, Dict, Any

from google.genai import types
from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_response import LlmResponse
from google.adk.models.llm_request import LlmRequest
from google.adk.tools import FunctionTool
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.base_tool import BaseTool

from settings import settings
from utils import call_agent_async, print_new_section

os.environ["GOOGLE_API_KEY"] = settings.GOOGLE_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore Google ADK with the following features:
- before_model_callback / after_model_callback: intercept and modify LLM calls
- before_tool_callback: inspect and modify tool arguments before execution
- before_agent_callback / after_agent_callback: control agent lifecycle

Callbacks in ADK let you hook into an agent's execution pipeline at key
points — before and after LLM calls, tool calls, and agent invocations.
They enable guardrails (blocking requests), input/output modification,
and observability without changing core agent logic.

For more details, visit:
https://google.github.io/adk-docs/callbacks/
-------------------------------------------------------
"""


# ----------------------------------------------------------------
#              1. Before & After LLM Call Callbacks
# ----------------------------------------------------------------


def my_before_llm_logic(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> Optional[LlmResponse]:
    """Inspects the LLM request and optionally blocks the call."""
    last_user_message = ""
    if llm_request.contents and llm_request.contents[-1].role == "user":
        parts = llm_request.contents[-1].parts
        if parts and parts[0].text:
            last_user_message = parts[0].text
    print(f"  [Before LLM] Last user message: '{last_user_message}'")

    if "BLOCK" in last_user_message.upper():
        print("  [Before LLM] Keyword 'BLOCK' detected — skipping LLM call.")
        return LlmResponse(
            content=types.Content(
                role="model",
                parts=[
                    types.Part(
                        text="This request was blocked by the before_model_callback."
                    )
                ],
            )
        )
    print("  [Before LLM] Proceeding with LLM call.")
    return None


def my_after_llm_logic(
    callback_context: CallbackContext, llm_response: LlmResponse
) -> Optional[LlmResponse]:
    """Modifies the LLM response by replacing 'Lisbon' with 'Lisboa'."""
    if not (llm_response.content and llm_response.content.parts):
        return None
    original_text = llm_response.content.parts[0].text or ""
    print(f"  [After LLM] Original response: '{original_text[:80]}...'")

    if "Lisbon" in original_text:
        modified_text = original_text.replace("Lisbon", "Lisboa")
        print("  [After LLM] Replaced 'Lisbon' with 'Lisboa'.")
        modified_parts = [copy.deepcopy(part) for part in llm_response.content.parts]
        modified_parts[0].text = modified_text
        return LlmResponse(
            content=types.Content(role="model", parts=modified_parts),
            grounding_metadata=llm_response.grounding_metadata,
        )
    return None


model_callbacks_agent = LlmAgent(
    name="ModelCallbackAgent",
    model=settings.GOOGLE_MODEL_NAME,
    instruction="You are a helpful assistant. Respond in 1-2 sentences.",
    before_model_callback=my_before_llm_logic,
    after_model_callback=my_after_llm_logic,
)

print_new_section("1. Before & After LLM Call Callbacks")

query = "Please BLOCK this request."
print(f"Query: {query}")
asyncio.run(call_agent_async(model_callbacks_agent, query))

print("\n" + "-" * 50 + "\n")

query = "What is the capital of Portugal?"
print(f"Query: {query}")
asyncio.run(call_agent_async(model_callbacks_agent, query))

print("\n" + "-" * 50 + "\n")


# ----------------------------------------------------------------
#                 2. Before Tool Call Callback
# ----------------------------------------------------------------


def get_capital_city(country: str) -> str:
    """Returns the capital city of a given country."""
    print(f"  [Tool] get_capital_city called with country='{country}'")
    capitals = {
        "portugal": "Lisbon",
        "germany": "Berlin",
        "united states": "Washington, D.C.",
    }
    return capitals.get(country.lower(), f"Capital not found for '{country}'")


capital_tool = FunctionTool(func=get_capital_city)


def simple_before_tool_modifier(
    tool: BaseTool, args: Dict[str, Any], tool_context: ToolContext
) -> Optional[Dict]:
    """Modifies or blocks tool calls based on the country argument."""
    tool_name = tool.name
    country = args.get("country", "")
    print(f"  [Before Tool] Tool='{tool_name}', Args={args}")

    if tool_name == "get_capital_city" and country.lower() == "germany":
        print("  [Before Tool] Detected 'Germany' — redirecting to 'Portugal'.")
        args["country"] = "Portugal"
        return None  # allow the call with modified args

    if tool_name == "get_capital_city" and country.lower() == "spain":
        print("  [Before Tool] Detected 'Spain' — blocking tool call.")
        return {"result": "Tool execution blocked by before_tool_callback."}

    return None


tool_callback_agent = LlmAgent(
    name="ToolCallbackAgent",
    model=settings.GOOGLE_MODEL_NAME,
    instruction="You find capital cities using the get_capital_city tool. Respond in 1 sentence.",
    tools=[capital_tool],
    before_tool_callback=simple_before_tool_modifier,
)

print_new_section("2. Before Tool Call Callback")

query = "What is the capital of Germany?"
print(f"Query: {query} (will be redirected to Portugal)")
asyncio.run(
    call_agent_async(
        tool_callback_agent, query, tool_calls=True, tool_call_results=True
    )
)

print("\n" + "-" * 50 + "\n")

query = "What is the capital of Spain?"
print(f"Query: {query} (tool call will be blocked)")
asyncio.run(
    call_agent_async(
        tool_callback_agent, query, tool_calls=True, tool_call_results=True
    )
)

print("\n" + "-" * 50 + "\n")


# ----------------------------------------------------------------
#            3. Before & After Agent Call Callbacks
# ----------------------------------------------------------------


def check_if_agent_should_run(
    callback_context: CallbackContext,
) -> Optional[types.Content]:
    """Skips the agent if session state has skip_llm_agent=True."""
    agent_name = callback_context.agent_name
    current_state = callback_context.state.to_dict()
    print(f"  [Before Agent] Agent='{agent_name}', State={current_state}")

    if current_state.get("skip_llm_agent", False):
        print(f"  [Before Agent] Skipping agent '{agent_name}' due to state flag.")
        return types.Content(
            role="model",
            parts=[
                types.Part(
                    text=f"Agent '{agent_name}' was skipped by before_agent_callback."
                )
            ],
        )
    print(f"  [Before Agent] Proceeding with agent '{agent_name}'.")
    return None


def modify_output_after_agent(
    callback_context: CallbackContext,
) -> Optional[types.Content]:
    """Appends a note to the agent's output if add_concluding_note=True in state."""
    agent_name = callback_context.agent_name
    current_state = callback_context.state.to_dict()
    print(f"  [After Agent] Agent='{agent_name}', State={current_state}")

    if current_state.get("add_concluding_note", False):
        print(f"  [After Agent] Appending concluding note to '{agent_name}' output.")
        return types.Content(
            role="model",
            parts=[types.Part(text="[Note added by after_agent_callback] All done!")],
        )
    return None


agent_with_callbacks = LlmAgent(
    name="AgentWithLifecycleCallbacks",
    model=settings.GOOGLE_MODEL_NAME,
    instruction="You are a helpful assistant. Respond in 1 sentence.",
    before_agent_callback=check_if_agent_should_run,
    after_agent_callback=modify_output_after_agent,
)

print_new_section("3. Before & After Agent Call Callbacks")

query = "What is the capital of Portugal?"
print(f"Query: {query} (agent will be skipped)")
asyncio.run(
    call_agent_async(agent_with_callbacks, query, state={"skip_llm_agent": True})
)
