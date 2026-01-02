import os
import asyncio
import copy
from typing import Optional, Dict, Any

from google.genai import types
from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse, LlmRequest
from google.adk.tools import FunctionTool
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.base_tool import BaseTool

from utils import call_agent_async, print_new_section
from settings import settings

os.environ["GOOGLE_API_KEY"] = settings.GOOGLE_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore Google's ADK agents with the following features:
- Callbacks
- 1. Before & After LLM Call Logic
- 2. Before & After Tool Call Logic
- 3. Before & After Agent Call Logic

This example shows how to create an 3 agents with 
different types of callbacks:
1. Before & After LLM Call Logic
2. Before & After Tool Call Logic
3. Before & After Agent Call Logic

Navigate to the above sections to see how each callback works.
See more details in: https://google.github.io/adk-docs/callbacks/
-------------------------------------------------------
"""


# ----------------------------------------------------------------
#               1. Before & After LLM Call Logic
# ----------------------------------------------------------------

# 1.1. Define the before LLM call logic
def my_before_llm_logic(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> Optional[LlmResponse]:
    """Inspects/modifies the LLM request or skips the call."""
    # Inspect the last user message in the request contents
    last_user_message = ""
    if llm_request.contents and llm_request.contents[-1].role == 'user':
        if llm_request.contents[-1].parts:
            last_user_message = llm_request.contents[-1].parts[0].text
    print(f"[Before LLM Callback] Inspecting last user message: '{last_user_message}'")

    # Add more logic to change the LLM request if needed ...
    # llm_request.config.system_instruction = ...
    
    # Check if the last user message contains "BLOCK"
    if "BLOCK" in last_user_message.upper():
        print("[Before LLM Callback] 'BLOCK' keyword found. Skipping LLM call.")
        # Return an LlmResponse to skip the actual LLM call
        return LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part(text="LLM call was blocked by before_model_callback.")],
            )
        )
    else:
        print("[Before LLMCallback] Proceeding with LLM call.")
        # Return None to allow the (modified) request to go to the LLM
        return None
    
# 1.2. Define the after LLM call logic
def my_after_llm_logic(
    callback_context: CallbackContext, llm_response: LlmResponse
) -> Optional[LlmResponse]:
    """Inspects/modifies the LLM response after it's received."""
    original_text = ""
    if llm_response.content and llm_response.content.parts:
        # Assuming simple text response for this example
        if llm_response.content.parts[0].text:
            original_text = llm_response.content.parts[0].text
            print(f"[After LLM Callback] Inspected original response text: '{original_text[:100]}...'")
        else:
            print("[After LLM Callback] Not text content found in response parts, check for function calls.")
            return None
    elif llm_response.error_message:
        print(f"[After LLM Callback] Inspected response: Contains error '{llm_response.error_message}'. No modification.")
        return None
    else:
        print("[After LLM Callback] Inspected response: Empty LlmResponse.")
        return None

    # --- Modification Example ---
    # Replace "Lisbon" with "Lisbon, Portugal" (case-insensitive)
    search_term = "Lisbon"
    replace_term = "Lisboa"
    if search_term in original_text:
        print(f"[After LLM Callback] Found '{search_term}'. Modifying response.")
        modified_text = original_text.replace(search_term, replace_term)

        # Create a NEW LlmResponse with the modified content
        modified_parts = [copy.deepcopy(part) for part in llm_response.content.parts]
        modified_parts[0].text = modified_text # Update the text in the copied part

        new_response = LlmResponse(
            content=types.Content(role="model", parts=modified_parts),
            # Copy other relevant fields if necessary, e.g., grounding_metadata
            grounding_metadata=llm_response.grounding_metadata
        )
        print(f"[After LLM Callback] Returning modified response.")
        return new_response # Return the modified response
    else:
        print(f"[After LLM Callback] '{search_term}' not found. Passing original response through.")
        # Return None to use the original llm_response
        return None

# 1.3. Create the agent with callbacks
model_callbacks_agent = LlmAgent(
    name="MyCallbackAgent",
    model=settings.GOOGLE_MODEL_NAME,
    instruction="Be helpful.",
    # Other agent parameters...
    before_model_callback=my_before_llm_logic,
    after_model_callback=my_after_llm_logic,
)

# 1.4. Run the agent
print_new_section("1. Before & After LLM Call Logic")
input = "Please block this request."
print("Input: ", input)
asyncio.run(
    call_agent_async(model_callbacks_agent, input)
)
print("\n" + "-" * 50 + "\n")

input = "What is the capital of Portugal?"
print("Input: ", input)
asyncio.run(
    call_agent_async(model_callbacks_agent, input)
)


# ----------------------------------------------------------------
#               2. Before & After Tool Call Logic
# ----------------------------------------------------------------

# 2.1 Define a simple tool
def get_capital_city(country: str) -> str:
    """Retrieves the capital city of a given country."""
    print(f"--- Tool 'get_capital_city' executing with country: {country} ---")
    country_capitals = {
        "portugal": "Lisbon",
        "germany": "Berlin",
        "united states": "Washington, D.C.",
    }
    return country_capitals.get(country.lower(), f"Capital not found for {country}")

capital_tool = FunctionTool(func=get_capital_city)

# 2.2 Create a before tool call modifier
def simple_before_tool_modifier(
    tool: BaseTool, args: Dict[str, Any], tool_context: ToolContext
) -> Optional[Dict]:
    """Inspects/modifies tool args or skips the tool call."""
    agent_name = tool_context.agent_name
    tool_name = tool.name
    print(f"[Before Tool Callback] Before tool call for tool '{tool_name}' in agent '{agent_name}'")
    print(f"[Before Tool Callback] Original args: {args}")

    # If the country is 'Germany', modify the args to 'Portugal'
    if tool_name == 'get_capital_city' and args.get('country', '').lower() == 'germany':
        print("[Before Tool Callback] Detected 'Germany'. Modifying args to 'Portugal'.")
        args['country'] = 'Portugal'
        print(f"[Before Tool Callback] Modified args: {args}")
        return None

    # If the country is 'Spain', skip the tool execution
    if tool_name == 'get_capital_city' and args.get('country', '').lower() == 'spain':
        print("[Before Tool Callback] Detected 'Spain'. Skipping tool execution.")
        return {"result": "Tool execution was blocked by before_tool_callback."}

    print("[Before Tool Callback] Proceeding with original or previously modified args.")
    return None

my_llm_agent = LlmAgent(
    name="ToolCallbackAgent",
    model=settings.GOOGLE_MODEL_NAME,
    instruction="You are an agent that can find capital cities. Use the get_capital_city tool.",
    description="An LLM agent demonstrating before_tool_callback and after_tool_callback.",
    tools=[capital_tool],
    before_tool_callback=simple_before_tool_modifier,
    after_tool_callback=None
)

# 2.4.
print_new_section("2. Before & After Tool Call Logic")
input = "What is the capital of Germany?"
print("Input: ", input)
asyncio.run(
    call_agent_async(my_llm_agent, input, tool_calls=True, tool_call_results=True)
)
print("\n" + "-" * 50 + "\n")

input = "What is the capital of Spain?"
print("Input: ", input)
asyncio.run(
    call_agent_async(my_llm_agent, input, tool_calls=True, tool_call_results=True)
)

# ----------------------------------------------------------------
#               3. Before & After Agent Call Logic
# ----------------------------------------------------------------

# 3.1. Define the before agent call logic
def check_if_agent_should_run(callback_context: CallbackContext) -> Optional[types.Content]:
    """
    Logs entry and checks 'skip_llm_agent' in session state.
    If True, returns Content to skip the agent's execution.
    If False or not present, returns None to allow execution.
    """
    agent_name = callback_context.agent_name
    invocation_id = callback_context.invocation_id
    current_state = callback_context.state.to_dict()

    print(f"\n[Before Agent Callback] Entering agent: {agent_name} (Inv: {invocation_id})")
    print(f"[Before Agent Callback] Current State: {current_state}")

    # Check the condition in session state dictionary
    if current_state.get("skip_llm_agent", False):
        print(f"[Before Agent Callback] State condition 'skip_llm_agent=True' met: Skipping agent {agent_name}.")
        # Return Content to skip the agent's run
        return types.Content(
            role="model",
            parts=[types.Part(text=f"Agent {agent_name} skipped by before_agent_callback due to state.")]
        )
    else:
        print(f"[Before Agent Callback] State condition not met: Proceeding with agent {agent_name}.")
        # Return None to allow the LlmAgent's normal execution
        return None

# 3.2. Define the after agent call logic
def modify_output_after_agent(callback_context: CallbackContext) -> Optional[types.Content]:
    """
    Logs exit from an agent and checks 'add_concluding_note' in session state.
    If True, returns new Content to *replace* the agent's original output.
    If False or not present, returns None, allowing the agent's original output to be used.
    """
    agent_name = callback_context.agent_name
    invocation_id = callback_context.invocation_id
    current_state = callback_context.state.to_dict()

    print(f"[After Agent Callback] Exiting agent: {agent_name} (Inv: {invocation_id})")
    print(f"[After Agent Callback] Current State: {current_state}")

    # Example: Check state to decide whether to modify the final output
    if current_state.get("add_concluding_note", False):
        print(f"[After Agent Callback] State condition 'add_concluding_note=True' met: Replacing agent {agent_name}'s output.")
        # Return Content to *replace* the agent's own output
        return types.Content(
            parts=[types.Part(text=f"Concluding note added by after_agent_callback, replacing original output.")],
            role="model" # Assign model role to the overriding response
        )
    else:
        print(f"[Callback] State condition not met: Using agent {agent_name}'s original output.")
        # Return None - the agent's output produced just before this callback will be used.
        return None
    
# 3.3. Create the agent with callbacks
agent_with_callbacks = LlmAgent(
    name="AgentWithCallbacks",
    model=settings.GOOGLE_MODEL_NAME,
    instruction=(
        "You are an agent that can answer questions. "
        "Use the before_agent_callback and after_agent_callback to modify your behavior."
    ),
    description="An LLM agent demonstrating before_agent_callback and after_agent_callback.",
    before_agent_callback=check_if_agent_should_run,
    after_agent_callback=modify_output_after_agent,
)

# 3.4. Run the agent
print_new_section("3. Before & After Agent Call Logic")
input = "What is the capital of Portugal?"
print("Input: ", input)
asyncio.run(
    call_agent_async(
        agent_with_callbacks, input, tool_calls=True, tool_call_results=True, 
        state={
            "skip_llm_agent": True,
        }
    )
)
print("\n" + "-" * 50 + "\n")