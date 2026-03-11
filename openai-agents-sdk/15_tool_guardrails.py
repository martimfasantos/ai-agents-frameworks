import os
import json

from agents import (
    Agent,
    Runner,
    ToolGuardrailFunctionOutput,
    ToolInputGuardrailData,
    ToolOutputGuardrailData,
    function_tool,
    tool_input_guardrail,
    tool_output_guardrail,
)

from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore OpenAI's Agents SDK with the following features:
- Tool input guardrails (pre-execution validation)
- Tool output guardrails (post-execution validation)
- Rejecting or allowing tool calls at the tool level

Tool guardrails wrap individual function tools and validate inputs
before execution and outputs after execution.  Unlike agent-level
guardrails, these fire on every invocation of the tool regardless of
which agent calls it.  Here we block secrets in tool input and redact
sensitive data from tool output.

For more details, visit:
https://openai.github.io/openai-agents-python/guardrails/#tool-guardrails
-------------------------------------------------------
"""


# --- 1. Define a tool input guardrail that blocks secrets ---
@tool_input_guardrail
def block_secrets(data: ToolInputGuardrailData) -> ToolGuardrailFunctionOutput:
    """Reject tool calls whose arguments contain API-key-like strings."""
    args = json.loads(data.context.tool_arguments or "{}")
    if "sk-" in json.dumps(args):
        return ToolGuardrailFunctionOutput.reject_content(
            "Remove secrets before calling this tool."
        )
    return ToolGuardrailFunctionOutput.allow()


# --- 2. Define a tool output guardrail that redacts sensitive data ---
@tool_output_guardrail
def redact_output(data: ToolOutputGuardrailData) -> ToolGuardrailFunctionOutput:
    """Replace tool output if it contains secret-like strings."""
    text = str(data.output or "")
    if "sk-" in text:
        return ToolGuardrailFunctionOutput.reject_content(
            "Output contained sensitive data and was redacted."
        )
    return ToolGuardrailFunctionOutput.allow()


# --- 3. Define the tool with both guardrails attached ---
@function_tool(
    tool_input_guardrails=[block_secrets],  # type: ignore[list-item]
    tool_output_guardrails=[redact_output],  # type: ignore[list-item]
)
def classify_text(text: str) -> str:
    """Classify the given text for internal routing."""
    return f"classification: general | length: {len(text)}"


# --- 4. Create the agent ---
agent = Agent(
    name="Classifier",
    instructions="Classify user messages using the classify_text tool.",
    tools=[classify_text],
    model=settings.OPENAI_MODEL_NAME,
)

# --- 5. Run with a safe input ---
result = Runner.run_sync(agent, "Classify this message: hello world")
print(f"Safe input result: {result.final_output}")

# --- 6. Run with an input containing a secret (guardrail should block) ---
result2 = Runner.run_sync(agent, "Classify this message: my key is sk-abc123secret")
print(f"Secret input result: {result2.final_output}")
