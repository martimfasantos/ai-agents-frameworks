from typing import Any, Dict

from dotenv import load_dotenv

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.run.agent import RunInput, RunOutput
from agno.utils.pprint import pprint_run_response

from settings import settings

load_dotenv()

"""
-------------------------------------------------------
In this example, we explore Agno with the following features:
- Pre-hooks that run before the agent processes input
- Post-hooks that run after the agent produces output
- Hook functions with access to RunInput / RunOutput
- Logging, monitoring, and transformation use cases

Hooks provide injection points before and after agent runs.
Pre-hooks can log, validate, or transform input. Post-hooks
can log, audit, or post-process output. Unlike guardrails
(which block requests), hooks are general-purpose callbacks
that don't raise exceptions to stop execution.

For more details, visit:
https://docs.agno.com/agents/hooks
-------------------------------------------------------
"""

# --- 1. Define hook functions ---
call_log: list[Dict[str, Any]] = []


def log_input_hook(run_input: RunInput) -> None:
    """Pre-hook: logs every incoming request."""
    content = str(run_input.input_content)
    entry = {"type": "input", "content": content[:100]}
    call_log.append(entry)
    print(f"  [PRE-HOOK] Received input: {content[:80]}...")


def log_output_hook(run_output: RunOutput) -> None:
    """Post-hook: logs every outgoing response."""
    content = str(run_output.content)[:100] if run_output.content else "(no content)"
    entry = {"type": "output", "content": content}
    call_log.append(entry)
    print(f"  [POST-HOOK] Generated output: {content[:80]}...")


def add_disclaimer_hook(run_output: RunOutput) -> None:
    """Post-hook: appends a disclaimer to the agent's response."""
    if run_output.content and isinstance(run_output.content, str):
        run_output.content += "\n\n---\n*Disclaimer: This is AI-generated content for demonstration purposes.*"
    print("  [POST-HOOK] Disclaimer appended.")


# --- 2. Create the agent with hooks ---
agent = Agent(
    model=OpenAIChat(id=settings.OPENAI_MODEL_NAME),
    pre_hooks=[log_input_hook],
    post_hooks=[log_output_hook, add_disclaimer_hook],
    instructions="You are a helpful assistant. Keep your answers brief (1-2 sentences).",
    markdown=True,
)

# --- 3. Run the agent ---
print("=== Run 1 ===\n")
run_output = agent.run("What is the speed of light?")
pprint_run_response(run_output)

print("\n=== Run 2 ===\n")
run_output = agent.run("Who invented the telephone?")
pprint_run_response(run_output)

# --- 4. Show the collected log ---
print("\n=== Hook call log ===")
for i, entry in enumerate(call_log, 1):
    print(f"  {i}. [{entry['type'].upper()}] {entry['content']}")
