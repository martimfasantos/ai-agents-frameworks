import os
import tempfile

from autogen import ConversableAgent, LLMConfig
from autogen.coding import LocalCommandLineCodeExecutor

from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore AG2 with the following features:
- Code execution using LocalCommandLineCodeExecutor
- Code writer agent and code executor agent pattern
- Automatic extraction and execution of code blocks

AG2 agents can generate code and execute it locally.
A code writer agent produces Python code blocks, and a
code executor agent runs them using a command-line executor,
returning the output for further conversation.

For more details, visit:
https://docs.ag2.ai/latest/docs/user-guide/advanced-concepts/code-execution/
-------------------------------------------------------
"""

# --- 1. Configure LLM ---
llm_config = LLMConfig({"model": settings.OPENAI_MODEL_NAME})

# --- 2. Create a temporary directory for code execution ---
temp_dir = tempfile.TemporaryDirectory()

# --- 3. Create the code executor ---
executor = LocalCommandLineCodeExecutor(
    timeout=30,
    work_dir=temp_dir.name,
)

# --- 4. Create the code executor agent ---
code_executor = ConversableAgent(
    name="code_executor",
    llm_config=False,
    code_execution_config={"executor": executor},
    human_input_mode="NEVER",
    is_termination_msg=lambda x: "TERMINATE" in (x.get("content", "") or ""),
)

# --- 5. Create the code writer agent ---
code_writer = ConversableAgent(
    name="code_writer",
    system_message=(
        "You are a helpful coding assistant. Solve tasks using Python code. "
        "Always put code in a ```python code block. "
        "Do not suggest incomplete code. Do not use input(). "
        "Check the execution result and fix errors if any. "
        "When the task is done, reply with TERMINATE."
    ),
    llm_config=llm_config,
    code_execution_config=False,
    human_input_mode="NEVER",
)

# --- 6. Run a coding task ---
print("=== Code Execution: Fibonacci Calculator ===\n")

result = code_executor.initiate_chat(
    code_writer,
    message="Write Python code to calculate and print the first 10 Fibonacci numbers.",
    max_turns=3,
)

print("\n=== Code Execution Complete ===")
print(f"Chat turns: {len(result.chat_history)}")

# --- 7. Clean up ---
temp_dir.cleanup()
