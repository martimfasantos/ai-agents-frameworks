import asyncio

from dotenv import load_dotenv

from claude_agent_sdk import (
    query,
    ClaudeAgentOptions,
    AgentDefinition,
    ResultMessage,
    AssistantMessage,
    SystemMessage,
    ToolUseBlock,
)

from settings import settings

load_dotenv()

"""
-------------------------------------------------------
In this example, we explore Claude Agent SDK with the following features:
- Defining subagents with AgentDefinition
- Agent delegation via the "Agent" tool
- Specialized subagents with scoped tools and prompts
- Subagent model selection

Subagents let you decompose complex tasks. Each subagent has its own
prompt, tools, and optionally a different model. The main agent delegates
by calling the "Agent" tool, which spawns a focused subtask. Subagents
run in their own context but return results to the parent.

For more details, visit:
https://platform.claude.com/docs/en/agent-sdk/subagents
-------------------------------------------------------
"""

# --- 1. Define specialized subagents ---
agents = {
    "code_reviewer": AgentDefinition(
        description="Reviews code for bugs, style issues, and best practices.",
        prompt=(
            "You are a senior code reviewer. Analyze code snippets for bugs, "
            "style issues, and best practices. Be concise and actionable."
        ),
        model="sonnet",
    ),
    "doc_writer": AgentDefinition(
        description="Writes clear documentation and docstrings for code.",
        prompt=(
            "You are a documentation specialist. Write clear, concise "
            "docstrings and documentation. Follow Google-style docstring format."
        ),
        model="haiku",
    ),
}

# --- 2. Configure the orchestrator agent ---
options = ClaudeAgentOptions(
    system_prompt=(
        "You are a team lead. You have two subagents available:\n"
        "- code_reviewer: for reviewing code quality\n"
        "- doc_writer: for writing documentation\n"
        "Delegate tasks to the appropriate subagent."
    ),
    agents=agents,
    # Must include "Agent" in allowed_tools to permit delegation
    allowed_tools=["Agent"],
    permission_mode="bypassPermissions",
)


# --- 3. Run a task that requires delegation ---
async def main():
    code_snippet = """
def calc(x, y, op):
    if op == "add":
        return x + y
    elif op == "sub":
        return x - y
    elif op == "mul":
        return x * y
    elif op == "div":
        return x / y
"""

    prompt = (
        f"Here is a Python function:\n```python\n{code_snippet}\n```\n\n"
        "1. Have the code reviewer identify any issues.\n"
        "2. Have the doc writer add a proper docstring."
    )

    async for message in query(prompt=prompt, options=options):
        if isinstance(message, SystemMessage):
            # Subagent lifecycle events
            if message.subtype in (
                "task_started",
                "task_progress",
                "task_notification",
            ):
                print(f"  [{message.subtype}] {message.data}")
        if isinstance(message, ResultMessage) and message.subtype == "success":
            print(f"\n--- Final Result ---\n{message.result}")


if __name__ == "__main__":
    asyncio.run(main())
