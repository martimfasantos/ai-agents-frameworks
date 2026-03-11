import os
import asyncio

from agents import Agent, Runner, RunState, function_tool

from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore OpenAI's Agents SDK with the following features:
- Human-in-the-loop tool approval
- Pausing and resuming runs with RunState
- Marking tools with needs_approval

When a tool is decorated with needs_approval=True, the runner pauses
execution and returns an interruption.  The caller can inspect the
pending tool call, approve or reject it, and resume the run — enabling
human oversight of sensitive operations.

For more details, visit:
https://openai.github.io/openai-agents-python/human_in_the_loop/
-------------------------------------------------------
"""


# --- 1. Define a tool that always needs approval ---
@function_tool(needs_approval=True)
async def delete_file(filename: str) -> str:
    """Delete a file from the system. Requires human approval."""
    return f"File '{filename}' has been deleted."


# --- 2. Define a regular tool (no approval needed) ---
@function_tool
async def list_files() -> str:
    """List files in the current directory."""
    return "Files: report.csv, notes.txt, temp_data.json"


# --- 3. Create the agent with both tools ---
agent = Agent(
    name="File Manager",
    instructions=(
        "You help manage files. Use list_files to see available files "
        "and delete_file to remove them when asked."
    ),
    tools=[list_files, delete_file],
    model=settings.OPENAI_MODEL_NAME,
)


async def main() -> None:
    # --- 4. Run a request that will trigger the approval flow ---
    print("Asking agent to delete temp_data.json...")
    result = await Runner.run(agent, "Please delete the file temp_data.json")

    # --- 5. Check for interruptions (pending approvals) ---
    if result.interruptions:
        print(f"\nRun paused — {len(result.interruptions)} approval(s) pending:")
        state: RunState = result.to_state()

        for interruption in result.interruptions:
            print(f"  Tool: {interruption.name}, Args: {interruption.arguments}")
            # Simulate approval (in production this would be a human decision)
            print("  -> Auto-approving for demo purposes")
            state.approve(interruption)

        # --- 6. Resume the run after approval ---
        result = await Runner.run(agent, state)
        print(f"\nResumed output: {result.final_output}")
    else:
        print(f"Output (no approval needed): {result.final_output}")


if __name__ == "__main__":
    asyncio.run(main())
