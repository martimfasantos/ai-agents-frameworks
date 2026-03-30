import asyncio
import os
import tempfile

from dotenv import load_dotenv

from claude_agent_sdk import (
    ClaudeSDKClient,
    ClaudeAgentOptions,
    ResultMessage,
    UserMessage,
)

from settings import settings

load_dotenv()

"""
-------------------------------------------------------
In this example, we explore Claude Agent SDK with the following features:
- File checkpointing with enable_file_checkpointing=True
- extra_args for replay-user-messages to capture checkpoint UUIDs
- Tracking file changes made by the agent during a session
- Rewinding files to a previous state with rewind_files()

File checkpointing tracks every file modification the agent makes,
creating snapshots at each turn. You can rewind all files to their
state at any previous user message, which is useful for recovery,
A/B testing agent approaches, or undoing unwanted changes.

For more details, visit:
https://platform.claude.com/docs/en/agent-sdk/file-checkpointing
-------------------------------------------------------
"""

# --- 1. Set up a temporary working directory ---
work_dir = tempfile.mkdtemp(prefix="claude_checkpoint_")
print(f"Working directory: {work_dir}")

# --- 2. Create initial file for the agent to modify ---
test_file = os.path.join(work_dir, "example.txt")
with open(test_file, "w") as f:
    f.write("Original content: Hello World\n")
print(f"Initial file content: {open(test_file).read().strip()}")


# --- 3. Run the agent with file checkpointing enabled ---
async def main():
    options = ClaudeAgentOptions(
        enable_file_checkpointing=True,
        cwd=work_dir,
        allowed_tools=["Read", "Write"],
        permission_mode="acceptEdits",
        # Required to receive UserMessage UUIDs in the response stream
        extra_args={"replay-user-messages": None},
    )

    checkpoint_id = None
    session_id = None

    async with ClaudeSDKClient(options) as client:
        # Turn 1: Ask the agent to modify the file
        print("\n=== Turn 1: Modify the file ===")
        await client.query(
            "Read example.txt, then overwrite it with 'Modified by agent: version 2'"
        )
        async for message in client.receive_response():
            if isinstance(message, UserMessage) and message.uuid and not checkpoint_id:
                checkpoint_id = message.uuid
                print(f"Checkpoint UUID: {checkpoint_id}")
            if isinstance(message, ResultMessage):
                session_id = message.session_id
                print(f"Result: {message.result}")
                content = open(test_file).read().strip()
                print(f"File now contains: {content}")

        # Turn 2: Ask for another modification
        print("\n=== Turn 2: Modify again ===")
        await client.query("Overwrite example.txt with 'Modified again: version 3'")
        async for message in client.receive_response():
            if isinstance(message, ResultMessage):
                content = open(test_file).read().strip()
                print(f"File now contains: {content}")

    # --- 4. Rewind by resuming the session with an empty prompt ---
    if checkpoint_id and session_id:
        print("\n=== Rewinding to Turn 1 checkpoint ===")
        async with ClaudeSDKClient(
            ClaudeAgentOptions(
                enable_file_checkpointing=True,
                resume=session_id,
                cwd=work_dir,
            )
        ) as client:
            await client.query("")  # Empty prompt to open the connection
            async for message in client.receive_response():
                await client.rewind_files(checkpoint_id)
                break

        content = open(test_file).read().strip()
        print(f"File after rewind: {content}")
    else:
        print("\nNote: Could not capture checkpoint ID or session ID.")


if __name__ == "__main__":
    asyncio.run(main())
