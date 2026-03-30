import asyncio

from dotenv import load_dotenv

from claude_agent_sdk import (
    query,
    ClaudeSDKClient,
    ClaudeAgentOptions,
    ResultMessage,
    AssistantMessage,
    TextBlock,
)

from settings import settings

load_dotenv()

"""
-------------------------------------------------------
In this example, we explore Claude Agent SDK with the following features:
- Session management with resume, continue_conversation, and fork_session
- Retrieving session IDs from ResultMessage
- Continuing a previous conversation by session ID

Sessions allow an agent to maintain context across multiple interactions.
You can resume an exact session by ID, continue the most recent session
in a directory, or fork a session to branch a conversation without
modifying the original.

For more details, visit:
https://platform.claude.com/docs/en/agent-sdk/sessions
-------------------------------------------------------
"""

# --- 1. Start a new session and capture its ID ---
print("=== Step 1: Start a new session ===")


async def start_session() -> str:
    """Start a conversation and return the session ID."""
    session_id = ""

    async for message in query(
        prompt="Remember this: the secret code is ALPHA-7. Confirm you stored it.",
        options=ClaudeAgentOptions(),
    ):
        if isinstance(message, ResultMessage):
            session_id = message.session_id
            if message.subtype == "success":
                print(f"Response: {message.result}")
                print(f"Session ID: {session_id}")

    return session_id


saved_session_id = asyncio.run(start_session())

# --- 2. Resume the session by ID ---
print("\n=== Step 2: Resume the session ===")


async def resume_session(session_id: str):
    """Resume a previous session and ask about stored context."""
    options = ClaudeAgentOptions(resume=session_id)

    async for message in query(
        prompt="What was the secret code I told you earlier?",
        options=options,
    ):
        if isinstance(message, ResultMessage) and message.subtype == "success":
            print(f"Response: {message.result}")


asyncio.run(resume_session(saved_session_id))

# --- 3. Fork the session ---
print("\n=== Step 3: Fork the session ===")


async def fork_and_diverge(session_id: str):
    """Fork a session to branch the conversation."""
    options = ClaudeAgentOptions(
        resume=session_id,
        fork_session=True,
    )

    async for message in query(
        prompt="Forget the code. What is 2 + 2?",
        options=options,
    ):
        if isinstance(message, ResultMessage):
            if message.subtype == "success":
                print(f"Forked session response: {message.result}")
                print(f"New session ID: {message.session_id}")
                print(f"(Original session '{session_id}' is unchanged)")


asyncio.run(fork_and_diverge(saved_session_id))
