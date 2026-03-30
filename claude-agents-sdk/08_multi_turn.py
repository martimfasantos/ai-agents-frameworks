import asyncio

from dotenv import load_dotenv

from claude_agent_sdk import (
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
- ClaudeSDKClient for multi-turn conversations
- Async context manager for session lifecycle management
- Sending multiple prompts within the same session
- Accessing session_id and conversation history

ClaudeSDKClient manages a persistent connection for multi-turn
conversations. Unlike query() which is one-shot, the client
maintains session state and lets you send multiple prompts while
keeping full conversation context.

For more details, visit:
https://platform.claude.com/docs/en/agent-sdk/streaming-vs-single-mode
-------------------------------------------------------
"""


# --- 1. Create a multi-turn conversation ---
async def main():
    options = ClaudeAgentOptions()

    async with ClaudeSDKClient(options) as client:
        # Turn 1: Set context
        print("=== Turn 1 ===")
        await client.query(
            "I'm building a REST API with FastAPI. Remember this context."
        )
        async for message in client.receive_response():
            if isinstance(message, ResultMessage) and message.subtype == "success":
                print(f"Response: {message.result}")
                print(f"Session: {message.session_id}")

        # Turn 2: Build on previous context
        print("\n=== Turn 2 ===")
        await client.query(
            "What authentication library would you recommend for my project?"
        )
        async for message in client.receive_response():
            if isinstance(message, ResultMessage) and message.subtype == "success":
                print(f"Response: {message.result}")

        # Turn 3: Continue the thread
        print("\n=== Turn 3 ===")
        await client.query(
            "Show me a minimal example of JWT auth with that library. Keep it under 20 lines."
        )
        async for message in client.receive_response():
            if isinstance(message, ResultMessage) and message.subtype == "success":
                print(f"Response: {message.result}")


if __name__ == "__main__":
    asyncio.run(main())
