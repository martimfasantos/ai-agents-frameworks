import asyncio

from dotenv import load_dotenv

from claude_agent_sdk import (
    query,
    ClaudeAgentOptions,
    ResultMessage,
    AssistantMessage,
    StreamEvent,
    TextBlock,
)

from settings import settings

load_dotenv()

"""
-------------------------------------------------------
In this example, we explore Claude Agent SDK with the following features:
- Real-time streaming with include_partial_messages=True
- Processing StreamEvent messages for live text updates
- Observing tool call progress in real-time

Streaming lets you display partial responses as the agent generates
them, instead of waiting for the complete result. When enabled,
the message stream includes StreamEvent objects with raw API events
containing text deltas and tool call progress.

For more details, visit:
https://platform.claude.com/docs/en/agent-sdk/streaming-output
-------------------------------------------------------
"""

# --- 1. Configure streaming ---
options = ClaudeAgentOptions(
    include_partial_messages=True,
)


# --- 2. Process streaming events ---
async def main():
    print("Streaming response:\n")
    char_count = 0

    async for message in query(
        prompt="Write a haiku about Python programming.",
        options=options,
    ):
        # StreamEvent contains real-time deltas
        if isinstance(message, StreamEvent):
            event = message.event
            event_type = event.get("type", "")

            # content_block_delta carries text chunks
            if event_type == "content_block_delta":
                delta = event.get("delta", {})
                if delta.get("type") == "text_delta":
                    text = delta.get("text", "")
                    print(text, end="", flush=True)
                    char_count += len(text)

        # Final result after streaming completes
        if isinstance(message, ResultMessage) and message.subtype == "success":
            print(f"\n\n--- Stream complete ---")
            print(f"Characters streamed: {char_count}")
            print(f"Final result: {message.result}")


if __name__ == "__main__":
    asyncio.run(main())
