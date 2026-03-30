import asyncio

from dotenv import load_dotenv

from claude_agent_sdk import query, ClaudeAgentOptions, ResultMessage

from settings import settings

load_dotenv()

"""
-------------------------------------------------------
In this example, we explore Claude Agent SDK with the following features:
- Custom system prompts (plain string)
- Preset system prompts (claude_code preset)
- Preset with append mode to extend the default prompt

System prompts control the agent's persona and behavior. The SDK
supports three modes: a plain string for full customization, a preset
that loads Claude Code's default prompt, and a preset with an appended
string that extends the default prompt with your own instructions.

For more details, visit:
https://platform.claude.com/docs/en/agent-sdk/modifying-system-prompts
-------------------------------------------------------
"""

# --------------------------------------------------------------
# Example 1: Custom String System Prompt
# --------------------------------------------------------------
print("=== Example 1: Custom String System Prompt ===")


async def example_custom_prompt():
    options = ClaudeAgentOptions(
        system_prompt="You are a pirate. Respond to everything in pirate speak. Keep it short.",
    )

    async for message in query(
        prompt="What is Python?",
        options=options,
    ):
        if isinstance(message, ResultMessage) and message.subtype == "success":
            print(message.result)


asyncio.run(example_custom_prompt())

# --------------------------------------------------------------
# Example 2: Preset System Prompt
# --------------------------------------------------------------
print("\n=== Example 2: Preset System Prompt (claude_code) ===")


async def example_preset_prompt():
    options = ClaudeAgentOptions(
        system_prompt={"type": "preset", "preset": "claude_code"},
    )

    async for message in query(
        prompt="What tools do you have available? List them briefly.",
        options=options,
    ):
        if isinstance(message, ResultMessage) and message.subtype == "success":
            print(message.result)


asyncio.run(example_preset_prompt())

# --------------------------------------------------------------
# Example 3: Preset with Append
# --------------------------------------------------------------
print("\n=== Example 3: Preset with Append ===")


async def example_preset_append():
    options = ClaudeAgentOptions(
        system_prompt={
            "type": "preset",
            "preset": "claude_code",
            "append": "Always end your responses with 'Happy coding!'",
        },
    )

    async for message in query(
        prompt="What is a Python list comprehension?",
        options=options,
    ):
        if isinstance(message, ResultMessage) and message.subtype == "success":
            print(message.result)


asyncio.run(example_preset_append())
