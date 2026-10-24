import asyncio

from dotenv import load_dotenv

from claude_agent_sdk import query, ClaudeAgentOptions, ResultMessage

from settings import settings

load_dotenv()

"""
-------------------------------------------------------
In this example, we explore Claude Agent SDK with the following features:
- verbatim_prompts=True to deliver prompt text exactly as written
- How Claude Code expands an @path file mention in a prompt by default
- Passing untrusted text (reviews, emails, tool output) to the agent safely

By default Claude Code treats a prompt like something typed into its
terminal: "@settings.py" reads that local file and attaches it, and a
leading "/" runs a slash command. When the text comes from someone else,
that lets them pull local files into the model's context. With
verbatim_prompts=True every user message is sent marked client_composed,
so the CLI passes it through as plain text (needs Claude Code 2.1.248+).

For more details, visit:
https://code.claude.com/docs/en/agent-sdk/python#claudeagentoptions
-------------------------------------------------------
"""

# --- 1. Untrusted text that happens to contain a file mention ---
UNTRUSTED_REVIEW = "Great product! See @settings.py for my notes."

# Makes the model report what actually reached it
ECHO_INSTRUCTIONS = (
    "You are an echo service. Reply with exactly one line. "
    "If file contents were attached to the message, reply 'FILE SEEN: ' "
    "followed by the first line of that file. Otherwise reply 'LITERAL: ' "
    "followed by the message text exactly as received."
)


# --- 2. Send the same text with and without verbatim_prompts ---
async def echo(verbatim: bool) -> str:
    options = ClaudeAgentOptions(
        model=settings.ANTHROPIC_MODEL_NAME,
        system_prompt=ECHO_INSTRUCTIONS,
        tools=[],  # No Read tool, so file contents can only come from @ expansion
        max_turns=1,
        verbatim_prompts=verbatim,
    )

    reply = ""
    async for message in query(prompt=UNTRUSTED_REVIEW, options=options):
        if isinstance(message, ResultMessage):
            reply = message.result or ""
    return reply


async def main():
    print(f"Untrusted input: {UNTRUSTED_REVIEW}")

    # --- 3. Default: Claude Code interprets the @ mention ---
    print("\n=== Default: @settings.py is expanded ===")
    print(f"  Model saw: {await echo(verbatim=False)}")

    # --- 4. verbatim_prompts: the mention stays inert text ---
    print("\n=== verbatim_prompts=True: delivered as written ===")
    print(f"  Model saw: {await echo(verbatim=True)}")

    print("\n=== What verbatim_prompts turns off ===")
    print("@path mentions:  no local file is read and attached")
    print("/commands:       a leading '/' is not run as a slash command")
    print("Per-message:     leave it off and set \"client_composed\": True on streamed messages")


if __name__ == "__main__":
    asyncio.run(main())
