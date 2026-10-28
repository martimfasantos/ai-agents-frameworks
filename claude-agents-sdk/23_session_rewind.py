import asyncio

from dotenv import load_dotenv

from claude_agent_sdk import (
    query,
    ClaudeSDKClient,
    ClaudeAgentOptions,
    ResultMessage,
    ProcessError,
    get_session_messages,
)

from settings import settings

load_dotenv()

"""
-------------------------------------------------------
In this example, we explore Claude Agent SDK with the following features:
- Reading a session transcript with get_session_messages()
- resume_session_at to continue from an earlier message, dropping later turns
- resume_drops_turn to declare which turn the rewind is allowed to discard
- fork_session=True so the original session stays intact

Rewinding is how you build "undo last message" or "edit and resend":
resume at the last entry of the turn you keep, and everything after it
is never loaded. resume_drops_turn makes the truncation checked -- the CLI
refuses the resume when the discarded range is not exactly that turn, for
example a message the session absorbed that your app never saw.

For more details, visit:
https://code.claude.com/docs/en/agent-sdk/python#claudeagentoptions
-------------------------------------------------------
"""

INSTRUCTIONS = "You are a memory assistant. Reply in one short line."
QUESTION = (
    "What are my favourite colour and favourite city? Reply 'colour=<x>, city=<y>', "
    "using 'unknown' for anything I have not told you."
)
REJECTED = "Resume rejected by --resume-drops-turn:"


# --- 1. Build a two-turn session: fact A, then fact B ---
async def build_session() -> str:
    options = ClaudeAgentOptions(model=settings.ANTHROPIC_MODEL_NAME, system_prompt=INSTRUCTIONS, tools=[], max_turns=1)
    session_id = ""

    async with ClaudeSDKClient(options) as client:
        for prompt in [
            "Fact A: my favourite colour is teal. Reply 'Noted A.'",
            "Fact B: my favourite city is Lisbon. Reply 'Noted B.'",
        ]:
            await client.query(prompt)
            print(f"  You:    {prompt}")
            async for message in client.receive_response():
                if isinstance(message, ResultMessage):
                    session_id = message.session_id
                    print(f"  Claude: {message.result}")
    return session_id


# --- 2. Resume at the end of turn 1, declaring turn 2 as the one dropped ---
async def rewind(session_id: str, keep_uuid: str, drop_uuid: str) -> None:
    options = ClaudeAgentOptions(
        model=settings.ANTHROPIC_MODEL_NAME,
        system_prompt=INSTRUCTIONS,
        tools=[],
        max_turns=1,
        resume=session_id,
        resume_session_at=keep_uuid,
        resume_drops_turn=drop_uuid,
        fork_session=True,
    )
    async for message in query(prompt=QUESTION, options=options):
        if isinstance(message, ResultMessage):
            print(f"  Reply: {message.result}")
            print(f"  Forked session: {message.session_id[:8]}")


async def main():
    print("=== Step 1: Two turns in one session ===")
    session_id = await build_session()

    # --- 3. Inspect the transcript and pick the fork point ---
    messages = get_session_messages(session_id)
    user_indices = [i for i, m in enumerate(messages) if m.type == "user"]
    drop_index = user_indices[1]
    keep_uuid = messages[drop_index - 1].uuid
    drop_uuid = messages[drop_index].uuid

    print(f"\n=== Step 2: Transcript of {session_id[:8]} ===")
    for i, m in enumerate(messages):
        marker = ""
        if m.uuid == keep_uuid:
            marker = "  <- resume_session_at (last entry of turn 1)"
        elif m.uuid == drop_uuid:
            marker = "  <- resume_drops_turn (turn 2 prompt)"
        print(f"  [{i}] {m.type:<9} {m.uuid[:8]}{marker}")

    print("\n=== Step 3: Rewind to turn 1 and ask what is known ===")
    await rewind(session_id, keep_uuid, drop_uuid)
    print(f"  Original session still has {len(get_session_messages(session_id))} messages")

    # --- 4. A drop declaration that doesn't match the range is refused ---
    print("\n=== Step 4: Declaring the wrong turn as dropped ===")
    try:
        await rewind(session_id, keep_uuid, messages[user_indices[0]].uuid)
    except ProcessError as error:
        if REJECTED not in str(error):
            raise
        print(f"  Refused with {type(error).__name__}: {error}")


if __name__ == "__main__":
    asyncio.run(main())
