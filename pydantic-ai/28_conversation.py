import asyncio

from dotenv import load_dotenv

from pydantic_ai import Agent, AgentRunResult, ConversationTypeAdapter, UsageLimits
from pydantic_ai.exceptions import UsageLimitExceeded

from settings import settings

load_dotenv()

"""
-----------------------------------------------------------------------
In this example, we explore Pydantic AI with the following features:
- result.conversation: messages, running usage and conversation_id in one object
- agent.run(..., conversation=...) to continue a conversation whole
- ConversationTypeAdapter to store a conversation as JSON and restore it
- UsageLimits that apply across every turn, not just the current one

message_history= carries only the messages, so a conversation reassembled
from it silently loses its running usage: every turn's UsageLimits budget
starts over from zero and a conversation-wide cap is never reached. A
Conversation carries the messages, the usage and the conversation_id (plus
any pending deferred tool calls) together; passing it alongside
message_history=, usage= or conversation_id= raises UserError.

For more details, visit:
https://pydantic.dev/docs/ai/core-concepts/message-history/#carrying-a-conversation-whole
-----------------------------------------------------------------------
"""

agent = Agent(
    model=settings.OPENAI_MODEL_NAME,
    instructions="Be concise. Reply in one short sentence.",
)

# The same budget is passed to every turn: two model requests per conversation
BUDGET = UsageLimits(request_limit=2)


def show_turn(prompt: str, result: AgentRunResult[str]) -> None:
    usage = result.conversation.usage
    print(f"User:  {prompt}")
    print(f"Agent: {result.output}")
    print(f"conversation_id: {result.conversation_id}")
    print(f"running usage:   requests={usage.requests}, tokens={usage.total_tokens}")


async def main() -> None:
    # --- 1. First turn starts a conversation ---
    print("=== Turn 1 ===")
    prompt = "Hi, I'm Ana and I'm flying to Porto next week."
    first = await agent.run(prompt, usage_limits=BUDGET)
    show_turn(prompt, first)

    # --- 2. Store the whole conversation and restore it ---
    print("\n=== Store and restore with ConversationTypeAdapter ===")
    stored = ConversationTypeAdapter.dump_json(first.conversation)
    restored = ConversationTypeAdapter.validate_json(stored)
    print(f"Saved {len(stored)} bytes of JSON")
    print(
        f"Restored: {len(restored.messages)} messages, "
        f"requests={restored.usage.requests}, "
        f"same id: {restored.conversation_id == first.conversation_id}"
    )

    # --- 3. Continue from the restored copy: the agent recalls turn 1 ---
    print("\n=== Turn 2 (continued from the restored conversation) ===")
    prompt = "What's my name, and which city am I visiting?"
    second = await agent.run(prompt, conversation=restored, usage_limits=BUDGET)
    show_turn(prompt, second)

    # --- 4. The budget counts every turn, so the third one is refused ---
    print("\n=== Turn 3 with conversation= ===")
    prompt = "What should I eat there?"
    try:
        await agent.run(prompt, conversation=second.conversation, usage_limits=BUDGET)
    except UsageLimitExceeded as exc:
        print(f"UsageLimitExceeded: {exc}")

    # --- 5. Contrast: message_history= alone drops the running usage ---
    print("\n=== Turn 3 with message_history= only ===")
    third = await agent.run(
        prompt, message_history=second.all_messages(), usage_limits=BUDGET
    )
    print(f"Agent: {third.output}")
    print(f"usage restarted at requests={third.usage.requests}, so the cap never bit")


if __name__ == "__main__":
    asyncio.run(main())
