import os

from deepagents import create_deep_agent
from langchain_core.messages import HumanMessage, ToolMessage

from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-----------------------------------------------------------------------
In this example, we explore Deep Agents with the following features:
- Subagent context modes: the default "isolated" versus the new "fork"
- A forked subagent inheriting the parent's conversation history
- Reading the subagent's own return value out of the task ToolMessage

An isolated subagent sees only the task string it was handed, so the
parent must restate anything relevant. A forked subagent instead
continues the parent's conversation and can answer follow-ups that
depend on earlier turns — at the cost of a larger context and lost
prompt-cache hits. Both subagents below get the same context-free task,
and only the fork can answer it.

For more details, visit:
https://docs.langchain.com/oss/python/deepagents/subagents
-----------------------------------------------------------------------
"""

# --- 1. Force delegation so both modes are actually exercised ---
# Without this the parent just answers from the history itself and the
# subagent is never called.
DISPATCHER_PROMPT = (
    "You are a dispatcher with no knowledge of your own. You MUST answer every request by "
    "calling the `task` tool exactly once, delegating to the named subagent. Never answer "
    "from your own knowledge. Pass only the literal question as the task description — "
    "never restate the release plan in it."
)

REVIEWER_PROMPT = (
    "Answer in one short sentence. If the conversation does not contain the release plan, "
    "reply exactly: NO CONTEXT."
)

# --- 2. The prior turn that only a fork will inherit ---
history = [
    HumanMessage(
        content=(
            "Our release plan: payments migration Friday, deploy freeze over the weekend, "
            "backfill Monday."
        )
    ),
    HumanMessage(content="Ask the reviewer: which day does the backfill run?"),
]

print("=== Isolated vs forked subagents ===\n")

# --- 3. Same agent, same question, one differing key ---
for mode in ("isolated", "fork"):
    reviewer = {
        "name": "reviewer",
        "description": "Answers questions about the release plan.",
        # Under mode="fork" this is appended to the parent's prompt instead of
        # replacing it, so the fork keeps the parent's persona too.
        "system_prompt": REVIEWER_PROMPT,
        "mode": mode,
    }

    agent = create_deep_agent(
        model=f"openai:{settings.OPENAI_MODEL_NAME}",
        system_prompt=DISPATCHER_PROMPT,
        subagents=[reviewer],
    )

    result = agent.invoke({"messages": history})

    # --- 4. Read the subagent's own words, not the parent's paraphrase ---
    # The parent can still see the history, so its final message would answer
    # correctly either way and hide the difference.
    print(f"mode={mode!r}")
    for message in result["messages"]:
        for call in getattr(message, "tool_calls", None) or []:
            if call["name"] == "task":
                print(f"  delegated task: {call['args']['description']!r}")
        if isinstance(message, ToolMessage):
            content = message.content
            text = content if isinstance(content, str) else str(content)
            print(f"  subagent returned: {text.strip()}")
    print()

print("Neither task string carried the plan — only the fork inherited it.")
