import textwrap
from collections.abc import Callable

from dotenv import load_dotenv
from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call
from langchain.messages import AnyMessage
from langgraph.checkpoint.memory import InMemorySaver

from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from deepagents.middleware.summarization import SummarizationMiddleware

from settings import settings

load_dotenv()

"""
-----------------------------------------------------------------------
In this example, we explore Deep Agents with the following features:
- SummarizationMiddleware(summarizer=...): swap out who writes the summary
- A plain Python function as the summarizer (deterministic, no LLM call)
- A model string as the summarizer, prompted with a custom summary_prompt
- Replacing the default summarization middleware via middleware=[...]

When the conversation crosses the trigger, older turns are folded into a
summary written by the agent's own model unless summarizer= says otherwise:
a model (prompted with summary_prompt, typically a cheaper one) or a
function that returns the summary text itself. The same four-turn chat runs
under both modes, and we print the summary the agent actually received.

For more details, visit:
https://docs.langchain.com/oss/python/deepagents/context-engineering
-----------------------------------------------------------------------
"""

FACTS = [
    "Our project codename is Bluefin.",
    "The launch date is March 14.",
    "The budget is 2 million euros.",
]
QUESTION = "What are the codename, launch date and budget? Answer in one sentence."


# --- 1. A deterministic summarizer: keeps what the user said, drops the rest ---
summarizer_calls: list[int] = []


def user_facts_summarizer(messages: list[AnyMessage]) -> str:
    """Summarize without an LLM by listing the user's own statements."""
    summarizer_calls.append(len(messages))
    stated = [msg.text for msg in messages if msg.type == "human"]
    return "User-stated facts:\n" + "\n".join(f"- {text}" for text in stated)


# --- 2. A short prompt for the model summarizer ({messages} is filled in) ---
SUMMARY_PROMPT = (
    "Summarize the conversation below as at most 3 short bullet points of facts.\n\n"
    "<messages>\n{messages}\n</messages>"
)


# --- 3. Record what the model receives; the summary is not in result["messages"] ---
model_input: list[AnyMessage] = []


@wrap_model_call
def record_model_input(
    request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Keep the messages sent on the latest model call."""
    model_input[:] = request.messages
    return handler(request)


# --- 4. Run the same multi-turn chat with a given summarization middleware ---
backend = StateBackend()


def run_conversation(label: str, summarization: SummarizationMiddleware) -> None:
    """Run the same multi-turn chat and print the summary the agent received."""
    # Same .name as the built-in instance, so it replaces the default
    agent = create_deep_agent(
        model=f"openai:{settings.OPENAI_MODEL_NAME}",
        backend=backend,
        middleware=[summarization, record_model_input],
        checkpointer=InMemorySaver(),
        system_prompt=(
            "When the user states a fact, reply with exactly 'Noted.'. "
            "Answer questions in one sentence."
        ),
    )
    config = {"configurable": {"thread_id": label}}

    print(f"\n=== {label} ===")
    print(f"Middleware name: {summarization.name}")
    for text in [*FACTS, QUESTION]:
        result = agent.invoke({"messages": [{"role": "user", "content": text}]}, config)
        print(f"User: {text}")
        print(f"  Agent: {result['messages'][-1].text}")

    # The injected summary is a HumanMessage tagged lc_source="summarization"
    summary = next(
        (m for m in model_input if m.additional_kwargs.get("lc_source") == "summarization"),
        None,
    )
    print(f"\nSummarization fired: {summary is not None}")
    print(f"Thread history: {len(result['messages'])} messages; last model call received: {len(model_input)}")
    print("Summary message the model saw instead of the older turns:")
    print(textwrap.indent(summary.text, "    "))


# --- 5. Run both modes: summarize at 6 messages, keep the last 2 verbatim ---
print("=== Deep Agents Custom Summarizer ===")

# --------------------------------------------------------------
# Mode A: plain Python function (deterministic, no LLM call)
# --------------------------------------------------------------
run_conversation(
    "Mode A: summarizer=user_facts_summarizer",
    SummarizationMiddleware(
        f"openai:{settings.OPENAI_MODEL_NAME}",  # still used for token counting
        backend=backend,
        trigger=("messages", 6),
        keep=("messages", 2),
        summarizer=user_facts_summarizer,
    ),
)
print(f"\nFunction summarizer calls (messages received): {summarizer_calls}")

# --------------------------------------------------------------
# Mode B: a model string (point it at a cheaper model in production)
# --------------------------------------------------------------
run_conversation(
    f"Mode B: summarizer='openai:{settings.OPENAI_MODEL_NAME}'",
    SummarizationMiddleware(
        f"openai:{settings.OPENAI_MODEL_NAME}",
        backend=backend,
        trigger=("messages", 6),
        keep=("messages", 2),
        summarizer=f"openai:{settings.OPENAI_MODEL_NAME}",
        summary_prompt=SUMMARY_PROMPT,  # only used by model summarizers
    ),
)
