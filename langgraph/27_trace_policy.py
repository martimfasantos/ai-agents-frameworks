from typing import Any
from uuid import UUID

from dotenv import load_dotenv

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import TracePolicy, omit_payload

from settings import settings

load_dotenv()

"""
-----------------------------------------------------------------------
In this example, we explore LangGraph with the following features:
- Per-node trace shaping with TracePolicy on add_node
- process_inputs to summarize a large payload before it is recorded
- process_outputs and the omit_payload helper to drop a payload entirely

Tracing a graph that carries long message histories or large documents
quickly produces traces nobody can read — and sends payloads you may not
want leaving the process. TracePolicy rewrites what a node records
without touching what the node actually receives or returns, so you can
keep traces small while execution stays identical.

For more details, visit:
https://docs.langchain.com/oss/python/langgraph/graph-api
-----------------------------------------------------------------------
"""


# --- 1. A callback handler that shows what a tracer would record ---
class RecordingHandler(BaseCallbackHandler):
    """Captures the inputs each node reports to the tracing layer."""

    def __init__(self) -> None:
        self.recorded: dict[str, Any] = {}

    def on_chain_start(
        self,
        serialized: dict[str, Any] | None,
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        name = kwargs.get("name") or "unnamed"
        # Only keep the node-level runs, not the graph root or the writers.
        if name in ("summarize_history", "answer"):
            self.recorded[name] = inputs


# --- 2. Trace processors ---
def summarize_history(state: MessagesState) -> dict[str, Any]:
    """Record a count and the last message instead of the whole transcript."""
    messages = state["messages"]
    return {
        "message_count": len(messages),
        "last_message": messages[-1].content if messages else None,
    }


# --- 3. Node bodies — these see the full, untransformed state ---
def summarize_history_node(state: MessagesState) -> dict[str, Any]:
    print(f"  [node] actually received {len(state['messages'])} messages")
    return {"messages": [AIMessage(content=f"Summary of {len(state['messages'])} turns")]}


def answer_node(state: MessagesState) -> dict[str, Any]:
    print(f"  [node] actually received {len(state['messages'])} messages")
    return {"messages": [AIMessage(content="Here is the answer.")]}


# --- 4. Attach a TracePolicy to each node ---
builder = StateGraph(MessagesState)

builder.add_node(
    "summarize_history",
    summarize_history_node,
    # Records a compact summary in place of the full message list.
    trace_policy=TracePolicy(process_inputs=summarize_history),
)
builder.add_node(
    "answer",
    answer_node,
    # Records nothing at all for this node's inputs and outputs.
    trace_policy=TracePolicy(process_inputs=omit_payload, process_outputs=omit_payload),
)

builder.add_edge(START, "summarize_history")
builder.add_edge("summarize_history", "answer")
builder.add_edge("answer", END)

graph = builder.compile()

# --- 5. Run with a deliberately bulky message history ---
print("=== Per-node trace policies ===\n")

history = [
    HumanMessage(content=f"Turn {i}: a long question about model {settings.OPENAI_MODEL_NAME}.")
    for i in range(12)
]

handler = RecordingHandler()
result = graph.invoke({"messages": history}, config={"callbacks": [handler]})

# --- 6. Compare what ran against what was recorded ---
print("\nWhat the tracing layer recorded:")
for node, recorded in handler.recorded.items():
    print(f"  {node}: {recorded}")

print(f"\nFinal message count in state: {len(result['messages'])}")
print("Execution is unaffected — the nodes saw the full history above.")
