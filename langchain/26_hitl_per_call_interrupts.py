import os
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, Interrupt
from pydantic import ValidationError

from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore LangChain with the following features:
- HumanInTheLoopMiddleware(interrupt_mode="per_call")
- One "tool_approval" interrupt per gated tool call, each with a typed response_schema
- Resuming with Command(resume={interrupt_id: decision, ...})
- Invalid answers rejected with a pydantic ValidationError before being saved

In the default "batched" mode, one interrupt lists every gated call of a model
turn and you answer with an ordered list of decisions. In "per_call" mode each
gated call pauses on its own, so a reviewer answers it by interrupt ID. Each
interrupt carries a JSON schema of the answers that tool accepts: only its
allowed decision types, and for edits the tool's own argument schema. LangGraph
validates the answer against it, so a malformed decision fails fast and the
same interrupt can simply be answered again.

For more details, visit:
https://docs.langchain.com/oss/python/langchain/human-in-the-loop
-------------------------------------------------------
"""


# --- 1. Define the gated tools (mock side effects) ---
@tool
def write_file(path: str, content: str) -> str:
    """Write content to a file."""
    return f"Wrote {len(content)} characters to {path}"


@tool
def execute_sql(query: str) -> str:
    """Run a SQL query against the orders database."""
    return "Deleted 42 rows"


# --- 2. Create the agent with per-call HITL ---
agent = create_agent(
    model=ChatOpenAI(model=settings.OPENAI_MODEL_NAME, temperature=0),
    tools=[write_file, execute_sql],
    system_prompt=(
        "You are a database assistant. When a request needs several tools, call all "
        "of them in the same turn as parallel tool calls. Reply in 1-2 sentences."
    ),
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "write_file": True,  # approve, edit, reject, respond
                "execute_sql": {"allowed_decisions": ["approve", "reject"]},
            },
            interrupt_mode="per_call",
        ),
    ],
    checkpointer=InMemorySaver(),
)
config = {"configurable": {"thread_id": "per-call-demo"}}


def summarize_schema(schema: dict[str, Any]) -> str:
    """Summarize an interrupt's response_schema as allowed decisions + edit args."""
    defs = schema.get("$defs", {})
    mapping = schema.get("discriminator", {}).get("mapping")
    decisions = list(mapping) if mapping else [schema["properties"]["type"]["const"]]
    summary = f"decisions={' | '.join(decisions)}"
    if "EditedAction" in defs:
        args_def = defs["EditedAction"]["properties"]["args"]["$ref"].split("/")[-1]
        fields = ", ".join(f"{n}: {p['type']}" for n, p in defs[args_def]["properties"].items())
        summary += f"; edit args=({fields})"
    return summary


def show_interrupts(interrupts: tuple[Interrupt, ...]) -> None:
    """Print one line per pending tool_approval interrupt."""
    print(f"  {len(interrupts)} pending interrupt(s):")
    for item in interrupts:
        value = item.value
        print(f"    [{value['type']}] id={item.id[:8]}... {value['name']}({value['args']})")
        print(f"        response_schema: {summarize_schema(item.response_schema)}")


def decide(item: Interrupt) -> dict[str, Any]:
    """Reviewer policy: approve the archive write, reject the delete."""
    if item.value["name"] == "execute_sql":
        return {"type": "reject", "message": "Keep the orders until finance signs off. Do not retry."}
    return {"type": "approve"}


# --- 3. Run until the gated calls pause ---
print("=== Step 1: model proposes the calls, each gated call pauses separately ===")
result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Do both now, in parallel: write 'id,total\\n1,20\\n' to "
                    "orders_archive.csv, and run "
                    "DELETE FROM orders WHERE created_at < '2025-01-01'."
                ),
            }
        ]
    },
    config=config,
    version="v2",
)
show_interrupts(result.interrupts)

# --- 4. Send an invalid answer: rejected before it is saved ---
print("\n=== Step 2: an invalid edit is rejected by the response_schema ===")
target = next(
    (item for item in result.interrupts if item.value["name"] == "write_file"),
    result.interrupts[0],
)
bad_edit = {
    "type": "edit",
    "edited_action": {"name": target.value["name"], "args": {"path": "archive/orders_2024.csv"}},
}
print(f"  answer for {target.value['name']}: {bad_edit}")
try:
    agent.invoke(Command(resume={target.id: bad_edit}), config=config, version="v2")
except ValidationError as error:
    for problem in error.errors():
        location = ".".join(str(part) for part in problem["loc"]) or "type"
        print(f"  ValidationError: {location} -> {problem['msg']}")
pending = agent.get_state(config).interrupts
print(f"  nothing was saved: {len(pending)} interrupt(s) still pending")

# --- 5. Answer again by interrupt ID (loops if the model calls the tools sequentially) ---
print("\n=== Step 3: approve write_file, reject execute_sql (keyed by interrupt ID) ===")
while result.interrupts:
    answers = {item.id: decide(item) for item in result.interrupts}
    for item in result.interrupts:
        print(f"  {item.value['name']:<12} -> {answers[item.id]}")
    result = agent.invoke(Command(resume=answers), config=config, version="v2")
    if result.interrupts:
        print("  model proposed another gated call:")
        show_interrupts(result.interrupts)

# --- 6. Show what the model received for each call ---
print("\n=== Result ===")
for message in result.value["messages"]:
    if message.type == "tool":
        print(f"  ToolMessage {message.name:<12} [{message.status}] {message.content}")
print(f"  final reply: {result.value['messages'][-1].text}")
