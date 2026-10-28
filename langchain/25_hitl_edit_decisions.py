import os
import textwrap
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore LangChain with the following features:
- HITL "edit" decisions that change a pending tool call's arguments
- Redirecting an edited call to a different tool
- HumanInTheLoopMiddleware(edit_notice=) to control what the model is told

An edit leaves the model's own call untouched in its AIMessage and swaps in
the reviewer's version only when the tool runs. The model would otherwise
read a result that does not match the call it made and may try to "fix" it,
so the ToolMessage is prefixed with an edit notice naming what actually ran.
Pass a custom string to reword the notice, or None to turn it off.

For more details, visit:
https://docs.langchain.com/oss/python/langchain/human-in-the-loop
-------------------------------------------------------
"""


def describe(name: str, args: dict[str, Any]) -> str:
    """Render a tool call as name(arg=value, ...)."""
    return f"{name}({', '.join(f'{k}={v!r}' for k, v in args.items())})"


# --- 1. Define the payment tools (mock side effects) ---
@tool
def transfer_funds(to_account: str, amount_eur: float) -> str:
    """Transfer money to an account immediately."""
    return f"Transferred {amount_eur:.2f} EUR to {to_account}."


@tool
def schedule_transfer(to_account: str, amount_eur: float, date: str) -> str:
    """Schedule a transfer to an account for a future date (YYYY-MM-DD)."""
    return f"Scheduled {amount_eur:.2f} EUR to {to_account} on {date}."


model = ChatOpenAI(model=settings.OPENAI_MODEL_NAME, temperature=0)
REQUEST = "Pay 5000 EUR to account PT-7781 for invoice INV-42."


# --- 2. Pause on the transfer, resume with the reviewer's edit, report the outcome ---
def run_with_edit(thread_id: str, edited_action: dict[str, Any], **hitl_options: Any) -> None:
    """Run the payment request and answer its interrupt with an edit decision."""
    agent = create_agent(
        model=model,
        tools=[transfer_funds, schedule_transfer],
        system_prompt="You are a payments assistant. Use the tools to move money.",
        middleware=[HumanInTheLoopMiddleware(interrupt_on={"transfer_funds": True}, **hitl_options)],
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": thread_id}}
    agent.invoke({"messages": [{"role": "user", "content": REQUEST}]}, config=config, version="v2")
    result = agent.invoke(
        Command(resume={"decisions": [{"type": "edit", "edited_action": edited_action}]}),
        config=config,
        version="v2",
    )

    messages = result.value["messages"]
    model_call = next(m for m in messages if m.type == "ai" and m.tool_calls).tool_calls[0]
    notice, _, tool_output = next(m for m in messages if m.type == "tool").content.rpartition("\n")
    print(f"  model's call (AIMessage) : {describe(model_call['name'], model_call['args'])}")
    print(f"  reviewer's edit          : {describe(edited_action['name'], edited_action['args'])}")
    print(f"  tool output (what ran)   : {tool_output}")
    if notice:
        text, _, executed = notice.partition(" Executed instead: ")
        print(f"  ToolMessage starts with  : {textwrap.shorten(text, width=66, placeholder=' ...')}")
        print(f"                             Executed instead: {executed.splitlines()[0]}")
    else:
        print("  ToolMessage starts with  : the tool output (no edit notice)")
    if result.interrupts:
        pending = result.interrupts[0].value["action_requests"][0]
        print(f"  model's next call        : {describe(pending['name'], pending['args'])} <- paused again\n")
    else:
        print(f"  final reply              : {messages[-1].text}\n")


LOWER_AMOUNT = {"name": "transfer_funds", "args": {"to_account": "PT-7781", "amount_eur": 1000}}

# --- 3. Edit the arguments, default edit_notice ---
print("=== Example 1: reviewer lowers the amount (default edit_notice) ===")
run_with_edit("edit-default", LOWER_AMOUNT)

# --- 4. Redirect to a different tool, custom edit_notice ---
print("=== Example 2: reviewer redirects to schedule_transfer (custom edit_notice) ===")
run_with_edit(
    "edit-redirect",
    {"name": "schedule_transfer", "args": {"to_account": "PT-7781", "amount_eur": 5000, "date": "2026-10-30"}},
    edit_notice="A finance reviewer changed this payment before it ran.",
)

# --- 5. Same edit as Example 1 with the notice turned off ---
print("=== Example 3: edit_notice=None ===")
run_with_edit("edit-silent", LOWER_AMOUNT, edit_notice=None)

# --- 6. A redirect must name a tool the agent actually has ---
print("=== Example 4: redirect to a tool the agent does not have ===")
try:
    run_with_edit("edit-unknown", {"name": "wire_abroad", "args": {"iban": "PT50000201231234"}})
except ValueError as exc:
    print(f"  ValueError: {exc}")
