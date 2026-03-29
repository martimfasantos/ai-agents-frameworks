from typing import Annotated

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command

from settings import settings

load_dotenv()

"""
-----------------------------------------------------------------------
In this example, we explore LangGraph with the following features:
- Returning Command objects from tools to update state and route
- Tools that can navigate the graph (goto specific nodes)
- Combining state updates with tool responses

Normally tools just return data. With Command, a tool can also update
graph state and control where execution goes next. This is powerful for
tools that need side effects — like a "transfer to specialist" tool
that updates the active agent and routes to a different node.

For more details, visit:
https://docs.langchain.com/oss/python/langgraph/graph-api
-----------------------------------------------------------------------
"""


# --- 1. Define state with an extra field ---
class SupportState(MessagesState):
    current_department: str


# --- 2. Define tools that return Commands ---
@tool
def transfer_to_billing(
    reason: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Transfer the customer to the billing department."""
    return Command(
        update={
            "current_department": "billing",
            "messages": [
                ToolMessage(
                    content=f"Transferred to billing: {reason}",
                    tool_call_id=tool_call_id,
                )
            ],
        },
        goto="billing_agent",
    )


@tool
def transfer_to_technical(
    reason: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Transfer the customer to technical support."""
    return Command(
        update={
            "current_department": "technical",
            "messages": [
                ToolMessage(
                    content=f"Transferred to technical: {reason}",
                    tool_call_id=tool_call_id,
                )
            ],
        },
        goto="technical_agent",
    )


@tool
def lookup_account(account_id: str) -> str:
    """Look up account information."""
    accounts = {
        "ACC-001": "Plan: Premium, Balance: $142.50, Status: Active",
        "ACC-002": "Plan: Basic, Balance: $0.00, Status: Suspended",
    }
    return accounts.get(account_id, f"Account {account_id} not found")


# --- 3. Define agent nodes ---
llm = ChatOpenAI(model=settings.OPENAI_MODEL_NAME)

triage_llm = llm.bind_tools(
    [transfer_to_billing, transfer_to_technical, lookup_account]
)
billing_llm = llm.bind_tools([lookup_account])
technical_llm = llm.bind_tools([lookup_account])


def triage_agent(state: SupportState):
    """Front-line agent that routes to specialists."""
    messages = [
        {
            "role": "system",
            "content": "You are a triage agent. For billing questions, transfer to billing. For technical issues, transfer to technical. Use lookup_account if the user asks about their account. Be concise.",
        }
    ] + state["messages"]
    return {"messages": [triage_llm.invoke(messages)]}


def billing_agent(state: SupportState):
    """Billing specialist."""
    messages = [
        {
            "role": "system",
            "content": "You are a billing specialist. Help with payment and billing questions. Be concise.",
        }
    ] + state["messages"]
    return {"messages": [billing_llm.invoke(messages)]}


def technical_agent(state: SupportState):
    """Technical support specialist."""
    messages = [
        {
            "role": "system",
            "content": "You are a technical support specialist. Help with technical issues. Be concise.",
        }
    ] + state["messages"]
    return {"messages": [technical_llm.invoke(messages)]}


# --- 4. Build the graph ---
builder = StateGraph(SupportState)

builder.add_node("triage_agent", triage_agent)
builder.add_node("billing_agent", billing_agent)
builder.add_node("technical_agent", technical_agent)
builder.add_node(
    "tools", ToolNode([transfer_to_billing, transfer_to_technical, lookup_account])
)

builder.add_edge(START, "triage_agent")
builder.add_conditional_edges("triage_agent", tools_condition)
builder.add_conditional_edges("billing_agent", tools_condition)
builder.add_conditional_edges("technical_agent", tools_condition)
builder.add_edge("tools", "triage_agent")
builder.add_edge("billing_agent", END)
builder.add_edge("technical_agent", END)

graph = builder.compile()

# --- 5. Run — the triage agent should transfer to billing ---
print("=== Command from Tools ===\n")

result = graph.invoke(
    {
        "messages": [
            (
                "user",
                "I have a question about my bill for account ACC-001. Why was I charged $142.50?",
            )
        ],
        "current_department": "triage",
    }
)

print(f"Final department: {result['current_department']}\n")
for msg in result["messages"]:
    role = msg.__class__.__name__.replace("Message", "")
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        print(f"{role}: [Calling: {[tc['name'] for tc in msg.tool_calls]}]")
    elif hasattr(msg, "content") and msg.content:
        content = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
        print(f"{role}: {content}")
