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

Normally a tool just returns data (a string, a dict...) and the graph's
edges decide which node runs next. A tool can instead return a Command,
which carries two things:
- update: values to write into the graph state, like a node's return value
- goto:   the name of the node that should run next

So one tool call can both change state and steer the graph. Here the
triage agent calls a "transfer" tool, and that tool's Command switches
current_department and sends execution straight to the specialist agent,
without any routing edge for it in the graph.

For more details, visit:
https://docs.langchain.com/oss/python/langgraph/graph-api
-----------------------------------------------------------------------
"""


# --- 1. Define state with an extra field ---
class SupportState(MessagesState):
    current_department: str


# --- 2. Define tools that return Commands ---
# Because these tools return a Command instead of plain text, they must add
# their own ToolMessage to "messages". The tool_call_id (injected by LangGraph,
# hidden from the LLM) links that message to the LLM's tool call, as the chat
# API requires every tool call to get a matching response.
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

# The triage agent can only call the two transfer tools, so whatever it calls
# returns a Command that decides where to go next.
# parallel_tool_calls=False stops the LLM from calling both transfers at once:
# that would produce two Commands pointing at two different agents.
triage_llm = llm.bind_tools(
    [transfer_to_billing, transfer_to_technical],
    parallel_tool_calls=False,
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

# A ToolNode runs whatever tool calls the previous agent made. Each agent gets
# its own, so the graph always knows which agent to return to afterwards.
# With one shared ToolNode we would need a fixed edge back to a single agent,
# and that edge would fire *in addition to* the Command's goto, running two
# agents at the same time.
builder.add_node("triage_tools", ToolNode([transfer_to_billing, transfer_to_technical]))
builder.add_node("billing_tools", ToolNode([lookup_account]))
builder.add_node("technical_tools", ToolNode([lookup_account]))

builder.add_edge(START, "triage_agent")

# tools_condition returns "tools" if the agent's last message has tool calls,
# otherwise END. The mapping sends "tools" to that agent's own ToolNode.
builder.add_conditional_edges(
    "triage_agent", tools_condition, {"tools": "triage_tools", END: END}
)
builder.add_conditional_edges(
    "billing_agent", tools_condition, {"tools": "billing_tools", END: END}
)
builder.add_conditional_edges(
    "technical_agent", tools_condition, {"tools": "technical_tools", END: END}
)

# triage_tools has no outgoing edge on purpose: the transfer tool's
# Command(goto=...) picks the next node at runtime. The specialists' tools
# return plain strings, so they need a normal edge back to their agent.
builder.add_edge("billing_tools", "billing_agent")
builder.add_edge("technical_tools", "technical_agent")

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
