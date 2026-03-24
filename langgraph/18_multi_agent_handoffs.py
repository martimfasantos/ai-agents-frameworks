from typing import Annotated, Literal

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command

from settings import settings

load_dotenv()

"""
-----------------------------------------------------------------------
In this example, we explore LangGraph with the following features:
- Multi-agent handoffs using Command(goto=...) for routing
- Specialist agents that hand control to each other
- Handoff tools that maintain valid conversation history
- Dynamic agent routing in a multi-agent system

Multi-agent handoffs let specialized agents transfer control to each
other via a coordinator. Each agent has handoff tools that use Command
to navigate to a different node in the graph. The coordinator routes
to the appropriate specialist, and specialists can hand back to the
coordinator or to other agents.

For more details, visit:
https://docs.langchain.com/oss/python/langgraph/workflows-agents
-----------------------------------------------------------------------
"""

llm = ChatOpenAI(model=settings.OPENAI_MODEL_NAME)


# --- 1. Define state ---
class AgentState(MessagesState):
    active_agent: str


# --- 2. Define specialist lookup tools ---
@tool
def lookup_pricing(product: str) -> str:
    """Look up pricing for a product."""
    pricing = {
        "basic": "$9.99/month - 10 users, 5GB storage",
        "pro": "$29.99/month - 100 users, 50GB storage",
        "enterprise": "$99.99/month - unlimited users, 500GB storage",
    }
    return pricing.get(
        product.lower(),
        f"No pricing found for '{product}'. Available: basic, pro, enterprise",
    )


@tool
def check_system_status(service: str) -> str:
    """Check the status of a service."""
    statuses = {
        "api": "API: Operational (99.9% uptime last 30 days)",
        "dashboard": "Dashboard: Degraded performance (investigating)",
        "database": "Database: Operational",
    }
    return statuses.get(
        service.lower(),
        f"No status for '{service}'. Available: api, dashboard, database",
    )


# --- 3. Define coordinator with routing logic ---
coordinator_llm = llm.bind_tools([lookup_pricing, check_system_status])


def coordinator_node(
    state: AgentState,
) -> Command[Literal["sales_agent", "support_agent", "coordinator_tools", "__end__"]]:
    """Coordinator that routes to the right specialist or answers directly."""
    messages = [
        SystemMessage(
            content=(
                "You are a coordinator. For pricing and purchasing questions, you should respond with "
                "'I'll transfer you to our sales team' and nothing else. "
                "For technical issues and troubleshooting, respond with "
                "'I'll transfer you to our support team' and nothing else. "
                "For simple questions, answer directly."
            )
        )
    ] + state["messages"]

    response = llm.invoke(messages)

    # Check if the coordinator wants to hand off
    response_lower = response.content.lower()
    if "sales" in response_lower and "transfer" in response_lower:
        return Command(
            update={"messages": [response], "active_agent": "sales"},
            goto="sales_agent",
        )
    elif "support" in response_lower and "transfer" in response_lower:
        return Command(
            update={"messages": [response], "active_agent": "support"},
            goto="support_agent",
        )
    else:
        # Handle directly or use tools
        tool_response = coordinator_llm.invoke(messages)
        if tool_response.tool_calls:
            return Command(
                update={"messages": [tool_response]},
                goto="coordinator_tools",
            )
        return Command(
            update={"messages": [response]},
            goto="__end__",
        )


# --- 4. Define specialist agents ---
sales_llm = llm.bind_tools([lookup_pricing])


def sales_agent(state: AgentState) -> Command[Literal["sales_tools", "__end__"]]:
    """Sales specialist handles pricing and purchasing."""
    messages = [
        SystemMessage(
            content=(
                "You are a sales specialist. Help with pricing, plans, and purchasing questions. "
                "Use lookup_pricing to get product details. Be concise and helpful."
            )
        )
    ] + state["messages"]

    response = sales_llm.invoke(messages)

    if response.tool_calls:
        return Command(update={"messages": [response]}, goto="sales_tools")
    return Command(update={"messages": [response]}, goto="__end__")


support_llm = llm.bind_tools([check_system_status])


def support_agent(state: AgentState) -> Command[Literal["support_tools", "__end__"]]:
    """Support specialist handles technical issues."""
    messages = [
        SystemMessage(
            content=(
                "You are a technical support specialist. Help troubleshoot issues. "
                "Use check_system_status to look up service health. Be concise."
            )
        )
    ] + state["messages"]

    response = support_llm.invoke(messages)

    if response.tool_calls:
        return Command(update={"messages": [response]}, goto="support_tools")
    return Command(update={"messages": [response]}, goto="__end__")


# --- 5. Build the multi-agent graph ---
builder = StateGraph(AgentState)

builder.add_node("coordinator", coordinator_node)
builder.add_node("coordinator_tools", ToolNode([lookup_pricing, check_system_status]))
builder.add_node("sales_agent", sales_agent)
builder.add_node("sales_tools", ToolNode([lookup_pricing]))
builder.add_node("support_agent", support_agent)
builder.add_node("support_tools", ToolNode([check_system_status]))

builder.add_edge(START, "coordinator")
builder.add_edge("coordinator_tools", "coordinator")
builder.add_edge("sales_tools", "sales_agent")
builder.add_edge("support_tools", "support_agent")

graph = builder.compile()

# --- 6. Run: Sales inquiry ---
print("=== Multi-Agent Handoffs ===\n")
print("--- Scenario 1: Sales inquiry ---\n")

result = graph.invoke(
    {
        "messages": [HumanMessage(content="What's the pricing for the pro plan?")],
        "active_agent": "coordinator",
    }
)

for msg in result["messages"]:
    role = msg.__class__.__name__.replace("Message", "")
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        print(f"{role}: [Calling: {[tc['name'] for tc in msg.tool_calls]}]")
    elif hasattr(msg, "content") and msg.content:
        content = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
        print(f"{role}: {content}")

print(f"\nFinal active agent: {result['active_agent']}")

# --- 7. Run: Support inquiry ---
print("\n--- Scenario 2: Support inquiry ---\n")

result = graph.invoke(
    {
        "messages": [
            HumanMessage(
                content="The dashboard is loading very slowly. What's going on?"
            )
        ],
        "active_agent": "coordinator",
    }
)

for msg in result["messages"]:
    role = msg.__class__.__name__.replace("Message", "")
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        print(f"{role}: [Calling: {[tc['name'] for tc in msg.tool_calls]}]")
    elif hasattr(msg, "content") and msg.content:
        content = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
        print(f"{role}: {content}")

print(f"\nFinal active agent: {result['active_agent']}")
