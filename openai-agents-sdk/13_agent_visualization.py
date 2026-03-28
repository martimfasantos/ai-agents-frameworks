import os
from agents import Agent, function_tool
from agents.extensions.visualization import draw_graph
from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-------------------------------------------------------------------------
In this example, we explore Agent Visualization — generating a graphical
representation of agents, tools, and handoff relationships.

Features demonstrated:
- draw_graph() to visualize agent architecture
- Multiple agents with handoff relationships
- Tools attached to specific agents
- Saving the visualization as a PNG file

The visualization shows:
- Yellow boxes: Agents
- Green ellipses: Tools
- Solid arrows: Handoffs between agents
- Dotted arrows: Tool connections
- __start__ and __end__ nodes for flow entry/exit

Requires: pip install "openai-agents[viz]" and Graphviz installed.
Install Graphviz: brew install graphviz (macOS) or apt install graphviz (Linux)
-------------------------------------------------------------------------
"""


# 1. Define tools for different agents
@function_tool
def search_knowledge_base(query: str) -> str:
    """Search the internal knowledge base for answers."""
    return f"Found 3 articles matching '{query}'"


@function_tool
def create_ticket(subject: str, priority: str) -> str:
    """Create a support ticket."""
    return f"Created ticket: {subject} (priority: {priority})"


@function_tool
def check_billing(account_id: str) -> str:
    """Check billing information for an account."""
    return f"Account {account_id}: $49.99/mo, next billing: Jan 1"


@function_tool
def process_refund(order_id: str, amount: float) -> str:
    """Process a refund for an order."""
    return f"Refund of ${amount:.2f} processed for order {order_id}"


@function_tool
def get_order_status(order_id: str) -> str:
    """Get the current status of an order."""
    return f"Order {order_id}: Shipped, arriving in 2 days"


# 2. Define specialist agents
knowledge_agent = Agent(
    name="Knowledge Agent",
    instructions="You answer questions using the knowledge base.",
    tools=[search_knowledge_base],
    model=settings.OPENAI_MODEL_NAME,
)

billing_agent = Agent(
    name="Billing Agent",
    instructions="You handle billing inquiries and refunds.",
    tools=[check_billing, process_refund],
    model=settings.OPENAI_MODEL_NAME,
)

orders_agent = Agent(
    name="Orders Agent",
    instructions="You handle order status and tracking.",
    tools=[get_order_status],
    model=settings.OPENAI_MODEL_NAME,
)

# 3. Define the triage agent that hands off to specialists
triage_agent = Agent(
    name="Triage Agent",
    instructions=(
        "You are the front-line support agent. Based on the customer's request, "
        "hand off to the appropriate specialist agent:\n"
        "- Knowledge Agent: for general questions\n"
        "- Billing Agent: for billing and refund inquiries\n"
        "- Orders Agent: for order status and tracking"
    ),
    handoffs=[knowledge_agent, billing_agent, orders_agent],
    tools=[create_ticket],
    model=settings.OPENAI_MODEL_NAME,
)


# 4. Visualize the agent graph
print("=== Agent Visualization Example ===\n")
print("Generating agent architecture graph...\n")

# Generate and display the graph
graph = draw_graph(triage_agent)
print("Graph generated! Node legend:")
print("  - Yellow boxes  = Agents")
print("  - Green ellipses = Tools")
print("  - Solid arrows   = Handoffs (agent -> agent)")
print("  - Dotted arrows  = Tool connections")
print()

# 5. Save the graph to a file
os.makedirs("res", exist_ok=True)
output_filename = "res/agent_graph"
draw_graph(triage_agent, filename=output_filename)
print(f"Graph saved to {output_filename}.png")
print()

# 6. Display graph structure textually
print("Agent Architecture:")
print(f"  {triage_agent.name}")
print(f"    Tools: {', '.join(t.name for t in triage_agent.tools)}")
print(f"    Handoffs:")
for handoff_agent in [knowledge_agent, billing_agent, orders_agent]:
    tools_str = ", ".join(t.name for t in handoff_agent.tools)
    print(f"      -> {handoff_agent.name} (tools: {tools_str})")

print("\n=== Agent Visualization Demo Complete ===")
print("Open res/agent_graph.png to see the visual graph!")
