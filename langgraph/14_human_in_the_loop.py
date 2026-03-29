from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import interrupt, Command

from settings import settings

load_dotenv()

"""
-----------------------------------------------------------------------
In this example, we explore LangGraph with the following features:
- Human-in-the-loop with interrupt() for approval workflows
- Resuming execution with Command(resume=...)
- Approval/reject patterns for sensitive operations
- Interrupt inside tool execution

The interrupt() function pauses graph execution and returns a value
to the caller. The caller can then inspect the pending action, get
human approval, and resume with Command(resume=value). This is critical
for workflows where certain actions (payments, deletions, deployments)
require human sign-off.

For more details, visit:
https://docs.langchain.com/oss/python/langgraph/interrupts
-----------------------------------------------------------------------
"""

llm = ChatOpenAI(model=settings.OPENAI_MODEL_NAME)
memory = InMemorySaver()


# --- 1. Define tools with approval gates ---
@tool
def get_balance(account_id: str) -> str:
    """Check account balance — no approval needed."""
    balances = {"ACC-001": "$5,432.10", "ACC-002": "$12,891.50"}
    return balances.get(account_id, "Account not found")


@tool
def transfer_funds(from_account: str, to_account: str, amount: float) -> str:
    """Transfer funds between accounts — requires human approval."""
    # This is a sensitive operation — ask for human approval
    approval = interrupt(
        {
            "action": "transfer_funds",
            "from": from_account,
            "to": to_account,
            "amount": amount,
            "message": f"Approve transfer of ${amount:.2f} from {from_account} to {to_account}?",
        }
    )

    if approval == "approved":
        return f"Transfer of ${amount:.2f} from {from_account} to {to_account} completed successfully."
    else:
        return f"Transfer rejected by user. Reason: {approval}"


tools = [get_balance, transfer_funds]
llm_with_tools = llm.bind_tools(tools)


# --- 2. Build the agent ---
def agent_node(state: MessagesState):
    messages = [
        SystemMessage(
            content="You are a banking assistant. You MUST use tools for all operations — never answer without calling the appropriate tool first. For transfers, always use transfer_funds. For balance inquiries, always use get_balance. Be concise."
        )
    ] + state["messages"]
    return {"messages": [llm_with_tools.invoke(messages)]}


builder = StateGraph(MessagesState)
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")

graph = builder.compile(checkpointer=memory)

# --- 3. Start a conversation that triggers approval ---
print("=== Human-in-the-Loop: Approval Flow ===\n")

config = {"configurable": {"thread_id": "hitl-demo"}}

# User requests a transfer — this will trigger interrupt()
result = graph.invoke(
    {"messages": [HumanMessage(content="Transfer $500 from ACC-001 to ACC-002")]},
    config=config,
)

# Check if the graph was interrupted
state = graph.get_state(config)
print(f"Graph status: {'interrupted' if state.next else 'completed'}")
print(f"Next nodes to run: {state.next}\n")

# Show the interrupt payload
if state.tasks:
    for task in state.tasks:
        if hasattr(task, "interrupts") and task.interrupts:
            for intr in task.interrupts:
                print(f"Pending approval: {intr.value}")

# --- 4. Simulate human approval ---
print("\n--- Human approves the transfer ---\n")

result = graph.invoke(
    Command(resume="approved"),
    config=config,
)

# Print the final conversation
for msg in result["messages"]:
    role = msg.__class__.__name__.replace("Message", "")
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        print(f"{role}: [Calling: {[tc['name'] for tc in msg.tool_calls]}]")
    elif hasattr(msg, "content") and msg.content:
        content = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
        print(f"{role}: {content}")
