from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    RemoveMessage,
    trim_messages,
)
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import InMemorySaver

from settings import settings

load_dotenv()

"""
-----------------------------------------------------------------------
In this example, we explore LangGraph with the following features:
- Trimming messages to fit context window limits
- Deleting specific messages with RemoveMessage
- Summarizing old messages to compress conversation history

As conversations grow, message history can exceed context limits or
waste tokens. LangGraph provides tools to manage this: trim_messages
cuts by token count, RemoveMessage deletes specific entries, and
summarization compresses old messages into a compact summary while
preserving key context.

For more details, visit:
https://docs.langchain.com/oss/python/langgraph/add-memory
-----------------------------------------------------------------------
"""

llm = ChatOpenAI(model=settings.OPENAI_MODEL_NAME)

# --------------------------------------------------------------
# Example 1: Trim Messages
# --------------------------------------------------------------
print("=== Example 1: Trim Messages ===\n")

# Simulate a long conversation
long_history = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is Python?"),
    AIMessage(
        content="Python is a high-level programming language known for its simplicity and readability."
    ),
    HumanMessage(content="What about JavaScript?"),
    AIMessage(
        content="JavaScript is a versatile language primarily used for web development, both frontend and backend."
    ),
    HumanMessage(content="Compare their type systems."),
    AIMessage(
        content="Python uses dynamic typing with optional type hints. JavaScript also uses dynamic typing but has TypeScript as a typed superset."
    ),
    HumanMessage(content="Which is better for AI/ML?"),
    AIMessage(
        content="Python dominates AI/ML due to libraries like TensorFlow, PyTorch, and scikit-learn."
    ),
    HumanMessage(content="Summarize what we discussed so far."),
]

print(f"Original message count: {len(long_history)}")

# Trim to keep roughly the last few messages by token count
trimmed = trim_messages(
    long_history,
    max_tokens=200,
    strategy="last",
    token_counter=len,  # Simple character-based counter for demo
    allow_partial=False,
    start_on="human",  # Always start on a human message
)

print(f"Trimmed message count: {len(trimmed)}")
for msg in trimmed:
    role = msg.__class__.__name__.replace("Message", "")
    content = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
    print(f"  [{role}] {content}")


# --------------------------------------------------------------
# Example 2: Remove Specific Messages
# --------------------------------------------------------------
print("\n=== Example 2: Remove Specific Messages ===\n")


class ChatState(MessagesState):
    summary: str


def chat_node(state: ChatState) -> dict:
    """Normal chat response."""
    return {"messages": [llm.invoke(state["messages"])]}


def summarize_and_trim_node(state: ChatState) -> dict:
    """Summarize old messages and remove them."""
    messages = state["messages"]

    if len(messages) <= 4:
        return {}

    # Summarize older messages (all but the last 2)
    old_messages = messages[:-2]
    summary_prompt = f"Summarize this conversation concisely:\n\n"
    for msg in old_messages:
        role = "User" if isinstance(msg, HumanMessage) else "AI"
        summary_prompt += f"{role}: {msg.content}\n"

    summary_response = llm.invoke(
        [
            SystemMessage(
                content="Summarize the following conversation in 1-2 sentences."
            ),
            HumanMessage(content=summary_prompt),
        ]
    )

    # Remove old messages using RemoveMessage
    removals = [RemoveMessage(id=msg.id) for msg in old_messages]

    return {
        "summary": summary_response.content,
        "messages": removals,
    }


def route_after_chat(state: ChatState) -> str:
    """Summarize if we have too many messages."""
    if len(state["messages"]) > 6:
        return "summarize"
    return "end"


builder = StateGraph(ChatState)
builder.add_node("chat", chat_node)
builder.add_node("summarize", summarize_and_trim_node)
builder.add_edge(START, "chat")
builder.add_conditional_edges(
    "chat",
    route_after_chat,
    {
        "summarize": "summarize",
        "end": END,
    },
)
builder.add_edge("summarize", END)

memory = InMemorySaver()
graph = builder.compile(checkpointer=memory)

# Simulate a conversation that triggers summarization
config = {"configurable": {"thread_id": "summary-demo"}}

exchanges = [
    "Hi! I'm learning about machine learning.",
    "What's the difference between supervised and unsupervised learning?",
    "Can you give an example of supervised learning?",
    "What about reinforcement learning?",
]

for user_msg in exchanges:
    result = graph.invoke({"messages": [HumanMessage(content=user_msg)]}, config=config)
    ai_response = result["messages"][-1].content
    print(f"User: {user_msg}")
    content = ai_response[:120] + "..." if len(ai_response) > 120 else ai_response
    print(f"AI: {content}\n")

# Check final state
final_state = graph.get_state(config)
print(f"Messages remaining: {len(final_state.values['messages'])}")
if final_state.values.get("summary"):
    print(f"Summary: {final_state.values['summary']}")
