from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_store
from langgraph.store.memory import InMemoryStore

from settings import settings

load_dotenv()

"""
-----------------------------------------------------------------------
In this example, we explore LangGraph with the following features:
- Long-term memory with InMemoryStore
- Storing and retrieving user preferences across threads
- Namespace-based memory organization
- Using store in graph nodes via the store parameter

Unlike persistence (which saves conversation history per thread),
long-term memory stores facts, preferences, and knowledge that persist
across all threads. InMemoryStore provides a key-value store organized
by namespaces, letting you store user profiles, learned preferences,
or accumulated knowledge that any conversation can access.

For more details, visit:
https://docs.langchain.com/oss/python/langgraph/persistence
-----------------------------------------------------------------------
"""

llm = ChatOpenAI(model=settings.OPENAI_MODEL_NAME)

# --- 1. Set up store and checkpointer ---
store = InMemoryStore()
checkpointer = InMemorySaver()


# --- 2. Define graph with store access ---
def chat_node(state: MessagesState) -> dict:
    """Chat node that reads/writes long-term memory."""
    # Access the store via get_store() — injected by the graph runtime
    store = get_store()

    # Retrieve stored preferences for this user
    user_id = "user-123"  # In production, extract from config
    namespace = ("preferences", user_id)

    # Read existing memories
    memories = store.search(namespace)
    memory_context = ""
    if memories:
        memory_context = "Known user preferences:\n"
        for mem in memories:
            memory_context += f"- {mem.key}: {mem.value}\n"

    # Check if the user is telling us a preference
    last_msg = state["messages"][-1].content.lower()
    preference_keywords = ["i like", "i prefer", "my favorite", "i love", "i enjoy"]

    if any(kw in last_msg for kw in preference_keywords):
        # Store the preference
        store.put(
            namespace,
            key=f"pref_{len(memories)}",
            value={"text": state["messages"][-1].content, "type": "preference"},
        )
        memory_context += (
            f"\n(Just stored new preference: {state['messages'][-1].content})"
        )

    # Build system message with memory context
    system_content = (
        "You are a helpful assistant with memory of user preferences. Be concise."
    )
    if memory_context:
        system_content += f"\n\n{memory_context}"

    messages = [SystemMessage(content=system_content)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


# --- 3. Build graph ---
builder = StateGraph(MessagesState)
builder.add_node("chat", chat_node)
builder.add_edge(START, "chat")
builder.add_edge("chat", END)

graph = builder.compile(checkpointer=checkpointer, store=store)

# --- 4. Conversation on thread 1 — teach preferences ---
print("=== Thread 1: Teaching Preferences ===\n")

config1 = {"configurable": {"thread_id": "thread-A"}}

messages_thread1 = [
    "I like programming in Python and I prefer dark mode editors.",
    "I enjoy hiking on weekends.",
]

for msg in messages_thread1:
    result = graph.invoke({"messages": [HumanMessage(content=msg)]}, config=config1)
    print(f"User: {msg}")
    print(f"AI: {result['messages'][-1].content}\n")

# --- 5. Conversation on thread 2 — preferences persist ---
print("=== Thread 2: Preferences Remembered ===\n")

config2 = {"configurable": {"thread_id": "thread-B"}}

result = graph.invoke(
    {"messages": [HumanMessage(content="What do you know about my preferences?")]},
    config=config2,
)
print(f"User: What do you know about my preferences?")
print(f"AI: {result['messages'][-1].content}\n")

# --- 6. Inspect the store ---
print("=== Store Contents ===\n")

all_memories = store.search(("preferences", "user-123"))
print(f"Stored memories: {len(all_memories)}")
for mem in all_memories:
    print(f"  [{mem.key}] {mem.value}")
