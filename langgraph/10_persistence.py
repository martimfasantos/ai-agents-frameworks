from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import InMemorySaver

from settings import settings

load_dotenv()

"""
-----------------------------------------------------------------------
In this example, we explore LangGraph with the following features:
- Persistence with InMemorySaver checkpointer
- Thread-based conversation memory (thread_id)
- Multi-turn conversations that remember context
- Inspecting state with get_state()

Persistence gives your graph memory across invocations. By attaching a
checkpointer, each invocation's state is saved and restored using a
thread_id. This enables multi-turn chatbots where the LLM remembers
the full conversation history without you manually managing messages.

For more details, visit:
https://docs.langchain.com/oss/python/langgraph/persistence
-----------------------------------------------------------------------
"""

# --- 1. Create graph with persistence ---
llm = ChatOpenAI(model=settings.OPENAI_MODEL_NAME)


def chatbot(state: MessagesState):
    return {"messages": [llm.invoke(state["messages"])]}


builder = StateGraph(MessagesState)
builder.add_node("chatbot", chatbot)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

# Attach checkpointer for persistence
memory = InMemorySaver()
graph = builder.compile(checkpointer=memory)

# --- 2. Multi-turn conversation on thread 1 ---
print("=== Thread 1: Multi-turn Conversation ===\n")

config_thread1 = {"configurable": {"thread_id": "thread-1"}}

# Turn 1
response = graph.invoke(
    {
        "messages": [
            HumanMessage(content="My name is Alice and I'm a software engineer.")
        ]
    },
    config=config_thread1,
)
print(f"User: My name is Alice and I'm a software engineer.")
print(f"AI: {response['messages'][-1].content}\n")

# Turn 2 — the graph remembers Alice's name and profession
response = graph.invoke(
    {"messages": [HumanMessage(content="What's my name and profession?")]},
    config=config_thread1,
)
print(f"User: What's my name and profession?")
print(f"AI: {response['messages'][-1].content}\n")

# --- 3. Separate thread has no memory of thread 1 ---
print("=== Thread 2: Independent Conversation ===\n")

config_thread2 = {"configurable": {"thread_id": "thread-2"}}

response = graph.invoke(
    {"messages": [HumanMessage(content="Do you know my name?")]},
    config=config_thread2,
)
print(f"User: Do you know my name?")
print(f"AI: {response['messages'][-1].content}\n")

# --- 4. Inspect persisted state ---
print("=== Inspecting Persisted State ===\n")

state = graph.get_state(config_thread1)
print(f"Thread 1 - Number of messages: {len(state.values['messages'])}")
print(f"Thread 1 - Checkpoint ID: {state.config['configurable']['checkpoint_id']}")

state2 = graph.get_state(config_thread2)
print(f"Thread 2 - Number of messages: {len(state2.values['messages'])}")
print(f"Thread 2 - Checkpoint ID: {state2.config['configurable']['checkpoint_id']}")
