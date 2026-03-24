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
- Viewing checkpoint history with get_state_history()
- Replaying from a previous checkpoint
- Forking execution by modifying state at a past checkpoint
- Time travel debugging for agent workflows

Time travel lets you inspect every checkpoint in a thread's history,
replay execution from any point, or fork by injecting modified state.
This is invaluable for debugging: you can find where an agent went
wrong, rewind, and try a different path without re-running everything.

For more details, visit:
https://docs.langchain.com/oss/python/langgraph/use-time-travel
-----------------------------------------------------------------------
"""

llm = ChatOpenAI(model=settings.OPENAI_MODEL_NAME)


# --- 1. Build a simple chatbot with persistence ---
def chatbot(state: MessagesState):
    return {"messages": [llm.invoke(state["messages"])]}


builder = StateGraph(MessagesState)
builder.add_node("chatbot", chatbot)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

memory = InMemorySaver()
graph = builder.compile(checkpointer=memory)

# --- 2. Build up a conversation ---
print("=== Building Conversation History ===\n")

config = {"configurable": {"thread_id": "time-travel-demo"}}

exchanges = [
    "Hi! I'm building a web app with React.",
    "Should I use Redux or Zustand for state management?",
    "I decided to go with Zustand. What middleware should I use?",
]

for msg in exchanges:
    result = graph.invoke({"messages": [HumanMessage(content=msg)]}, config=config)
    ai_content = result["messages"][-1].content[:100] + "..."
    print(f"User: {msg}")
    print(f"AI: {ai_content}\n")

# --- 3. View checkpoint history ---
print("=== Checkpoint History ===\n")

checkpoints = list(graph.get_state_history(config))
print(f"Total checkpoints: {len(checkpoints)}\n")

for i, checkpoint in enumerate(checkpoints):
    num_msgs = len(checkpoint.values.get("messages", []))
    checkpoint_id = checkpoint.config["configurable"]["checkpoint_id"]
    print(f"  Checkpoint {i}: {num_msgs} messages (id: {checkpoint_id})")

# --- 4. Replay from an earlier checkpoint ---
print("\n=== Replay from Checkpoint (after first exchange) ===\n")

# Find the checkpoint after the first exchange (should have 2 messages: human + AI)
target_checkpoint = None
for cp in checkpoints:
    if len(cp.values.get("messages", [])) == 2:
        target_checkpoint = cp
        break

if target_checkpoint:
    replay_config = target_checkpoint.config
    print(
        f"Replaying from checkpoint with {len(target_checkpoint.values['messages'])} messages"
    )

    # Continue from that checkpoint with a different question
    result = graph.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Actually, I changed my mind. I want to use Vue.js instead."
                )
            ]
        },
        config=replay_config,
    )
    ai_content = result["messages"][-1].content[:150] + "..."
    print(f"User: Actually, I changed my mind. I want to use Vue.js instead.")
    print(f"AI: {ai_content}\n")

    # Show that the fork created a separate timeline
    print("=== Forked Timeline ===\n")
    forked_state = graph.get_state(replay_config)
    print(f"Messages in forked timeline: {len(forked_state.values['messages'])}")

    # Original timeline is unchanged
    original_state = graph.get_state(config)
    print(f"Messages in original timeline: {len(original_state.values['messages'])}")
