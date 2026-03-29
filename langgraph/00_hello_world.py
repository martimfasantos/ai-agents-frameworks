from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END

from settings import settings

load_dotenv()

"""
-----------------------------------------------------------------------
In this example, we explore LangGraph with the following features:
- Building a minimal graph with a single chatbot node
- Using MessagesState for automatic message list management
- Compiling and invoking a graph

LangGraph models AI applications as graphs: nodes are functions that
process state, and edges define the flow between them. This hello world
example shows the simplest possible graph — a single LLM call wrapped
in a node.

For more details, visit:
https://docs.langchain.com/oss/python/langgraph/quickstart
-----------------------------------------------------------------------
"""

# --- 1. Initialize the LLM ---
llm = ChatOpenAI(model=settings.OPENAI_MODEL_NAME)


# --- 2. Define the chatbot node ---
def chatbot(state: MessagesState):
    """Call the LLM with the current message history."""
    return {"messages": [llm.invoke(state["messages"])]}


# --- 3. Build the graph ---
graph_builder = StateGraph(MessagesState)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()

# --- 4. Invoke the graph ---
print("=== LangGraph Hello World ===")
result = graph.invoke({"messages": [("user", "Where does 'hello world' come from?")]})

# The last message is the AI response
ai_message = result["messages"][-1]
print(f"\nUser: Where does 'hello world' come from?")
print(f"AI: {ai_message.content}")
