import os

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI

from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore LangChain with the following features:
- Agentic RAG (Retrieval-Augmented Generation)
- Using tools as retrieval sources for the agent
- Agent deciding when and how to retrieve information

Agentic RAG combines retrieval with agent-based reasoning. Instead
of always retrieving before answering, the agent decides when to
search for external knowledge. The only thing needed is a tool
that can fetch relevant content — the agent handles the rest.

For more details, visit:
https://docs.langchain.com/oss/python/langchain/retrieval
-------------------------------------------------------
"""

# --- 1. Simulate a knowledge base ---
KNOWLEDGE_BASE = {
    "langchain overview": (
        "LangChain is a framework for building applications powered by language models. "
        "It provides tools for prompt management, chains, agents, memory, and retrieval. "
        "The latest version (v1.0+) uses create_agent() as the primary agent factory."
    ),
    "langchain agents": (
        "LangChain agents use create_agent() to build a graph-based agent runtime. "
        "Agents can use tools, maintain memory via checkpointers, and support streaming. "
        "The agent loop alternates between model calls and tool execution."
    ),
    "langchain tools": (
        "Tools in LangChain are created with the @tool decorator. The function name, "
        "docstring, and type hints become the tool schema visible to the LLM. "
        "Tools can access runtime context via the ToolRuntime parameter."
    ),
    "langchain middleware": (
        "Middleware hooks into the agent lifecycle at strategic points. Available hooks "
        "include @before_model, @after_model, @dynamic_prompt, and @wrap_model_call. "
        "Built-in middleware includes SummarizationMiddleware and HumanInTheLoopMiddleware."
    ),
}


# --- 2. Create a retrieval tool ---
@tool
def search_knowledge_base(query: str) -> str:
    """Search the LangChain knowledge base for relevant documentation.

    Args:
        query: The search query to find relevant documentation.
    """
    query_lower = query.lower()
    results = []
    for topic, content in KNOWLEDGE_BASE.items():
        # Simple keyword matching (real RAG would use embeddings)
        if any(word in topic for word in query_lower.split()):
            results.append(f"[{topic}]: {content}")

    if results:
        return "\n\n".join(results)
    return "No relevant documentation found. Try different search terms."


# --- 3. Create the model ---
model = ChatOpenAI(model=settings.OPENAI_MODEL_NAME)

# --- 4. Create the RAG agent ---
agent = create_agent(
    model=model,
    tools=[search_knowledge_base],
    system_prompt=(
        "You are a LangChain documentation assistant. "
        "When asked about LangChain features, ALWAYS use the search_knowledge_base "
        "tool first to find relevant documentation, then answer based on what you find. "
        "Quote or reference the documentation when possible."
    ),
)

# --- 5. Ask questions that require retrieval ---
print("=== Question 1: What are LangChain agents? ===")
result = agent.invoke(
    {"messages": [{"role": "user", "content": "How do agents work in LangChain?"}]}
)
print(f"{result['messages'][-1].content}\n")

print("=== Question 2: What is middleware? ===")
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Tell me about LangChain middleware."}]}
)
print(f"{result['messages'][-1].content}\n")

# --- 6. Ask a question that might not need retrieval ---
print("=== Question 3: Simple greeting (no retrieval needed) ===")
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Hello! What can you help me with?"}]}
)
print(f"{result['messages'][-1].content}")
