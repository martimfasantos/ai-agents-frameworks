import os

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI

from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore LangChain with the following features:
- Multi-agent subagents pattern
- Wrapping subagents as tools for a main (supervisor) agent
- Central orchestration with context isolation

In the subagents pattern, a main agent coordinates specialized
subagents by invoking them as tools. Each subagent runs in its
own context window — it receives a task and returns a result.
The main agent decides which subagent to invoke and synthesizes
the final response for the user.

For more details, visit:
https://docs.langchain.com/oss/python/langchain/multi-agent/subagents
-------------------------------------------------------
"""

# --- 1. Create the model ---
model = ChatOpenAI(
    model=settings.OPENAI_MODEL_NAME,
    temperature=0.1,
    max_tokens=1000,
    timeout=30,
)


# --- 2. Create specialized subagents ---


# Research subagent with its own tools
@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    results = {
        "python popularity": "Python is the #1 most popular programming language in 2025 according to TIOBE.",
        "rust safety": "Rust's ownership model prevents memory safety bugs at compile time.",
        "javascript frameworks": "React, Vue, and Angular remain the top JavaScript frameworks.",
    }
    for key, value in results.items():
        if any(word in query.lower() for word in key.split()):
            return value
    return f"Search results for: {query} — (general information available)"


research_agent = create_agent(
    model=model,
    tools=[search_web],
    system_prompt=(
        "You are a research specialist. Search for information using the search_web tool "
        "and provide a concise, factual summary. Call search_web ONCE with the best query, "
        "then immediately respond with your findings. Do NOT search multiple times. "
        "Your response will be read by a supervisor agent."
    ),
)

# Writer subagent (no tools, just writing skill)
writer_agent = create_agent(
    model=model,
    tools=[],
    system_prompt=(
        "You are a writing specialist. Take research findings and write "
        "a clear, engaging summary. Keep it under 100 words. Your response "
        "will be returned to a supervisor agent."
    ),
)


# --- 3. Wrap subagents as tools ---
@tool("research", description="Research a topic and return findings")
def call_research_agent(query: str) -> str:
    """Delegate a research task to the research specialist."""
    result = research_agent.invoke(
        {"messages": [{"role": "user", "content": query}]},
        config={"recursion_limit": 25},
    )
    return result["messages"][-1].content


@tool("write_summary", description="Write a polished summary from research notes")
def call_writer_agent(research_notes: str) -> str:
    """Delegate a writing task to the writing specialist."""
    result = writer_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": f"Write a summary based on: {research_notes}",
                }
            ]
        },
        config={"recursion_limit": 25},
    )
    return result["messages"][-1].content


# --- 4. Create the main (supervisor) agent ---
main_agent = create_agent(
    model=model,
    tools=[call_research_agent, call_writer_agent],
    system_prompt=(
        "You are a supervisor agent that coordinates specialists. "
        "When a user asks a question:\n"
        "1. Use the 'research' tool to gather information\n"
        "2. Use the 'write_summary' tool to create a polished response\n"
        "3. Present the final summary to the user\n"
        "Always use both tools in sequence."
    ),
)

# --- 5. Run the multi-agent system ---
print("=== Multi-Agent: Research + Write ===")
result = main_agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "Why is Python so popular for programming?"}
        ]
    },
    config={"recursion_limit": 50},
)
print(f"Final response:\n{result['messages'][-1].content}")
