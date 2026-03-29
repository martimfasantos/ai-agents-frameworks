import os
import operator
from typing import Annotated, TypedDict

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore LangChain with the following features:
- Multi-agent router pattern for dispatching to specialized agents
- Parallel fan-out to multiple agents using Send
- Result synthesis from multiple agent responses

In the router architecture, a routing step classifies the input
and directs it to specialized agents. This is useful when you have
distinct verticals (separate knowledge domains) that each require
their own agent. The router decomposes the query, agents run in
parallel, and results are synthesized into a coherent response.

For more details, visit:
https://docs.langchain.com/oss/python/langchain/multi-agent/router
-------------------------------------------------------
"""

# --- 1. Create the model ---
model = ChatOpenAI(
    model=settings.OPENAI_MODEL_NAME,
    temperature=0.1,
    max_tokens=1000,
    timeout=30,
)


# --- 2. Define domain tools and create specialized agents ---
@tool
def search_docs(query: str) -> str:
    """Search technical documentation."""
    docs = {
        "python": "Python is a versatile language used for web dev, data science, and AI.",
        "api": "REST APIs use HTTP methods: GET, POST, PUT, DELETE for CRUD operations.",
        "database": "PostgreSQL and MongoDB are popular choices for relational and NoSQL databases.",
    }
    for key, value in docs.items():
        if key in query.lower():
            return value
    return f"Documentation for: {query} — general technical reference available."


@tool
def search_hr(query: str) -> str:
    """Search HR policies and procedures."""
    policies = {
        "vacation": "Employees get 25 days of paid vacation per year. Unused days carry over (max 5).",
        "remote": "Remote work is allowed 3 days per week. Full remote requires manager approval.",
        "benefits": "Benefits include health insurance, 401k matching (5%), and gym membership.",
    }
    for key, value in policies.items():
        if key in query.lower():
            return value
    return f"HR policy for: {query} — contact HR@company.com for details."


@tool
def search_finance(query: str) -> str:
    """Search financial data and reports."""
    data = {
        "revenue": "Q4 2025 revenue: $12.5M (+15% YoY). Annual revenue: $45M.",
        "budget": "2026 engineering budget: $8M. Marketing: $3M. Operations: $2M.",
        "expenses": "Monthly cloud costs: $150K. Office lease: $80K. Payroll: $1.2M.",
    }
    for key, value in data.items():
        if key in query.lower():
            return value
    return f"Financial data for: {query} — check the finance dashboard."


tech_agent = create_agent(
    model=model,
    tools=[search_docs],
    name="tech_agent",
    system_prompt="You are a technical documentation expert. Search for technical info and give a concise answer in 1-2 sentences.",
)

hr_agent = create_agent(
    model=model,
    tools=[search_hr],
    name="hr_agent",
    system_prompt="You are an HR policy expert. Search for HR policies and give a concise answer in 1-2 sentences.",
)

finance_agent = create_agent(
    model=model,
    tools=[search_finance],
    name="finance_agent",
    system_prompt="You are a finance expert. Search for financial data and give a concise answer in 1-2 sentences.",
)

AGENT_MAP = {
    "tech": tech_agent,
    "hr": hr_agent,
    "finance": finance_agent,
}

# --- 3. Create the classification agent ---
classification_agent = create_agent(
    model=model,
    tools=[],
    system_prompt=(
        "You are a query classifier. Given a user query, determine which departments "
        "should handle it. Respond with ONLY a comma-separated list of departments from: "
        "tech, hr, finance. For example: 'tech' or 'tech, finance'. "
        "Do not include any other text."
    ),
)


# --- 4. Build the router graph ---
class RouterState(TypedDict):
    query: str
    agent_results: Annotated[list[str], operator.add]
    final_answer: str


class AgentTaskState(TypedDict):
    query: str
    agent: str


def route_query(state: RouterState) -> list[Send]:
    """Classify the query and fan out to relevant agents via Send."""
    result = classification_agent.invoke(
        {"messages": [{"role": "user", "content": state["query"]}]}
    )
    classification = result["messages"][-1].content.strip().lower()
    departments = [d.strip() for d in classification.split(",")]

    print(f"  [router] Query classified to: {departments}")

    sends = []
    for dept in departments:
        if dept in AGENT_MAP:
            sends.append(Send("run_agent", {"query": state["query"], "agent": dept}))

    # Fallback: if no valid departments, route to tech
    if not sends:
        sends.append(Send("run_agent", {"query": state["query"], "agent": "tech"}))

    return sends


def run_agent(state: AgentTaskState) -> dict:
    """Run a specialized agent and collect its response."""
    agent_name = state["agent"]
    agent = AGENT_MAP[agent_name]

    result = agent.invoke(
        {"messages": [{"role": "user", "content": state["query"]}]},
        config={"recursion_limit": 25},
    )
    response = result["messages"][-1].content
    print(f"  [{agent_name}] {response[:100]}...")
    return {"agent_results": [f"[{agent_name}]: {response}"]}


def synthesize(state: RouterState) -> dict:
    """Synthesize results from all agents into a final answer."""
    if not state.get("agent_results"):
        return {"final_answer": "No results from any agent."}

    combined = "\n\n".join(state["agent_results"])

    synthesis_result = model.invoke(
        [
            {
                "role": "system",
                "content": "Synthesize the following agent responses into a clear, concise answer. Keep it under 100 words.",
            },
            {"role": "user", "content": combined},
        ]
    )
    return {"final_answer": synthesis_result.content}


# --- 5. Assemble the graph ---
graph = StateGraph(RouterState)
graph.add_node("run_agent", run_agent)
graph.add_node("synthesize", synthesize)

graph.add_conditional_edges(START, route_query, ["run_agent"])
graph.add_edge("run_agent", "synthesize")
graph.add_edge("synthesize", END)

workflow = graph.compile()

# --- 6. Test single-domain routing ---
print("=== Single Domain: HR Query ===")
result = workflow.invoke({"query": "What is the vacation policy?", "agent_results": []})
print(f"Answer: {result['final_answer']}\n")

# --- 7. Test multi-domain routing ---
print("=== Multi-Domain: Tech + Finance Query ===")
result = workflow.invoke(
    {
        "query": "What is our cloud budget and what database should we use?",
        "agent_results": [],
    }
)
print(f"Answer: {result['final_answer']}")
