import operator
from typing import Annotated

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from settings import settings

load_dotenv()

"""
-----------------------------------------------------------------------
In this example, we explore LangGraph with the following features:
- Orchestrator-worker pattern with dynamic task delegation
- LLM-powered planning with structured output
- Send API for dispatching subtasks to workers
- Result aggregation and synthesis

The orchestrator-worker pattern uses an LLM to break a complex request
into subtasks, then dispatches each subtask to a worker using the Send
API. Workers process independently and their results are aggregated by
a synthesizer. This enables dynamic, LLM-driven task decomposition
where the number and nature of subtasks aren't known in advance.

For more details, visit:
https://docs.langchain.com/oss/python/langgraph/workflows-agents
-----------------------------------------------------------------------
"""

llm = ChatOpenAI(model=settings.OPENAI_MODEL_NAME)


# --- 1. Define structured output for task planning ---
class Subtask(BaseModel):
    """A single subtask to delegate to a worker."""

    title: str = Field(description="Brief title of the subtask")
    description: str = Field(
        description="Detailed description of what the worker should do"
    )


class TaskPlan(BaseModel):
    """A plan breaking down a complex task into subtasks."""

    subtasks: list[Subtask] = Field(description="List of subtasks to delegate")


# --- 2. Define state schemas ---
class OrchestratorState(TypedDict):
    request: str
    subtask_results: Annotated[list[str], operator.add]
    final_output: str


class WorkerState(TypedDict):
    title: str
    description: str
    subtask_results: Annotated[list[str], operator.add]


# --- 3. Orchestrator: plan and fan-out ---
planner_llm = llm.with_structured_output(TaskPlan)


def plan_and_delegate(state: OrchestratorState) -> list[Send]:
    """Use the LLM to break the request into subtasks and dispatch workers."""
    plan = planner_llm.invoke(
        [
            SystemMessage(
                content="Break this request into 2-4 independent subtasks that can be worked on in parallel. Each subtask should be specific and actionable."
            ),
            HumanMessage(content=state["request"]),
        ]
    )

    print(f"Orchestrator created {len(plan.subtasks)} subtasks:")
    for i, task in enumerate(plan.subtasks):
        print(f"  {i + 1}. {task.title}")
    print()

    return [
        Send(
            "worker",
            {
                "title": task.title,
                "description": task.description,
                "subtask_results": [],
            },
        )
        for task in plan.subtasks
    ]


# --- 4. Worker: execute subtask ---
def worker_node(state: WorkerState) -> dict:
    """Worker processes a single subtask."""
    response = llm.invoke(
        [
            SystemMessage(
                content="You are a focused worker. Complete the assigned subtask concisely in 2-3 sentences. Be specific and actionable."
            ),
            HumanMessage(
                content=f"Task: {state['title']}\n\nDetails: {state['description']}"
            ),
        ]
    )
    result = f"**{state['title']}**: {response.content}"
    print(f"  Worker completed: {state['title']}")
    return {"subtask_results": [result]}


# --- 5. Synthesizer: combine results ---
def synthesize_node(state: OrchestratorState) -> dict:
    """Combine all worker results into a cohesive final output."""
    results_text = "\n\n".join(state["subtask_results"])
    response = llm.invoke(
        [
            SystemMessage(
                content="You are a senior editor. Synthesize the following worker outputs into a cohesive, well-structured response. Keep it concise but comprehensive."
            ),
            HumanMessage(
                content=f"Original request: {state['request']}\n\nWorker outputs:\n\n{results_text}"
            ),
        ]
    )
    return {"final_output": response.content}


# --- 6. Build the graph ---
builder = StateGraph(OrchestratorState)

builder.add_node("worker", worker_node)
builder.add_node("synthesize", synthesize_node)

builder.add_conditional_edges(START, plan_and_delegate, ["worker"])
builder.add_edge("worker", "synthesize")
builder.add_edge("synthesize", END)

graph = builder.compile()

# --- 7. Run ---
print("=== Orchestrator-Worker Pattern ===\n")

result = graph.invoke(
    {
        "request": "Create a comprehensive guide for launching a new SaaS product, covering technical architecture, marketing strategy, and pricing model.",
        "subtask_results": [],
        "final_output": "",
    }
)

print(f"\n--- Final Output ---\n")
print(result["final_output"])
