import operator
from typing import Annotated

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

from settings import settings

load_dotenv()

"""
-----------------------------------------------------------------------
In this example, we explore LangGraph with the following features:
- Separate input and output schemas for a graph
- Filtering what data goes in and what comes out
- Keeping internal state private from callers

Input/output schemas let you define clean interfaces for your graph.
The internal state can carry extra fields (intermediate results,
scratch data) that callers never see. The input schema validates
what goes in, and the output schema filters what comes out.

For more details, visit:
https://docs.langchain.com/oss/python/langgraph/graph-api
-----------------------------------------------------------------------
"""


# --- 1. Define internal state, input schema, and output schema ---
class InternalState(TypedDict):
    """Full internal state — includes fields callers shouldn't see."""

    question: str
    raw_research: Annotated[list[str], operator.add]
    confidence: float
    answer: str


class InputSchema(TypedDict):
    """What the caller provides — just the question."""

    question: str


class OutputSchema(TypedDict):
    """What the caller receives — just the answer and confidence."""

    answer: str
    confidence: float


# --- 2. Define nodes that use internal state ---
def research_node(state: InternalState) -> dict:
    """Gather raw research (internal detail, not exposed to caller)."""
    question = state["question"]
    return {
        "raw_research": [
            f"Source A: Found relevant info about '{question}'",
            f"Source B: Additional context for '{question}'",
            f"Source C: Expert opinion on '{question}'",
        ],
    }


def analyze_node(state: InternalState) -> dict:
    """Analyze research and produce a confidence score."""
    num_sources = len(state["raw_research"])
    confidence = min(num_sources / 3.0, 1.0)  # More sources = higher confidence
    return {"confidence": confidence}


def answer_node(state: InternalState) -> dict:
    """Synthesize research into a final answer."""
    sources = state["raw_research"]
    answer = f"Based on {len(sources)} sources: The answer to '{state['question']}' involves multiple perspectives from our research."
    return {"answer": answer}


# --- 3. Build graph with input/output schemas ---
builder = StateGraph(
    InternalState, input_schema=InputSchema, output_schema=OutputSchema
)

builder.add_node("research", research_node)
builder.add_node("analyze", analyze_node)
builder.add_node("answer", answer_node)

builder.add_edge(START, "research")
builder.add_edge("research", "analyze")
builder.add_edge("analyze", "answer")
builder.add_edge("answer", END)

graph = builder.compile()

# --- 4. Invoke — caller only provides InputSchema fields ---
print("=== Input/Output Schemas ===\n")

result = graph.invoke({"question": "What is quantum computing?"})

# Result only contains OutputSchema fields (answer, confidence)
print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']:.0%}")

# Internal fields (raw_research) are NOT in the output
print(f"\nKeys in result: {list(result.keys())}")
print("Note: 'raw_research' is internal — not exposed to the caller")
