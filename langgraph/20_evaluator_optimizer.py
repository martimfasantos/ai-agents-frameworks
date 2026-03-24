from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from settings import settings

load_dotenv()

"""
-----------------------------------------------------------------------
In this example, we explore LangGraph with the following features:
- Evaluator-optimizer loop: generate then evaluate in a cycle
- Structured output for evaluation scores and feedback
- Conditional looping with a quality threshold
- Self-improving content through iterative refinement

The evaluator-optimizer pattern uses two LLM roles: a generator that
produces content, and an evaluator that scores it and provides feedback.
If the score is below a threshold, the generator tries again using the
feedback. This creates a self-improving loop that converges on higher
quality output, useful for writing, code generation, and planning.

For more details, visit:
https://docs.langchain.com/oss/python/langgraph/workflows-agents
-----------------------------------------------------------------------
"""

llm = ChatOpenAI(model=settings.OPENAI_MODEL_NAME)

QUALITY_THRESHOLD = 8
MAX_ITERATIONS = 3


# --- 1. Define evaluation schema ---
class Evaluation(BaseModel):
    """Structured evaluation of generated content."""

    score: int = Field(ge=1, le=10, description="Quality score from 1-10")
    strengths: list[str] = Field(description="What was done well")
    improvements: list[str] = Field(
        description="Specific actionable improvements needed"
    )
    meets_threshold: bool = Field(
        description="Whether the content meets quality standards"
    )


# --- 2. Define state ---
class OptimizationState(TypedDict):
    task: str
    current_draft: str
    evaluation_history: list[str]
    iteration: int
    final_output: str
    final_score: int


# --- 3. Generator node ---
def generator_node(state: OptimizationState) -> dict:
    """Generate or improve content based on feedback."""
    iteration = state["iteration"]

    if iteration == 0:
        # First attempt — generate from scratch
        prompt = f"Write a concise, engaging product description (3-4 sentences) for: {state['task']}"
    else:
        # Subsequent attempts — use feedback to improve
        last_feedback = state["evaluation_history"][-1]
        prompt = (
            f"Improve this product description based on the feedback below.\n\n"
            f"Current draft:\n{state['current_draft']}\n\n"
            f"Feedback:\n{last_feedback}\n\n"
            f"Write an improved version (3-4 sentences)."
        )

    response = llm.invoke(
        [
            SystemMessage(
                content="You are a skilled copywriter. Write compelling, clear product descriptions."
            ),
            HumanMessage(content=prompt),
        ]
    )

    print(f"\n[Iteration {iteration + 1}] Generator output:")
    print(f"  {response.content[:150]}...")

    return {
        "current_draft": response.content,
        "iteration": iteration + 1,
    }


# --- 4. Evaluator node ---
evaluator_llm = llm.with_structured_output(Evaluation)


def evaluator_node(state: OptimizationState) -> dict:
    """Evaluate the current draft and provide structured feedback."""
    evaluation = evaluator_llm.invoke(
        [
            SystemMessage(
                content=(
                    f"You are a critical content evaluator. Score the content 1-10 based on: "
                    f"clarity, engagement, persuasiveness, and conciseness. "
                    f"Set meets_threshold=true only if score >= {QUALITY_THRESHOLD}."
                )
            ),
            HumanMessage(
                content=f"Task: {state['task']}\n\nContent to evaluate:\n{state['current_draft']}"
            ),
        ]
    )

    feedback = (
        f"Score: {evaluation.score}/10 | "
        f"Strengths: {', '.join(evaluation.strengths)} | "
        f"Improvements: {', '.join(evaluation.improvements)}"
    )

    print(
        f"[Iteration {state['iteration']}] Evaluator: score={evaluation.score}/10, meets_threshold={evaluation.meets_threshold}"
    )

    return {
        "evaluation_history": state.get("evaluation_history", []) + [feedback],
        "final_score": evaluation.score,
        "final_output": state["current_draft"] if evaluation.meets_threshold else "",
    }


# --- 5. Routing: loop or finish ---
def should_continue(state: OptimizationState) -> str:
    """Continue iterating if quality threshold not met and under max iterations."""
    if state.get("final_output"):
        return "done"
    if state["iteration"] >= MAX_ITERATIONS:
        return "max_reached"
    return "improve"


# --- 6. Final node ---
def finalize_node(state: OptimizationState) -> dict:
    """Set the final output (either quality met or max iterations reached)."""
    if not state.get("final_output"):
        return {"final_output": state["current_draft"]}  # Best effort
    return {}


# --- 7. Build the graph ---
builder = StateGraph(OptimizationState)

builder.add_node("generator", generator_node)
builder.add_node("evaluator", evaluator_node)
builder.add_node("finalize", finalize_node)

builder.add_edge(START, "generator")
builder.add_edge("generator", "evaluator")
builder.add_conditional_edges(
    "evaluator",
    should_continue,
    {
        "improve": "generator",  # Loop back for another iteration
        "done": "finalize",  # Quality threshold met
        "max_reached": "finalize",  # Max iterations reached
    },
)
builder.add_edge("finalize", END)

graph = builder.compile()

# --- 8. Run the optimization loop ---
print("=== Evaluator-Optimizer Loop ===")

result = graph.invoke(
    {
        "task": "A smart home device that uses AI to learn your daily routines and automatically adjusts lighting, temperature, and music throughout the day.",
        "current_draft": "",
        "evaluation_history": [],
        "iteration": 0,
        "final_output": "",
        "final_score": 0,
    }
)

print(f"\n--- Final Result ---")
print(f"Iterations: {result['iteration']}")
print(f"Final Score: {result['final_score']}/10")
print(f"Output:\n{result['final_output']}")
