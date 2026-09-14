import os

from crewai import Agent, Task, Crew, LLMGuardrail

from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore CrewAI with the following features:
- LLMGuardrail for AI-powered output validation
- Natural language validation criteria (no code needed)
- Automatic retry with LLM-generated feedback
- Combining LLMGuardrail with function-based guardrails

LLMGuardrail (new in 1.14.x) uses an LLM to validate task outputs
against natural language criteria. Unlike function-based guardrails
that require code logic, LLMGuardrail lets you describe validation
rules in plain English and the LLM decides if the output passes.

For more details, visit:
https://docs.crewai.com/en/concepts/tasks#task-guardrails
-------------------------------------------------------
"""


# --- 1. Define an LLM-based guardrail ---
# This guardrail validates output using natural language criteria.
# No code logic needed — the LLM judges the output.
factual_guardrail = LLMGuardrail(
    description="""Validate that the output:
    1. Contains only factual, verifiable statements
    2. Does not include speculative or opinion-based claims
    3. Cites or references specific data points (numbers, dates, names)
    If the output contains unverifiable speculation, reject it.""",
    llm=settings.OPENAI_MODEL_NAME,
)

# --- 2. Another LLM guardrail for format validation ---
# Criteria are structural on purpose. An LLM judge is reliable at "is this a
# bullet list" and unreliable at "is this under 50 words" — a numeric limit
# here makes the guardrail reject valid output at random.
format_guardrail = LLMGuardrail(
    description="""Validate that the output:
    1. Is structured as a bullet list, not a single paragraph
    2. Each bullet is a single sentence
    3. Contains between 3 and 5 bullets
    Reject if the output is a single paragraph or has too many/few points.""",
    llm=settings.OPENAI_MODEL_NAME,
)


# --- 3. Define agents and tasks ---
researcher = Agent(
    role="Research Analyst",
    goal="Provide factual, well-structured research summaries",
    backstory="You are a meticulous researcher who values accuracy and clarity.",
    llm=settings.OPENAI_MODEL_NAME,
    verbose=False,
)

research_task = Task(
    description="Summarize the key facts about the Python programming language's history.",
    # Mirrors what format_guardrail checks — an agent that is never told the
    # rule it is graded on fails validation at random.
    expected_output=(
        "A factual summary as 3-5 bullet points, each a single short sentence."
    ),
    agent=researcher,
    guardrails=[factual_guardrail, format_guardrail],  # Both guardrails applied
)


def main():
    # --- Run the crew ---
    print("=== LLM Guardrail Demo ===\n")
    print("Guardrails applied:")
    print("  1. Factual accuracy (LLM checks for speculation)")
    print("  2. Format validation (LLM checks structure)\n")

    crew = Crew(
        agents=[researcher],
        tasks=[research_task],
        verbose=False,
    )

    result = crew.kickoff()

    print(f"Result:\n{result.raw}\n")
    print("=== LLMGuardrail vs Function Guardrails ===")
    print("Function guardrails: Code logic (len check, regex, etc.)")
    print("LLMGuardrail:        Natural language criteria, LLM judges output")
    print("Both can be combined on the same task for layered validation.")


if __name__ == "__main__":
    main()
