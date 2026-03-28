from smolagents import CodeAgent, OpenAIModel, tool

from settings import settings

"""
-------------------------------------------------------
In this example, we explore smolagents advanced agent patterns:

- Custom instructions to shape agent behavior
- final_answer_checks for output validation
- Step-by-step execution with agent.step()
- Structured outputs with use_structured_outputs_internally
- Verbose control with verbosity_level

These advanced patterns give you fine-grained control over
agent behavior: instructions shape personality, final answer
checks validate outputs, and step-by-step execution lets you
control the agent loop manually.

For more details, visit:
https://huggingface.co/docs/smolagents/reference/agents
-------------------------------------------------------
"""

# --- 1. Create the model ---
model = OpenAIModel(
    model_id=settings.OPENAI_MODEL_NAME,
    api_key=settings.OPENAI_API_KEY.get_secret_value(),
)


# --- 2. Define tools ---
@tool
def analyze_sentiment(text: str) -> str:
    """Analyze the sentiment of a text.

    Args:
        text: The text to analyze.

    Returns:
        Sentiment analysis result with score.
    """
    # Simple keyword-based sentiment (simulated)
    positive = ["good", "great", "excellent", "amazing", "love", "happy", "wonderful"]
    negative = ["bad", "terrible", "awful", "hate", "angry", "sad", "horrible"]

    text_lower = text.lower()
    pos_count = sum(1 for w in positive if w in text_lower)
    neg_count = sum(1 for w in negative if w in text_lower)

    if pos_count > neg_count:
        return f"POSITIVE (score: {pos_count}/{pos_count + neg_count + 1})"
    elif neg_count > pos_count:
        return f"NEGATIVE (score: {neg_count}/{pos_count + neg_count + 1})"
    return "NEUTRAL (score: 0.5)"


# --- 3. Example 1: Custom Instructions ---
print("=== Advanced Patterns Demo ===\n")
print("--- Example 1: Custom Instructions ---")

agent_polite = CodeAgent(
    tools=[analyze_sentiment],
    model=model,
    max_steps=3,
    instructions=(
        "You are a polite British assistant. Always use formal language "
        "and address the user as 'dear user'. Keep responses brief (1-2 sentences)."
    ),
    verbosity_level=0,  # Suppress internal logs
)

result1 = agent_polite.run(
    "Analyze the sentiment of: 'This product is absolutely amazing and wonderful!' "
    "Summarize the result in your own words."
)
print(f"Polite agent: {result1}\n")


# --- 4. Example 2: Final Answer Checks ---
print("--- Example 2: Final Answer Checks ---")


def check_answer_not_empty(final_answer, memory, **kwargs) -> bool:
    """Validate that the agent's answer is not empty or too short."""
    if not final_answer or len(str(final_answer).strip()) < 5:
        print("  [Check FAILED] Answer too short!")
        return False
    print("  [Check PASSED] Answer is valid")
    return True


def check_answer_has_sentiment(final_answer, memory, **kwargs) -> bool:
    """Validate that the answer mentions a sentiment category."""
    answer_str = str(final_answer).lower()
    has_sentiment = any(
        word in answer_str for word in ["positive", "negative", "neutral"]
    )
    if not has_sentiment:
        print("  [Check FAILED] Answer missing sentiment label!")
        return False
    print("  [Check PASSED] Sentiment label found")
    return True


agent_checked = CodeAgent(
    tools=[analyze_sentiment],
    model=model,
    max_steps=4,
    final_answer_checks=[check_answer_not_empty, check_answer_has_sentiment],
    verbosity_level=0,
)

result2 = agent_checked.run(
    "Analyze the sentiment of: 'The weather is terrible and I feel awful.' "
    "Include the sentiment label (positive/negative/neutral) in your answer."
)
print(f"Checked agent: {result2}\n")


# --- 5. Example 3: Step-by-Step Execution ---
print("--- Example 3: Step-by-Step Execution ---")

agent_manual = CodeAgent(
    tools=[analyze_sentiment],
    model=model,
    max_steps=4,
    verbosity_level=0,
)

# Initialize a task but don't run it to completion
agent_manual.run(
    "Analyze sentiment of: 'Great product, love it!' Reply in one sentence.",
    stream=False,
)

# After run completes, inspect what happened step by step
print("Steps executed:")
for i, step in enumerate(agent_manual.memory.steps):
    step_type = type(step).__name__
    print(f"  Step {i}: {step_type}")
    if hasattr(step, "tool_calls") and step.tool_calls:
        for tc in step.tool_calls:
            print(f"    -> Tool: {tc.name}")

print(f"\nTotal steps taken: {len(agent_manual.memory.steps)}")
