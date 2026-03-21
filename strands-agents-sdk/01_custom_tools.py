from strands import Agent, tool
from strands.models.openai import OpenAIModel

from settings import settings

"""
-------------------------------------------------------
In this example, we explore Strands Agents SDK with the following features:
- Custom tools using the @tool decorator
- Docstring-driven tool schema generation
- Multi-tool agent that auto-selects the right tool

Tools are Python functions decorated with @tool. Strands uses docstrings
and type hints to generate the tool specification that the model sees,
so the agent can decide when and how to invoke each tool.

For more details, visit:
https://strandsagents.com/docs/user-guide/concepts/tools/custom-tools/
-------------------------------------------------------
"""

# --- 1. Define custom tools ---


@tool
def word_count(text: str) -> int:
    """Count the number of words in a given text.

    Args:
        text: The text to count words in
    """
    return len(text.split())


@tool
def reverse_string(text: str) -> str:
    """Reverse a given string.

    Args:
        text: The string to reverse
    """
    return text[::-1]


@tool
def letter_counter(word: str, letter: str) -> int:
    """Count occurrences of a specific letter in a word.

    Args:
        word: The input word to search in
        letter: The specific letter to count

    Returns:
        The number of occurrences of the letter in the word
    """
    if not isinstance(word, str) or not isinstance(letter, str):
        return 0
    if len(letter) != 1:
        raise ValueError("The 'letter' parameter must be a single character")
    return word.lower().count(letter.lower())


# --- 2. Configure model and create agent with custom tools ---
openai_model = OpenAIModel(
    client_args={
        "api_key": settings.OPENAI_API_KEY.get_secret_value()
        if settings.OPENAI_API_KEY
        else ""
    },
    model_id=settings.OPENAI_MODEL_NAME,
)
# Default: Agent(tools=[...]) uses Amazon Bedrock (requires AWS credentials)
agent = Agent(model=openai_model, tools=[word_count, reverse_string, letter_counter])

# --- 3. Run the agent (it auto-selects tools) ---
result = agent(
    "I have 3 requests:\n"
    "1. How many words are in the sentence 'The quick brown fox jumps over the lazy dog'?\n"
    "2. What is 'hello world' reversed?\n"
    "3. How many letter R's are in 'strawberry'?"
)

# --- 4. Print results ---
print("\n--- Agent Result ---")
print(f"Message: {result.message}")
