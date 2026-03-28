from smolagents import CodeAgent, OpenAIModel

from settings import settings

"""
-------------------------------------------------------
In this example, we explore smolagents with the simplest
possible agent — a CodeAgent that answers a question.

- Creating a CodeAgent with an OpenAI model
- Running a single query and printing the response

smolagents is a lightweight framework by Hugging Face for
building agents that write and execute Python code to solve
tasks. This hello world shows the minimal setup.

For more details, visit:
https://huggingface.co/docs/smolagents/guided_tour
-------------------------------------------------------
"""

# --- 1. Create the model ---
model = OpenAIModel(
    model_id=settings.OPENAI_MODEL_NAME,
    api_key=settings.OPENAI_API_KEY.get_secret_value(),
)

# --- 2. Create a CodeAgent ---
agent = CodeAgent(
    tools=[],
    model=model,
    max_steps=2,
)

# --- 3. Run a simple query ---
print("=== Hello World: CodeAgent ===")
result = agent.run("Where does 'hello world' come from? Reply in one sentence.")
print(f"\nAgent response: {result}")
