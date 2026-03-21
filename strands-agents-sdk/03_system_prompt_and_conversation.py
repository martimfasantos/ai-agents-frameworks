from strands import Agent
from strands.models.openai import OpenAIModel

from settings import settings

"""
-------------------------------------------------------
In this example, we explore Strands Agents SDK with the following features:
- System prompts to control agent behavior and persona
- Multi-turn conversation with automatic history
- Accessing conversation metrics

The agent maintains conversation history across invocations automatically,
so each follow-up message has full context of prior turns. System prompts
let you define the agent's personality, constraints, and response style.

For more details, visit:
https://strandsagents.com/docs/user-guide/concepts/agents/prompts/
-------------------------------------------------------
"""

# --- 1. Configure model and create an agent with a persona ---
openai_model = OpenAIModel(
    client_args={
        "api_key": settings.OPENAI_API_KEY.get_secret_value()
        if settings.OPENAI_API_KEY
        else ""
    },
    model_id=settings.OPENAI_MODEL_NAME,
)
# Default: Agent() uses Amazon Bedrock (requires AWS credentials)
agent = Agent(
    model=openai_model,
    system_prompt="""You are a friendly pirate captain named Captain Strandbeard.
You always respond in pirate speak and relate everything to sailing and the sea.
Keep your responses concise (2-3 sentences max).""",
)

# --- 2. First conversation turn ---
print("=== Conversation Turn 1 ===")
result1 = agent("What's the weather like today?")
print(f"Agent: {result1.message}\n")

# --- 3. Second turn (agent remembers context) ---
print("=== Conversation Turn 2 ===")
result2 = agent("What should I have for lunch?")
print(f"Agent: {result2.message}\n")

# --- 4. Third turn (multi-turn context) ---
print("=== Conversation Turn 3 ===")
result3 = agent("Tell me a joke about what we've been discussing.")
print(f"Agent: {result3.message}\n")

# --- 5. Show conversation metrics ---
print("=== Conversation Metrics ===")
print(f"Total messages in history: {len(agent.messages)}")
