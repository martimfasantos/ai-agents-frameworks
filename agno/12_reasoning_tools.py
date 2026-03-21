from dotenv import load_dotenv

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.reasoning import ReasoningTools
from agno.utils.pprint import pprint_run_response

from settings import settings

load_dotenv()

"""
-------------------------------------------------------
In this example, we explore Agno with the following features:
- ReasoningTools for step-by-step thinking
- think and analyze tool capabilities
- Enhanced problem-solving through explicit reasoning steps

ReasoningTools give the agent dedicated tools to "think out
loud" before answering. The agent can call think() to work
through a problem step-by-step and analyze() to break down
complex questions. This improves accuracy on tasks that
require multi-step reasoning like math, logic, or planning.

For more details, visit:
https://docs.agno.com/tools/reasoning-tools
-------------------------------------------------------
"""

# --- 1. Create the agent with reasoning tools ---
agent = Agent(
    model=OpenAIChat(id=settings.OPENAI_MODEL_NAME),
    tools=[ReasoningTools()],
    instructions=[
        "You are a careful problem solver.",
        "Always use the think tool to reason through problems step by step.",
        "Use the analyze tool for breaking down complex questions.",
        "Show your work clearly.",
    ],
    markdown=True,
)

# --- 2. Test with a math/logic problem ---
print("=== Problem 1: Logic puzzle ===\n")
run_output = agent.run(
    "A farmer has 15 sheep. All but 8 die. How many sheep are left alive?"
)
pprint_run_response(run_output)

# --- 3. Test with a planning problem ---
print("\n=== Problem 2: Planning ===\n")
run_output = agent.run(
    "I have a meeting at 2pm, a gym session that takes 1 hour, a 30-minute commute, "
    "and I need to eat lunch which takes 45 minutes. It's currently 11am. "
    "Plan my schedule so I make it to the meeting on time."
)
pprint_run_response(run_output)
