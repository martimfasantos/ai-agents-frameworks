from dotenv import load_dotenv

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.utils.pprint import pprint_run_response

from settings import settings

load_dotenv()

"""
-------------------------------------------------------
In this example, we explore Agno with the following features:
- Creating a basic agent with instructions
- Running a synchronous agent invocation
- Printing formatted output with pprint_run_response

This is the simplest possible Agno agent. We configure it
with an OpenAI model, give it a personality through
instructions, and run a single query. The RunOutput
object contains the agent's response.

For more details, visit:
https://docs.agno.com/agents/introduction
-------------------------------------------------------
"""

# --- 1. Create the agent ---
agent = Agent(
    model=OpenAIChat(id=settings.OPENAI_MODEL_NAME),
    instructions="You are a helpful assistant. Be concise, reply with one sentence.",
    markdown=True,
)

# --- 2. Run the agent ---
run_output = agent.run("Where does the phrase 'hello world' come from?")

# --- 3. Print the result ---
pprint_run_response(run_output)
