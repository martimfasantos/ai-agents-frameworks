from strands import Agent, tool
from strands.models.openai import OpenAIModel

from settings import settings

"""
-------------------------------------------------------
In this example, we explore Strands Agents SDK with the following features:
- Agent-to-Agent (A2A) protocol support
- Creating an A2A server from a Strands agent
- Consuming remote agents with A2AAgent client
- Async invocation and streaming patterns

The A2A protocol is an open standard for AI agents to discover, communicate,
and collaborate across platforms. Strands provides both server (expose your
agent) and client (consume remote agents) support out of the box.
Install A2A support with: pip install 'strands-agents[a2a]'

For more details, visit:
https://strandsagents.com/docs/user-guide/concepts/multi-agent/agent-to-agent/
-------------------------------------------------------
"""


# --- 1. Define a tool for the server agent ---


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression.

    Args:
        expression: A math expression to evaluate (e.g., '2 + 3 * 4')
    """
    try:
        result = eval(expression)  # noqa: S307 — demo only
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error: {e}"


# --- 2. Create the agent that would be served ---
print("=== Agent-to-Agent (A2A) Protocol ===\n")

openai_model = OpenAIModel(
    client_args={
        "api_key": settings.OPENAI_API_KEY.get_secret_value()
        if settings.OPENAI_API_KEY
        else ""
    },
    model_id=settings.OPENAI_MODEL_NAME,
)
# Default: Agent() uses Amazon Bedrock (requires AWS credentials)
server_agent = Agent(
    model=openai_model,
    name="Calculator Agent",
    description="A calculator agent that can perform basic arithmetic operations.",
    tools=[calculator],
    callback_handler=None,
)

# --- 3. Show A2A server setup ---
print("--- A2A Server Setup ---\n")

server_code = """
from strands import Agent
from strands.multiagent.a2a import A2AServer

agent = Agent(
    name="Calculator Agent",
    description="A calculator agent for arithmetic.",
    tools=[calculator],
    callback_handler=None,
)

# Create and start the A2A server
a2a_server = A2AServer(
    agent=agent,
    host="127.0.0.1",
    port=9000,
)
a2a_server.serve()
# Agent card served at http://127.0.0.1:9000/.well-known/agent-card.json
"""
print(server_code)

# --- 4. Show A2A client usage ---
print("--- A2A Client Usage ---\n")

client_code = """
from strands.agent.a2a_agent import A2AAgent

# Basic synchronous invocation
a2a_agent = A2AAgent(endpoint="http://localhost:9000")
result = a2a_agent("What is 10 ^ 6?")
print(result.message)

# Async invocation
import asyncio

async def main():
    a2a_agent = A2AAgent(endpoint="http://localhost:9000")
    result = await a2a_agent.invoke_async("Calculate 42 * 17")
    print(result.message)

asyncio.run(main())

# Streaming responses
async def stream():
    a2a_agent = A2AAgent(endpoint="http://localhost:9000")
    async for event in a2a_agent.stream_async("Explain 2^10"):
        if "data" in event:
            print(event["data"], end="", flush=True)

asyncio.run(stream())

# Fetch agent card (metadata)
async def get_card():
    a2a_agent = A2AAgent(endpoint="http://localhost:9000")
    card = await a2a_agent.get_agent_card()
    print(f"Agent: {card.name}, Skills: {card.skills}")

asyncio.run(get_card())
"""
print(client_code)

# --- 5. Show A2A in multi-agent patterns ---
print("--- A2A in Multi-Agent Patterns ---\n")

multiagent_code = '''
# Wrap A2AAgent as a tool in an orchestrator
from strands import Agent, tool
from strands.agent.a2a_agent import A2AAgent

calc_agent = A2AAgent(endpoint="http://calculator-service:9000", name="calculator")

@tool
def calculate(expression: str) -> str:
    """Perform a calculation using the remote calculator agent."""
    result = calc_agent(expression)
    return str(result.message["content"][0]["text"])

orchestrator = Agent(
    system_prompt="Use the calculate tool for math questions.",
    tools=[calculate],
)

# A2AAgent also works as a node in Graph workflows
from strands.multiagent import GraphBuilder

builder = GraphBuilder()
builder.add_node(calc_agent, "calculator")
builder.set_entry_point("calculator")
graph = builder.build()
result = graph("What is 100 / 7?")
'''
print(multiagent_code)

# --- 6. Summary ---
print("--- Summary ---")
print("A2A protocol capabilities in Strands:")
print("  - A2AServer: Expose any Strands agent as an A2A-compatible server")
print("  - A2AAgent: Consume remote A2A agents with a familiar Agent interface")
print("  - Works in Graph workflows and as tools in orchestrator agents")
print("  - Supports sync, async, and streaming invocation patterns")
print("  - Install with: pip install 'strands-agents[a2a]'")
