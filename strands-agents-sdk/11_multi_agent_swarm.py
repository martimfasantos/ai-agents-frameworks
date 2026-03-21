from strands import Agent
from strands.models.openai import OpenAIModel
from strands.multiagent import Swarm

from settings import settings

"""
-------------------------------------------------------
In this example, we explore Strands Agents SDK with the following features:
- Swarm multi-agent coordination
- Creating named agents with specialized roles
- Autonomous agent handoffs within a swarm
- Inspecting swarm execution results (status, history, timing)

The Swarm class enables autonomous coordination between multiple agents.
Agents can hand off tasks to each other based on the conversation context.
The swarm manages the handoff protocol, max iterations, and execution tracking.

For more details, visit:
https://strandsagents.com/docs/user-guide/concepts/multi-agent/swarm/
-------------------------------------------------------
"""

# --- 1. Configure model and create specialized agents ---
openai_model = OpenAIModel(
    client_args={
        "api_key": settings.OPENAI_API_KEY.get_secret_value()
        if settings.OPENAI_API_KEY
        else ""
    },
    model_id=settings.OPENAI_MODEL_NAME,
)
# Default: Agent() uses Amazon Bedrock (requires AWS credentials)
researcher = Agent(
    model=openai_model,
    name="researcher",
    system_prompt=(
        "You are a research specialist. When given a topic, provide key facts "
        "and findings. If analysis is needed, hand off to the analyst agent."
    ),
    callback_handler=None,
)

analyst = Agent(
    model=openai_model,
    name="analyst",
    system_prompt=(
        "You are a data analyst. When given research findings, analyze patterns "
        "and draw conclusions. If a written summary is needed, hand off to the writer agent."
    ),
    callback_handler=None,
)

writer = Agent(
    model=openai_model,
    name="writer",
    system_prompt=(
        "You are a technical writer. Take analysis results and produce a clear, "
        "concise summary report. You are the final agent — do not hand off."
    ),
    callback_handler=None,
)

# --- 2. Create the swarm ---
swarm = Swarm(
    nodes=[researcher, analyst, writer],
    entry_point=researcher,
    max_handoffs=5,
    max_iterations=10,
)

# --- 3. Execute the swarm ---
print("=== Multi-Agent Swarm ===\n")
result = swarm(
    "Research the key benefits of renewable energy, analyze the trends, and write a brief summary."
)

# --- 4. Inspect results ---
print(f"\nSwarm status: {result.status}")
print(f"Execution count: {result.execution_count}")
print(f"Execution time: {result.execution_time}ms")
print(f"Node history: {result.node_history}")

print(f"\n--- Node Results ---")
for node_id, node_result in result.results.items():
    # node_result.result contains the AgentResult
    agent_results = node_result.get_agent_results()
    for ar in agent_results:
        print(f"\n[{node_id}]: {ar.message}")
