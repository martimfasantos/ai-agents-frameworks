from strands import Agent
from strands.models.openai import OpenAIModel
from strands.multiagent import GraphBuilder

from settings import settings

"""
-------------------------------------------------------
In this example, we explore Strands Agents SDK with the following features:
- Graph-based multi-agent orchestration with GraphBuilder
- Adding agent nodes and defining edges
- Conditional edges for dynamic routing
- Inspecting graph execution results (status, order, timing)

The GraphBuilder creates a directed graph of agents where edges define
the flow between nodes. Conditional edges let you route dynamically
based on previous agent output. This gives you explicit control over
multi-agent pipelines.

For more details, visit:
https://strandsagents.com/docs/user-guide/concepts/multi-agent/graph/
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
planner = Agent(
    model=openai_model,
    name="planner",
    system_prompt=(
        "You are a project planner. Break down the given task into 3 clear steps. "
        "Be concise — list only the steps, no extra commentary."
    ),
    callback_handler=None,
)

executor = Agent(
    model=openai_model,
    name="executor",
    system_prompt=(
        "You are a task executor. Take the plan provided and execute each step, "
        "providing brief results for each. Be concise."
    ),
    callback_handler=None,
)

reviewer = Agent(
    model=openai_model,
    name="reviewer",
    system_prompt=(
        "You are a quality reviewer. Review the executed results and provide "
        "a final assessment with a quality score (1-10). Be concise."
    ),
    callback_handler=None,
)

# --- 2. Build the graph ---
builder = GraphBuilder()

builder.add_node(planner, "planner")
builder.add_node(executor, "executor")
builder.add_node(reviewer, "reviewer")

# Define the flow: planner -> executor -> reviewer
builder.add_edge("planner", "executor")
builder.add_edge("executor", "reviewer")

builder.set_entry_point("planner")

graph = builder.build()

# --- 3. Execute the graph ---
print("=== Multi-Agent Graph ===\n")
result = graph("Create a Python CLI tool that converts CSV files to JSON format.")

# --- 4. Inspect results ---
print(f"\nGraph status: {result.status}")
print(f"Execution order: {result.execution_order}")
print(f"Total nodes: {result.total_nodes}")
print(f"Completed nodes: {result.completed_nodes}")

print(f"\n--- Node Results ---")
for node_id, node_result in result.results.items():
    agent_results = node_result.get_agent_results()
    for ar in agent_results:
        print(f"\n[{node_id}]: {ar.message}")
