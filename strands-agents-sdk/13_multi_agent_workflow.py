from strands import Agent
from strands.models.openai import OpenAIModel
from strands_tools import workflow

from settings import settings

"""
-------------------------------------------------------
In this example, we explore Strands Agents SDK with the following features:
- Workflow tool for structured multi-agent task execution
- Task creation with dependencies and priorities
- Parallel execution of independent tasks
- Workflow status monitoring and lifecycle management

The workflow tool from strands-agents-tools provides a built-in way to
define, schedule, and execute multi-agent workflows. Tasks are defined
with dependencies and priorities, and independent tasks run in parallel
automatically. This is ideal for complex multi-step processes.

For more details, visit:
https://strandsagents.com/docs/user-guide/concepts/multi-agent/workflow/
-------------------------------------------------------
"""

# --- 1. Configure model and create an agent with workflow capability ---
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
    system_prompt="You are a workflow orchestrator. Use the workflow tool to manage multi-agent tasks.",
    tools=[workflow],
    callback_handler=None,
)

# --- 2. Show workflow patterns ---
print("=== Multi-Agent Workflow ===\n")

print("--- Workflow Tool Usage ---\n")
print("The workflow tool supports these actions:\n")
print("  create  - Define a new workflow with tasks and dependencies")
print("  start   - Execute the workflow (parallel where possible)")
print("  status  - Check workflow progress and task states")
print("  pause   - Pause a running workflow")
print("  resume  - Resume a paused workflow")
print("  list    - List all workflows")
print("  delete  - Remove a workflow\n")

# --- 3. Example workflow definition ---
print("--- Example: Data Analysis Pipeline ---\n")

workflow_definition = """
# Create a workflow via the agent:
agent.tool.workflow(
    action="create",
    workflow_id="data_analysis",
    tasks=[
        {
            "task_id": "data_extraction",
            "description": "Extract key financial data from the quarterly report",
            "system_prompt": "You extract and structure financial data from reports.",
            "priority": 5,
        },
        {
            "task_id": "market_research",
            "description": "Research current market conditions and competitors",
            "system_prompt": "You research market trends and competitive landscape.",
            "priority": 4,
        },
        {
            "task_id": "trend_analysis",
            "description": "Analyze trends in the data compared to previous quarters",
            "dependencies": ["data_extraction", "market_research"],
            "system_prompt": "You identify trends in financial time series.",
            "priority": 3,
        },
        {
            "task_id": "report_generation",
            "description": "Generate a comprehensive analysis report",
            "dependencies": ["trend_analysis"],
            "system_prompt": "You create clear financial analysis reports.",
            "priority": 2,
        },
    ],
)

# Execute workflow (data_extraction and market_research run in parallel)
agent.tool.workflow(action="start", workflow_id="data_analysis")

# Check results
status = agent.tool.workflow(action="status", workflow_id="data_analysis")
print(status["content"])
"""
print(workflow_definition)

# --- 4. Show sequential workflow pattern ---
print("--- Alternative: Manual Sequential Workflow ---\n")

sequential_code = """
from strands import Agent

# Create specialized agents
researcher = Agent(
    system_prompt="You are a research specialist. Find key information.",
    callback_handler=None,
)
analyst = Agent(
    system_prompt="You analyze research data and extract insights.",
    callback_handler=None,
)
writer = Agent(
    system_prompt="You create polished reports based on analysis.",
)

# Sequential pipeline — each agent's output feeds the next
def process_workflow(topic):
    research = researcher(f"Research the latest developments in {topic}")
    analysis = analyst(f"Analyze these findings: {research}")
    report = writer(f"Create a report from this analysis: {analysis}")
    return report
"""
print(sequential_code)

# --- 5. Summary ---
print("--- Summary ---")
print("Workflow capabilities in Strands:")
print("  - Built-in workflow tool for task-based orchestration")
print("  - Automatic dependency resolution and parallel execution")
print("  - Pause/resume for long-running processes")
print("  - Status monitoring with progress tracking")
print("  - Manual sequential pipelines for simpler use cases")
