import os
import asyncio

from google.adk.agents import LlmAgent, BaseAgent, SequentialAgent
from google.adk.events import Event
from google.adk.tools import agent_tool
from google.adk.agents.invocation_context import InvocationContext
from typing import AsyncGenerator

from utils import call_agent_async
from settings import settings

os.environ["GOOGLE_API_KEY"] = settings.GOOGLE_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore Google's ADK workflow agents with the following features:
- 1. Multi-Agent Systems
- 2. Agent Hierarchy (Parent agent, Sub Agents)
- 3. Shared Session State
- 4. LLM-Driven Delegation (Agent Transfer)
- 5. Agent as Tool
- 6. Workflow Agents as Orchestrators - (more in 5_workflow_agents.py)

This example demonstrates all four key multi-agent concepts in a single system:
1. Agent Hierarchy (Agent-Subagent relationships)
2. Shared Session State (State passing between agents)
3. LLM-Driven Delegation (Dynamic agent transfer)
4. Agent as Tool (Agents wrapped as tools for other agents)

It also introduces the concept of "Workflow Agents" as orchestrators 
for complex tasks that involve multiple agents.
-------------------------------------------------------
"""

# --- 1. Define Custom Research Agent (will be used as a tool) ---
class CustomResearchAgent(BaseAgent):
    """Custom agent that performs research and updates session context"""
    name: str = "ResearchAgent"
    description: str = "Research and report on given topics"

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        # Get the research topic from the tool arguments or session state
        topic = ctx.session.state.get("research_topic", "AI technologies")
        
        # Simulate research work
        research_findings = (
            f"Research findings on {topic}:\n"
            "- Advanced language models\n"
            "- Multimodal AI capabilities\n" 
            "- Improved efficiency\n"
            "- Wide industry adoption\n"
            "- AI safety focus\n"
            "- Specialized agents"
        )
        
        # Update session state with research data
        ctx.session.state["research_data"] = research_findings
        
        # Provide feedback about what was stored
        print(f"[{self.name}] Updated session state with research data for topic: {topic}")
        
        # Yield a simple event indicating completion
        yield Event(author=self.name)

# Create instance of the custom research agent
research_agent = CustomResearchAgent()

# --- 2. Analysis Agent (Sub-agent in hierarchy) ---
analysis_agent = LlmAgent(
    name="AnalysisAgent",
    model=settings.GOOGLE_MODEL_NAME,
    instruction=(
        "You are a data analysis specialist."
        "Analyze the research data: {research_data}"
        "Provide insights, trends, and recommendations."
        "Store your analysis summary in the session state with key 'analysis_summary'."
    ),
    description="Analyzes research data and provides insights",
    output_key="analysis_summary"  # Stores output in shared state
)

# --- 3. Report Generation Agent (Sub-agent in hierarchy) ---
report_agent = LlmAgent(
    name="ReportGenerator",
    model=settings.GOOGLE_MODEL_NAME,
    instruction=(
        "You are a professional report writer."
        "Create a comprehensive report based on:"
        "- Research findings: {research_data}"
        "- Analysis insights: {analysis_summary}"
        "Format the report with clear sections and professional language."
    ),
    description="Generates final reports from research and analysis",
    output_key="final_report"
)

# --- 4. Create Agent Tool from Custom Research Agent ---
research_tool = agent_tool.AgentTool(agent=research_agent)

# --- 5. Coordinator Agent (Parent in hierarchy with delegation capabilities) ---
coordinator_agent = LlmAgent(
    name="ProjectCoordinator",
    model=settings.GOOGLE_MODEL_NAME,
    instruction=(
        "You are a project coordinator managing a research and analysis workflow."
        "Follow this sequence:"
        "1. First, use the ResearchAgent tool to gather information on the requested topic"
        "2. After receiving research results, coordinate with your sub-agents for analysis and reporting"
        "3. Ensure data flows properly between all agents"
        ""
        "Available tools: ResearchAgent"
        "Sub-agents available for delegation: AnalysisAgent, ReportGenerator"
        ""
        "Start by using the ResearchAgent tool, then proceed with the workflow."
    ),
    description="Coordinates multi-agent research and analysis projects",
    tools=[research_tool],  # Agent as Tool
    sub_agents=[analysis_agent, report_agent]  # Agent Hierarchy
)

# --- 6. Workflow Orchestration Agent ---
# A simple sequential pipeline that uses the coordinator agent as the main orchestrator
workflow_pipeline = SequentialAgent(
    name="ResearchWorkflow",
    sub_agents=[
        coordinator_agent,  # Parent agent that uses tools and manages sub-agents
        # Analysis and report agents are managed by coordinator through hierarchy
    ],
    description=(
        "Complete research workflow with hierarchy, state sharing, delegation, and tool usage"
    )
)

# --- 7. Run the Example ---
input = (
    "Please conduct a comprehensive research project on AI technologies, "
    "including analysis and final report generation."
)

# Set up initial state for the research topic
initial_state = {
    "research_topic": "AI technologies",
    "research_data": "",   # safe default prevents KeyError
    "analysis_summary": ""
}

print("Input: ", input)
asyncio.run(
    call_agent_async(workflow_pipeline, input, tool_calls=True, state=initial_state)
)


# -------------------------------------------
#       Workflow Agents as Orchestrators
# -------------------------------------------
# NOTE: (For more complex workflows, see the 5_workflow_agents.py example)
# https://google.github.io/adk-docs/agents/multi-agents/#12-workflow-agents-as-orchestrators

def sequential_agent_example():
    """Sequential Agent Example"""
    from google.adk.agents import SequentialAgent, LlmAgent

    step1 = LlmAgent(name="Step1_Fetch", output_key="data") # Saves output to state['data']
    step2 = LlmAgent(name="Step2_Process", instruction="Process data from {data}.")

    pipeline = SequentialAgent(name="MyPipeline", sub_agents=[step1, step2])
    # When pipeline runs, Step2 can access the state['data'] set by Step1.
    
def parallel_agent_example():
    """Parallel Agent Example"""
    from google.adk.agents import ParallelAgent, LlmAgent

    fetch_weather = LlmAgent(name="WeatherFetcher", output_key="weather")
    fetch_news = LlmAgent(name="NewsFetcher", output_key="news")

    gatherer = ParallelAgent(name="InfoGatherer", sub_agents=[fetch_weather, fetch_news])
    # When gatherer runs, WeatherFetcher and NewsFetcher run concurrently.
    # A subsequent agent could read state['weather'] and state['news'].
    
def loop_agent_example():
    """Loop Agent Example"""
    from google.adk.agents import LoopAgent, LlmAgent, BaseAgent
    from google.adk.events import Event, EventActions
    from google.adk.agents.invocation_context import InvocationContext
    from typing import AsyncGenerator

    class CheckCondition(BaseAgent): # Custom agent to check state
        async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
            status = ctx.session.state.get("status", "pending")
            is_done = (status == "completed")
            yield Event(author=self.name, actions=EventActions(escalate=is_done)) # Escalate if done

    process_step = LlmAgent(name="ProcessingStep") # Agent that might update state['status']

    poller = LoopAgent(
        name="StatusPoller",
        max_iterations=10,
        sub_agents=[process_step, CheckCondition(name="Checker")]
    )
    # When poller runs, it executes process_step then Checker repeatedly
    # until Checker escalates (state['status'] == 'completed') or 10 iterations pass.
