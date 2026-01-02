import os
import asyncio

from google.adk.tools import agent_tool
from google.adk.agents import Agent
from google.adk.tools import google_search
from google.adk.code_executors import BuiltInCodeExecutor

from utils import call_agent_async
from settings import settings

os.environ["GOOGLE_API_KEY"] = settings.GOOGLE_API_KEY.get_secret_value()


"""
-------------------------------------------------------
In this example, we explore Google's ADK agents with the following features:
- Tool usage
- Agent as a Tool
- Code Execution
- Orchestrator Agent

This example shows how to create two agents with built-in tools:
1. A search agent that uses the Google Search tool.
2. A coding agent that uses the Built-In Code Executor tool.

And an orchestrator agent that uses both agents as tools to answer a question.

> NOTE: Agent as Tools vs. Sub-Agents
- Agent as a Tool → Like calling a function: "Do this one thing and return the result."
- Sub-Agent → Like spawning a helper: "Handle this whole process yourself and I'm out."
-------------------------------------------------------
"""

# 1. Create the agents that will be used as tools
search_agent = Agent(
    name='search_agent',
    model=settings.GOOGLE_MODEL_NAME,
    instruction=(
        "You're a specialist in Google Search"
    ),
    tools=[google_search], # built-in tool for Google Search
)

coding_agent = Agent(
    name='coding_agent',
    model=settings.GOOGLE_MODEL_NAME,
    instruction=(
        "You're a specialist in Code Execution"
    ),
    code_executor=BuiltInCodeExecutor(), # built-in tool for code execution
)

# 2. Create the orchestrator agent that uses the above agents as tools
orchestrator = Agent(
    name="orchestrator",
    model=settings.GOOGLE_MODEL_NAME,
    description="Orchestrator Agent",
    instruction=(
        "You are an orchestrator agent that can use other agents as tools."
    ),
    tools=[
        agent_tool.AgentTool(agent=search_agent), 
        agent_tool.AgentTool(agent=coding_agent)
    ],
)

# 3. Run the orchestrator agent
input = "Search for one python script to fetch the price of Bitcoin in USD and execute it."
print("Input: ", input)
asyncio.run(
    call_agent_async(orchestrator, input, tool_calls=True, tool_call_results=True)
)
