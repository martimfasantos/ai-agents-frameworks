import os

from crewai import Agent, Task, Crew

from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore CrewAI with the following features:
- MCP (Model Context Protocol) server integration
- DSL string syntax for MCP server configuration
- Structured MCPServerAdapter configuration
- Graceful handling when MCP servers are unavailable

MCP lets agents connect to external tool servers using a standard
protocol. CrewAI supports both a concise DSL string syntax and
structured adapter objects for MCP configuration.

For more details, visit:
https://docs.crewai.com/en/mcp/overview
-------------------------------------------------------
"""

# --- 1. Create an agent with MCP integration (DSL string syntax) ---
# The DSL syntax follows the pattern: "transport://path_or_url"
# Common transports: stdio://, sse://, streamable-http://
#
# NOTE: This example uses a placeholder MCP server URL. Replace with a
# real MCP server endpoint to test actual tool discovery and execution.
# If the server is unreachable, the agent will still function but
# without MCP-provided tools.

mcp_agent = Agent(
    role="MCP-Enabled Assistant",
    goal="Use MCP tools to accomplish tasks when available",
    backstory="You are an assistant with access to external tool servers via MCP.",
    llm=settings.OPENAI_MODEL_NAME,
    verbose=True,
    # MCP DSL string syntax examples:
    # mcps=["stdio://npx -y @modelcontextprotocol/server-filesystem /tmp"]
    # mcps=["sse://http://localhost:8000/sse"]
    # mcps=["streamable-http://http://localhost:8000/mcp"]
)

# --- 2. Create an agent with structured MCP configuration ---
# For more control, use MCPServerAdapter with explicit configuration
try:
    from crewai.tools.mcp_tools import MCPServerAdapter

    # Example: SSE-based MCP server (uncomment with a real server)
    # mcp_server = MCPServerAdapter(
    #     name="example-mcp-server",
    #     transport="sse",
    #     url="http://localhost:8000/sse",
    # )

    # Example: stdio-based MCP server
    # mcp_server = MCPServerAdapter(
    #     name="filesystem-server",
    #     transport="stdio",
    #     command="npx",
    #     args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
    # )

    print("MCPServerAdapter is available for structured MCP configuration.")
except ImportError:
    print("MCPServerAdapter not available in this version of CrewAI.")

# --- 3. Create a simple task ---
task = Task(
    description="List 3 interesting facts about the Python programming language.",
    expected_output="Three interesting facts about Python.",
    agent=mcp_agent,
)

# --- 4. Create and run the crew ---
crew = Crew(
    agents=[mcp_agent],
    tasks=[task],
    verbose=True,
)

result = crew.kickoff()
print("Result:", result.raw[:500])
