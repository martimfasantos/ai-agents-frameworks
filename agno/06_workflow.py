from dotenv import load_dotenv

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools import tool
from agno.workflow import Workflow, Step, Parallel
from agno.utils.pprint import pprint_run_response

from settings import settings

load_dotenv()

"""
-------------------------------------------------------
In this example, we explore Agno with the following features:
- Step-based workflows with the Workflow class
- Sequential steps using Step with agents
- Parallel step execution using Parallel
- Custom executor functions for non-agent steps

Workflows in Agno v2 are declarative pipelines made of Steps.
Each Step wraps an agent (or a custom executor function) and
runs in sequence. Parallel runs multiple steps concurrently
and merges their outputs. This replaces the old class-based
workflow pattern from v1.

For more details, visit:
https://docs.agno.com/workflows/introduction
-------------------------------------------------------
"""


# --- 1. Define custom executor functions ---
@tool
def get_stock_price(ticker: str) -> str:
    """Get the current stock price for a ticker symbol.

    Args:
        ticker: Stock ticker symbol (e.g., AAPL).

    Returns:
        A string with the stock price.
    """
    prices = {"AAPL": "$195.50", "GOOGL": "$142.30", "MSFT": "$420.10"}
    return prices.get(ticker.upper(), f"Price not found for {ticker}")


# --- 2. Create specialized agents ---
market_analyst = Agent(
    name="Market Analyst",
    model=OpenAIChat(id=settings.OPENAI_MODEL_NAME),
    instructions="You analyze stock market data. Be concise and data-driven.",
    tools=[get_stock_price],
)

report_writer = Agent(
    name="Report Writer",
    model=OpenAIChat(id=settings.OPENAI_MODEL_NAME),
    instructions=[
        "You write clear financial summary reports.",
        "Use bullet points and keep it under 200 words.",
    ],
    markdown=True,
)

risk_analyst = Agent(
    name="Risk Analyst",
    model=OpenAIChat(id=settings.OPENAI_MODEL_NAME),
    instructions="You assess investment risks. Be specific about risk factors.",
)


# --- 3. Build the workflow ---
workflow = Workflow(
    name="Stock Analysis Pipeline",
    description="Analyzes stocks and produces a summary report",
    steps=[
        # First: gather data in parallel from two analysts
        Parallel(
            Step(
                name="Analyze Prices",
                agent=market_analyst,
                description="Look up current prices for AAPL, GOOGL, and MSFT.",
            ),
            Step(
                name="Assess Risks",
                agent=risk_analyst,
                description="Assess the key investment risks for large-cap tech stocks in the current market.",
            ),
        ),
        # Then: synthesize into a report
        Step(
            name="Write Report",
            agent=report_writer,
            description="Combine the price analysis and risk assessment into a concise investment summary.",
        ),
    ],
)

# --- 4. Run the workflow ---
# The input= parameter provides the task for the first step (and Parallel steps).
# Step descriptions are metadata only — they are NOT passed as input to agents.
run_output = workflow.run(
    input="Look up the current stock prices for AAPL, GOOGL, and MSFT using the get_stock_price tool, then assess key investment risks for large-cap tech stocks, and produce a concise investment summary report."
)

# --- 5. Print the result ---
pprint_run_response(run_output)
