import asyncio
from typing import Literal
from pydantic import BaseModel, Field
from rich.prompt import Prompt

from pydantic_ai import Agent, ModelMessage, RunContext, RunUsage, UsageLimits

from dotenv import load_dotenv

from utils import show_metrics
from settings import settings

load_dotenv()

"""
-----------------------------------------------------------------------
In this example, we explore Pydantic AI with the following features:
- Multi-Agent Applications patterns
- Multiple agents called in succession by application code (not via tools)
- Passing message history between agent runs for context continuity
- Shared usage tracking across multiple agent runs
- Application code orchestrates which agent runs next

Programmatic hand-off differs from agent delegation by having application code control
the workflow rather than agents calling each other. Each specialized agent runs
independently, and the application passes context via message history between runs.

Reference: https://ai.pydantic.dev/multi-agent-applications/#programmatic-agent-hand-off
-----------------------------------------------------------------------
"""

# --- 1. Define output models ---
class FlightDetails(BaseModel):
    flight_number: str

class Failed(BaseModel):
    """Unable to find a satisfactory choice."""

# --- 2. Create flight search agent ---
flight_search_agent = Agent[None, FlightDetails | Failed](  
    model=settings.OPENAI_MODEL_NAME,
    output_type=FlightDetails | Failed,
    system_prompt=(
        "Use the 'flight_search' tool to find a flight "
        "from the given origin to the given destination."
    ),
)


@flight_search_agent.tool  
async def flight_search(
    ctx: RunContext[None], origin: str, destination: str
) -> FlightDetails | None:
    # In reality, this would call a flight search API or
    # use a browser to scrape a flight search website
    return FlightDetails(flight_number='AK456')

# Set usage limits to prevent excessive API calls
usage_limits = UsageLimits(request_limit=3)  

# --- 3. Define workflow functions ---
async def find_flight(usage: RunUsage) -> FlightDetails | None:
    """Find a flight by prompting the user and using the flight search agent."""
    message_history: list[ModelMessage] | None = None
    # Allow up to 3 attempts to find a flight
    for _ in range(3):
        prompt = Prompt.ask(
            "Where would you like to fly from and to?",
        )
        result = await flight_search_agent.run(
            prompt,
            message_history=message_history,
            usage=usage,  # Track cumulative usage
            usage_limits=usage_limits,
        )
        if isinstance(result.output, FlightDetails):
            return result.output
        else:
            # If failed, preserve message history and retry
            message_history = result.all_messages(
                output_tool_return_content="Please try again."
            )

# --- 4. Define seat preference models and agent ---
class SeatPreference(BaseModel):
    row: int = Field(ge=1, le=30)
    seat: Literal['A', 'B', 'C', 'D', 'E', 'F']

seat_preference_agent = Agent[None, SeatPreference | Failed](  
    model=settings.OPENAI_MODEL_NAME,
    output_type=SeatPreference | Failed,
    system_prompt=(
        "Extract the user's seat preference. "
        "Seats A and F are window seats. "
        "Row 1 is the front row and has extra leg room. "
        "Rows 14, and 20 also have extra leg room. "
    ),
)

async def find_seat(usage: RunUsage) -> SeatPreference:
    """Find a seat by prompting the user and using the seat preference agent."""
    message_history: list[ModelMessage] | None = None
    # Retry until we get a valid seat preference
    while True:
        answer = Prompt.ask("What seat would you like?")

        result = await seat_preference_agent.run(
            answer,
            message_history=message_history,
            usage=usage,  # Track cumulative usage
            usage_limits=usage_limits,
        )
        if isinstance(result.output, SeatPreference):
            return result.output
        else:
            # If failed, preserve message history and retry
            print('Could not understand seat preference. Please try again.')
            message_history = result.all_messages()


# --- 5. Run the sequential workflow ---
async def main():
    # Create shared usage tracker for all agent runs
    usage: RunUsage = RunUsage()

    # Step 1: Find flight using first agent
    opt_flight_details = await find_flight(usage)
    if opt_flight_details is not None:
        print(f"✈️ Flight found: {opt_flight_details.flight_number}")
        #> Flight found: AK456
        
        # Step 2: Select seat using second agent
        # Usage is passed to track combined metrics across both agents
        seat_preference = await find_seat(usage)
        print(f"💺 Seat preference: {seat_preference}")
        #> Seat preference: row=1 seat='A'
        
        # Application code controlled the entire workflow, calling each agent
        # in sequence and passing the shared usage tracker between them
        show_metrics(usage)


if __name__ == '__main__':
    asyncio.run(main())
