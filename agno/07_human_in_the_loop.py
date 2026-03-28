from dotenv import load_dotenv

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools import tool
from agno.utils.pprint import pprint_run_response

from settings import settings

load_dotenv()

"""
-------------------------------------------------------
In this example, we explore Agno with the following features:
- Human-in-the-loop (HITL) with @tool(requires_confirmation=True)
- Checking run_output.requirements for pending confirmations
- Approving or rejecting tool calls with agent.continue_run()

HITL lets you gate sensitive tool executions behind human
approval. When a tool has requires_confirmation=True, the
agent pauses before executing it and returns a RunOutput
with pending requirements. You then confirm or reject each
requirement and call continue_run() to resume.

For more details, visit:
https://docs.agno.com/agents/human-in-the-loop
-------------------------------------------------------
"""


# --- 1. Define tools with confirmation requirement ---
@tool(requires_confirmation=True)
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to a recipient.

    Args:
        to: The recipient's email address.
        subject: The email subject line.
        body: The email body text.

    Returns:
        Confirmation that the email was sent.
    """
    return f"Email sent to {to} with subject '{subject}'."


@tool
def draft_email(to: str, subject: str, body: str) -> str:
    """Draft an email without sending it (no confirmation needed).

    Args:
        to: The recipient's email address.
        subject: The email subject line.
        body: The email body text.

    Returns:
        Confirmation that the draft was saved.
    """
    return f"Draft saved: to={to}, subject='{subject}'."


# --- 2. Create the agent ---
agent = Agent(
    model=OpenAIChat(id=settings.OPENAI_MODEL_NAME),
    tools=[send_email, draft_email],
    instructions="You are an email assistant. When asked to send an email, use the send_email tool.",
    markdown=True,
)

# --- 3. Run the agent — this will pause at the send_email tool ---
print("=== Initial run (will pause for confirmation) ===\n")
run_output = agent.run(
    "Send an email to alice@example.com with subject 'Meeting Tomorrow' and body 'Hi Alice, confirming our meeting at 3pm.'"
)
pprint_run_response(run_output)

# --- 4. Check for pending requirements ---
if run_output.requirements:
    print(f"\nPending confirmations: {len(run_output.requirements)}")
    for req in run_output.requirements:
        tool_exec = req.tool_execution
        if tool_exec:
            print(f"  Tool: {tool_exec.tool_name}")
            print(f"  Args: {tool_exec.tool_args}")

    # --- 5. Approve the requirements and continue ---
    # In a real app, you'd show this to the user and get their decision.
    # Here we simulate approval using the confirm() method on each requirement.
    for req in run_output.requirements:
        req.confirm()

    print("\n=== Continuing run after approval ===\n")
    continued_output = agent.continue_run(
        run_response=run_output,
        requirements=run_output.requirements,
    )
    pprint_run_response(continued_output)
else:
    print("\nNo confirmations needed — tool executed directly.")
