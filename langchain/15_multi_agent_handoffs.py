import os
from typing import Callable

from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain.messages import ToolMessage
from langchain.tools import tool, ToolRuntime
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore LangChain with the following features:
- Multi-agent handoffs pattern using a single agent with middleware
- State-driven behavior transitions via Command
- Dynamic prompt and tool switching based on current_step

In the handoffs pattern, a single agent changes behavior based on
a state variable. Tools update the state to trigger transitions,
and middleware reconfigures the agent's prompt and available tools
for each step. This is ideal for multi-stage workflows like
customer support, where steps must happen in sequence.

For more details, visit:
https://docs.langchain.com/oss/python/langchain/multi-agent/handoffs
-------------------------------------------------------
"""

# --- 1. Create the model ---
model = ChatOpenAI(
    model=settings.OPENAI_MODEL_NAME,
    temperature=0.1,
    max_tokens=1000,
    timeout=30,
)


# --- 2. Define custom state with step tracking ---
class SupportState(AgentState):
    """Track the current step in the support workflow."""

    current_step: str = "collect_info"
    issue_type: str = ""


# --- 3. Define step-specific tools ---


# Step 1: Collect information
@tool
def record_issue(
    issue_type: str,
    runtime: ToolRuntime[None, SupportState],
) -> Command:
    """Record the customer's issue type and move to the resolution step.

    Args:
        issue_type: The type of issue: 'billing', 'technical', or 'account'.
    """
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=f"Issue recorded: {issue_type}. Moving to resolution.",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
            "issue_type": issue_type,
            "current_step": "resolve",
        }
    )


# Step 2: Resolve the issue
@tool
def provide_solution(solution: str) -> str:
    """Provide a solution to the customer's issue.

    Args:
        solution: The solution or resolution to present.
    """
    return f"Solution provided: {solution}"


@tool
def escalate_to_human(reason: str) -> str:
    """Escalate the issue to a human agent.

    Args:
        reason: Why the issue needs human intervention.
    """
    return f"Escalated to human agent. Reason: {reason}"


# --- 4. Middleware to switch behavior based on current_step ---
STEP_CONFIGS = {
    "collect_info": {
        "prompt": (
            "You are a support agent in the INFORMATION COLLECTION phase. "
            "You MUST use the record_issue tool to categorize the customer's "
            "issue as 'billing', 'technical', or 'account'. Do NOT answer "
            "the question yet — just classify it and call record_issue."
        ),
        "tools": [record_issue],
    },
    "resolve": {
        "prompt": (
            "You are a support agent in the RESOLUTION phase. "
            "The customer's issue type is: {issue_type}. "
            "Use provide_solution to help, or escalate_to_human if needed."
        ),
        "tools": [provide_solution, escalate_to_human],
    },
}


@wrap_model_call
def apply_step_config(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Dynamically configure the agent based on the current step."""
    step = request.state.get("current_step", "collect_info")
    config = STEP_CONFIGS[step]

    prompt = config["prompt"]
    if "{issue_type}" in prompt:
        prompt = prompt.format(issue_type=request.state.get("issue_type", "unknown"))

    request = request.override(
        system_prompt=prompt,
        tools=config["tools"],
    )
    return handler(request)


# --- 5. Create the agent ---
agent = create_agent(
    model=model,
    tools=[record_issue, provide_solution, escalate_to_human],
    state_schema=SupportState,
    middleware=[apply_step_config],
    checkpointer=InMemorySaver(),
)

config = {"configurable": {"thread_id": "support-123"}}

# --- 6. Step 1: Collect information ---
print("=== Step 1: Collect Information ===")
result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Hi, I'm having trouble with my bill. I was charged twice.",
            }
        ]
    },
    config=config,
)
print(f"Agent: {result['messages'][-1].content}\n")

# --- 7. Step 2: Resolve (agent should now be in resolve mode) ---
print("=== Step 2: Resolution ===")
result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Can you help me get a refund for the duplicate charge?",
            }
        ]
    },
    config=config,
)
print(f"Agent: {result['messages'][-1].content}")
