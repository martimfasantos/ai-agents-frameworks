from strands import Agent, tool
from strands.models.openai import OpenAIModel

from settings import settings

"""
-------------------------------------------------------
In this example, we explore Strands Agents SDK with the following features:
- Agents-as-tools multi-agent pattern
- Specialized sub-agents wrapped as callable tools
- Orchestrator agent that routes to the right specialist

This creates a hierarchical multi-agent system where an orchestrator agent
decides which specialist to delegate to. Each specialist is a full Agent
wrapped inside a @tool function so the orchestrator can invoke it.

For more details, visit:
https://strandsagents.com/docs/user-guide/concepts/multi-agent/agents-as-tools/
-------------------------------------------------------
"""


# --- 1. Configure model ---
openai_model = OpenAIModel(
    client_args={
        "api_key": settings.OPENAI_API_KEY.get_secret_value()
        if settings.OPENAI_API_KEY
        else ""
    },
    model_id=settings.OPENAI_MODEL_NAME,
)


# --- 2. Define specialist agents as tools ---


@tool
def math_expert(question: str) -> str:
    """Solve mathematical problems with step-by-step explanations.

    Args:
        question: A math problem to solve
    """
    math_agent = Agent(
        model=openai_model,
        system_prompt="You are a math expert. Solve problems step by step. Be concise.",
        callback_handler=None,
    )
    result = math_agent(question)
    return str(result.message)


@tool
def history_expert(question: str) -> str:
    """Answer questions about historical events, people, and periods.

    Args:
        question: A history-related question
    """
    history_agent = Agent(
        model=openai_model,
        system_prompt="You are a history expert. Provide accurate, concise historical information.",
        callback_handler=None,
    )
    result = history_agent(question)
    return str(result.message)


@tool
def code_expert(question: str) -> str:
    """Write code snippets, explain programming concepts, and debug code.

    Args:
        question: A programming-related question or task
    """
    code_agent = Agent(
        model=openai_model,
        system_prompt="You are a senior software engineer. Write clean, well-documented code. Be concise.",
        callback_handler=None,
    )
    result = code_agent(question)
    return str(result.message)


# --- 3. Create the orchestrator agent ---
orchestrator = Agent(
    model=openai_model,
    system_prompt="""You are a helpful assistant that routes questions to specialized experts:
- For math problems -> use the math_expert tool
- For history questions -> use the history_expert tool
- For programming questions -> use the code_expert tool
- For simple questions -> answer directly

Always use the most appropriate expert for the question.""",
    tools=[math_expert, history_expert, code_expert],
)

# --- 4. Run the orchestrator ---
print("=== Multi-Agent Orchestration ===\n")
result = orchestrator(
    "I have two questions:\n"
    "1. What is the factorial of 7?\n"
    "2. Who was the first person to walk on the moon and in what year?"
)

# --- 5. Print results ---
print(f"\n--- Orchestrator Response ---\n{result.message}")
