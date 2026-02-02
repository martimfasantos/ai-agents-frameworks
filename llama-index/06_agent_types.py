# SHow ReAct, Functiion, CodeAct and other agent types in llama-index
from re import S
from typing import Any, Dict, Tuple
from llama_index.llms.openai import OpenAI
from llama_index.core.agent import ReActAgent, CodeActAgent, FunctionAgent
from llama_index.core.tools import FunctionTool
from settings import settings


"""
-------------------------------------------------------
In this example, we explore LlamaIndex with the following features:
- ReAct Agent for reasoning and acting with tools
- Function Agent for function calling with multiple tools
- CodeAct Agent for executing code with state persistence

These agent types enable advanced interactions with tools and code execution,
facilitating complex workflows and dynamic responses.

For more details, visit:
https://developers.llamaindex.ai/python/examples/cookbooks/llama3_cookbook_ollama_replicate/#react-agent
https://developers.llamaindex.ai/python/examples/cookbooks/oreilly_course_cookbooks/module-6/agents/#with-function-calling
https://developers.llamaindex.ai/python/examples/agent/code_act_agent/
-------------------------------------------------------
"""

# --- 1. Create the LLM ---
llm = OpenAI(
    model=settings.OPENAI_MODEL_NAME,
    api_key=settings.OPENAI_API_KEY.get_secret_value(),
)

# --- 2. Create an example tool ---
def multiply(a: int, b: int) -> int:
    """Multiple two integers and returns the result integer"""
    return a * b

multiply_tool = FunctionTool.from_defaults(fn=multiply)


# --- 3. Create the 3 agent types ---
# 3.1 ReAct Agent
react_agent = ReActAgent(
    tools=[multiply_tool],
    llm=llm,
    verbose=True,
)

# 3.2 Function Agent
function_agent = FunctionAgent(
    tools=[multiply_tool],
    llm=llm,
)

# 3.3 CodeAct Agent
class SimpleCodeExecutor:
    """
    A simple code executor that runs Python code with state persistence.

    This executor maintains a global and local state between executions,
    allowing for variables to persist across multiple code runs.
    """

    def __init__(self, locals: Dict[str, Any], globals: Dict[str, Any]):
        # State that persists between executions
        self.globals = globals
        self.locals = locals
    
    def execute(self, code: str) -> Tuple[bool, str, Any]:
        """
        Execute Python code and capture output and return values.
        NOTE: This is a placeholder.
        """
        return True, "Execution successful", eval(code, self.globals, self.locals)

code_agent = CodeActAgent(
    code_execute_fn=SimpleCodeExecutor(locals={}, globals={}).execute,
    tools=[multiply_tool],
    llm=llm,
    verbose=True,
)

# 3.4 Agent Workflow
from llama_index.core.agent.workflow import AgentWorkflow

multi_agent = AgentWorkflow(
    agents=[react_agent, function_agent],
    root_agent=react_agent,
    initial_state={
        "state": "start"
    }
)
