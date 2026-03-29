import os

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore LangChain with the following features:
- Creating a simple agent with create_agent()
- Initializing a ChatOpenAI model with custom parameters
- Invoking an agent with a single message

This is the simplest possible LangChain agent. It uses the new
create_agent() API to build a minimal agent that can respond to
a user question without any tools or configuration.

For more details, visit:
https://docs.langchain.com/oss/python/langchain/agents
-------------------------------------------------------
"""

# --- 1. Create the model ---
model = ChatOpenAI(
    model=settings.OPENAI_MODEL_NAME,
    temperature=0.1,
    max_tokens=1000,
    timeout=30,
)

# --- 2. Create the agent ---
agent = create_agent(
    model=model,
    tools=[],
    system_prompt="Be concise, reply with one sentence.",
)

# --- 3. Invoke the agent ---
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Where does 'hello world' come from?"}]}
)

# --- 4. Print the result ---
print(result["messages"][-1].content)
