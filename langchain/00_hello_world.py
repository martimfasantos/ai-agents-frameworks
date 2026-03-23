import os

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore LangChain with the following features:
- Creating a simple agent with create_agent()
- Initializing a chat model with init_chat_model()
- Invoking an agent with a single message

This is the simplest possible LangChain agent. It uses the new
create_agent() API to build a minimal agent that can respond to
a user question without any tools or configuration.

For more details, visit:
https://docs.langchain.com/oss/python/langchain/quick-start
-------------------------------------------------------
"""

# --- 1. Create the agent ---
agent = create_agent(
    model=init_chat_model(f"openai:{settings.OPENAI_MODEL_NAME}"),
    tools=[],
    system_prompt="Be concise, reply with one sentence.",
)

# --- 2. Invoke the agent ---
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Where does 'hello world' come from?"}]}
)

# --- 3. Print the result ---
print(result["messages"][-1].content)
