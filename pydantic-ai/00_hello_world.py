from dotenv import load_dotenv

from pydantic_ai import Agent

from settings import settings

load_dotenv()

"""
-------------------------------------------------------
In this example, we explore a simple Hello World agent
-------------------------------------------------------
"""

agent = Agent(  
    model=settings.OPENAI_MODEL_NAME,
    instructions='Be concise, reply with one sentence.',  
)

result = agent.run_sync('Where does "hello world" come from?')  
print(result.output)
