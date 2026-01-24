
import asyncio
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI
from settings import settings


"""
-------------------------------------------------------
In this example, we explore LlamaIndex with the following features:
- Document loading and parsing
- Index creation for RAG
- Basic querying with vector search

This is the foundational pattern for building RAG applications with LlamaIndex.
LlamaIndex is a data framework designed for LLM applications with a focus on 
Retrieval-Augmented Generation (RAG).

For more details, visit:
https://docs.llamaindex.ai/en/stable/getting_started/starter_example/
-------------------------------------------------------
"""

# Create an agent workflow with our calculator tool
agent = FunctionAgent(
    name="hello_world_agent",
    description="A simple hello world agent.",
    llm=OpenAI(
        model=settings.OPENAI_MODEL_NAME,
        api_key=settings.OPENAI_API_KEY.get_secret_value()
    ),
    system_prompt="You are a helpful assistant that greets the user.",
)

async def main():
    # Run the agent
    response = await agent.run("Hello World!")
    print(str(response))

if __name__ == "__main__":
    asyncio.run(main())