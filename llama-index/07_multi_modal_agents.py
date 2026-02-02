from llama_index.llms.openai import OpenAI
from llama_index.core.agent import FunctionAgent
from llama_index.core.llms import ChatMessage, ImageBlock, TextBlock
from settings import settings

"""
-------------------------------------------------------
In this example, we explore LlamaIndex with the following features:
- Multi-modal agents with image and text inputs
- Using ChatMessage with content blocks for rich inputs
- Agents that can reason over multiple modalities

Some LLMs will support multiple modalities, such as images and text. 
Using chat messages with content blocks, we can pass in images to an agent for reasoning.

For more details, visit:
https://developers.llamaindex.ai/python/framework/module_guides/deploying/agents/#multi-modal-agents
-------------------------------------------------------
"""

# --- 1. Create a tool ---
def add(a: int, b: int) -> int:
    """Useful for adding two numbers together."""
    return a + b

# --- 2. Create the agent ---
workflow = FunctionAgent(
    tools=[add],
    llm=OpenAI(
        model=settings.OPENAI_MODEL_NAME,
        api_key=settings.OPENAI_API_KEY.get_secret_value(),
    ),
)

# --- 3. Create a multi-modal message ---
msg = ChatMessage(
    role="user",
    blocks=[
        TextBlock(text="Follow what the image says."),
        ImageBlock(path="./res/tool_example.png"),
    ],
)

# --- 4. Run the workflow ---
async def main():
    response = await workflow.run(msg)
    print(str(response))

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())