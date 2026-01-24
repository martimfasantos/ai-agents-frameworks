from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.llms import ChatMessage, ImageBlock, TextBlock
from llama_index.llms.openai import OpenAI

llm = OpenAI(model="gpt-4o-mini", api_key="sk-...")


def add(a: int, b: int) -> int:
    """Useful for adding two numbers together."""
    return a + b


workflow = FunctionAgent(
    tools=[add],
    llm=llm,
)

msg = ChatMessage(
    role="user",
    blocks=[
        TextBlock(text="Follow what the image says."),
        ImageBlock(path="./screenshot.png"),
    ],
)

response = await workflow.run(msg)
print(str(response))