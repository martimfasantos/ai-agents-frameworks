import asyncio
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.agent import FunctionAgent
from llama_index.core.tools import QueryEngineTool, FunctionTool
from llama_index.core.workflow import (
    Workflow,
    StartEvent,
    StopEvent,
    step,
    Context,
    Event
)
from settings import settings


"""
-------------------------------------------------------
In this example, we explore LlamaIndex with the following features:
- Event-based streaming with custom workflow
- Token-level streaming for real-time output
- Custom event definitions and handling
- Workflow-based streaming architecture
- Real-time event processing and monitoring

LlamaIndex provides first-class streaming support through its event-driven
workflow architecture. This enables real-time output and monitoring of agent
operations, essential for building responsive user interfaces.

For more details, visit:
https://developers.llamaindex.ai/python/framework/understanding/agent/streaming/
-------------------------------------------------------
"""
from typing import List
from pydantic import BaseModel, Field


class Song(BaseModel):
    """Data model for a song."""

    title: str
    length_seconds: int


class Album(BaseModel):
    """Data model for an album."""

    name: str
    artist: str
    songs: List[Song]
    

from llama_index.core.llms import ChatMessage

llm = OpenAI(
    model=settings.OPENAI_MODEL_NAME,
    api_key=settings.OPENAI_API_KEY.get_secret_value(),
)
sllm = llm.as_structured_llm(output_cls=Album)

# Create an agent workflow with our calculator tool
agent = FunctionAgent(
    name="multiply_agent",
    description="A simple multiply agent.",
    llm=sllm,
    system_prompt="You are a helpful assistant that can multiply two numbers.",
    streaming=True,
)


async def main():
    # Run the agent
    response = await agent.run("Generate an example album from The Shining")
    print(str(response))
    
    # Run the LLM directly for structured output
    from IPython.display import clear_output
    from pprint import pprint

    stream_output = sllm.stream_chat([input_msg])
    for partial_output in stream_output:
        clear_output(wait=True)
        pprint(partial_output.raw.dict())

    output_obj = partial_output.raw
    print(str(output))

if __name__ == "__main__":
    asyncio.run(main())