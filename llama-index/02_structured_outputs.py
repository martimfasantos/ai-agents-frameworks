import asyncio
from pydantic import BaseModel
from llama_index.llms.openai import OpenAI
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.agent.workflow.workflow_events import AgentOutput
from settings import settings


"""
-------------------------------------------------------
In this example, we explore LlamaIndex with the following features:
- Pydantic models for structured data validation
- Structured LLMs for enforcing output schemas
- Agents with structured output capabilities

Structured outputs ensure that agent responses conform to a specific schema,
enabling type-safe integration with downstream systems. LlamaIndex uses Pydantic
models to define and validate these schemas.

For more details, visit:
https://docs.llamaindex.ai/en/stable/module_guides/querying/structured_outputs/
-------------------------------------------------------
"""

# --- 1. Define the structured output models ---
class Song(BaseModel):
    """Data model for a song."""
    title: str
    length_seconds: int

# --- 2. Create the LLM and Structured LLM ---
# 2.1 Create the base LLM
llm = OpenAI(
    model=settings.OPENAI_MODEL_NAME,
    api_key=settings.OPENAI_API_KEY.get_secret_value(),
)
# 2.2 Create the structured LLM with Album output schema
sllm = llm.as_structured_llm(output_cls=Song)

# --- 3. Create the agent with structured output schema ---
agent = ReActAgent(
    name="music_agent",
    description="A simple music agent.",
    system_prompt="You are a helpful assistant that can generate song information.",
    llm=llm,  # use the normal LLM
    output_cls=Song,  # enforce structured output at the agent level
)

#--- 4. Run the agent and structured LLM ---
async def main():
    # 4.1 Run the agent
    response: AgentOutput = await agent.run("Suggest a song by Don Toliver.")
    print("Agent structured response: ", response.structured_response)
    print("Agent response as Song object: ", response.get_pydantic_model(Song))
    print(response)
    
    # 4.2 Run the LLM directly for structured output
    output = sllm.complete("Suggest a song by Don Toliver.")
    output_obj: Song = output.raw  # get actual object
    print("\nIs output of type Song? ", isinstance(output_obj, Song))
    print(output_obj)

if __name__ == "__main__":
    asyncio.run(main())
