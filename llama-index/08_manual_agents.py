from typing import List
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage
from llama_index.core.llms.llm import ToolSelection
from llama_index.core.tools import FunctionTool
from settings import settings


"""
In this example, we explore LlamaIndex with the following features:
- Using LLMs directly for custom agent loops
- Tool calling with LLMs

While the agent classes like FunctionAgent, ReActAgent, CodeActAgent, and AgentWorkflow 
abstract away a lot of details, sometimes its desirable to build your own lower-level agents.

Using the LLM objects directly, you can quickly implement a basic agent loop, 
while having full control over how the tool calling and error handling works.

For more details, visit:
https://developers.llamaindex.ai/python/framework/module_guides/deploying/agents/#manual-agents
"""

# --- 1. Create a LLM instance ---
llm = OpenAI(
    model=settings.OPENAI_MODEL_NAME,
    api_key=settings.OPENAI_API_KEY.get_secret_value()
)

# --- 2. Define a tool ---
def select_song(song_name: str) -> str:
    """Useful for selecting a song."""
    return f"Song selected: {song_name}"

tools = [FunctionTool.from_defaults(select_song)]

# --- 3. Call llm with initial tools + chat history ---
chat_history = [ChatMessage(role="user", content="Pick a random song for me")]
resp = llm.chat_with_tools(tools, chat_history=chat_history)

# --- 4. Get tool calls from response ---
tool_calls: List[ToolSelection] = llm.get_tool_calls_from_response(
    resp, error_on_no_tool_call=False
)

# --- 5. Loop until no more tool calls ---
while tool_calls:
    # add the LLM's response to the chat history
    chat_history.append(resp.message)

    # call every tool and add its result to chat_history
    for tool_call in tool_calls:
        tool_name = tool_call.tool_name
        tool_kwargs = tool_call.tool_kwargs

        print(f"Calling {tool_name} with {tool_kwargs}")
        if tool_name == "select_song":
            tool_output = select_song(**tool_kwargs)
            chat_history.append(
                ChatMessage(
                    role="tool",
                    content=str(tool_output),
                    # most LLMs like OpenAI need to know the tool call id
                    additional_kwargs={"tool_call_id": tool_call.tool_id},
                )
            )
        else:
            tool_output = "Unknown tool"

        # check if the LLM can write a final response or calls more tools
        resp = llm.chat_with_tools(tools, chat_history=chat_history)
        tool_calls = llm.get_tool_calls_from_response(
            resp, error_on_no_tool_call=False
        )

# --- 6. Print final response ---
print(resp.message.content)