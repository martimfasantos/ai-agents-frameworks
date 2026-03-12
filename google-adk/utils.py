import logging
from typing import Any, Optional

from google.adk.agents import (
    BaseAgent,
)  # common base for LlmAgent, SequentialAgent, etc.
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part, FunctionCall, FunctionResponse

# Suppress the SDK's "non-text parts in response" warning — it fires whenever
# a streaming event contains a function_call part alongside text parts and is
# purely informational noise in example output.
logging.getLogger("google_genai.types").setLevel(logging.ERROR)

APP_NAME = "my_agent"
USER_ID = "user"
SESSION_ID = "1234"


# Session and Runner
async def setup_session_and_runner(
    agent: BaseAgent,
    state: Optional[dict[str, Any]] = None,
):
    session_service = InMemorySessionService()
    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        state=state,
        session_id=SESSION_ID,
    )
    runner = Runner(agent=agent, app_name=APP_NAME, session_service=session_service)
    return session, runner


# Agent Interaction
async def call_agent_async(
    agent: BaseAgent,
    query: str,
    *,
    tool_calls: bool = False,
    tool_call_results: bool = False,
    state: Optional[dict[str, Any]] = None,
):
    content = Content(role="user", parts=[Part(text=query)])
    _, runner = await setup_session_and_runner(agent, state=state)
    events = runner.run_async(
        user_id=USER_ID, session_id=SESSION_ID, new_message=content
    )

    async for event in events:
        print(f"\n[{event.author.upper()}] ", end="")
        if event.is_final_response():
            if event.content and event.content.parts:
                final_response = event.content.parts[0].text
                print(f"{final_response}\n")
            else:
                print("\n")

        if tool_calls:
            function_calls = event.get_function_calls()
            if function_calls:
                handle_tool_calls(function_calls)
        if tool_call_results:
            function_responses = event.get_function_responses()
            if function_responses:
                handle_tool_responses(function_responses)


def handle_tool_calls(function_calls: list[FunctionCall]):
    for call in function_calls:
        tool_name = call.name
        arguments = call.args
        print(f"  Tool: {tool_name}, Args: {arguments}")
    print("")


def handle_tool_responses(function_responses: list[FunctionResponse]):
    for response in function_responses:
        tool_name = response.name
        result_dict = response.response
        print(f"  Tool Result: {tool_name} -> {result_dict}")
    print("")


def print_new_section(title: str):
    print("\n" + "-" * 65 + "\n" + " " * 15 + title + " " * 15 + "\n" + "-" * 65 + "\n")
