from datetime import date
from tavily import TavilyClient
import json
import time
import asyncio
from typing import Any

# Claude Agent SDK imports (aligned with claude-agents-sdk/ folder patterns)
from claude_agent_sdk import (
    query,
    tool,
    ClaudeAgentOptions,
    ResultMessage,
    create_sdk_mcp_server,
)

from settings import settings
from utils import get_tools_descriptions, parse_args, execute_agent

# Prompt components
from prompts import role, goal, instructions, knowledge


# --- Tool Definitions (async tools returning MCP dict, per framework patterns) ---


@tool(
    "date_tool",
    "Gets the current date",
    {},
)
async def date_tool(args: dict[str, Any]) -> dict[str, Any]:
    """Gets the current date."""
    today = date.today()
    return {"content": [{"type": "text", "text": today.strftime("%B %d, %Y")}]}


@tool(
    "web_search_tool",
    "Searches the web for information using Tavily",
    {"query_text": str},
)
async def web_search_tool(args: dict[str, Any]) -> dict[str, Any]:
    """Searches the web for information."""
    tavily_client = TavilyClient(api_key=settings.tavily_api_key.get_secret_value())
    query_text = args["query_text"]
    search_response = tavily_client.search(query_text)
    results = json.dumps(search_response.get("results", []))
    return {"content": [{"type": "text", "text": results}]}


# --- Bundle tools into an in-process MCP server (per framework patterns) ---
tools_server = create_sdk_mcp_server(
    name="agent_tools",
    version="1.0.0",
    tools=[date_tool, web_search_tool],
)


class Agent:
    def __init__(
        self,
        provider: str = "openai",
        memory: bool = True,
        verbose: bool = False,
        tokens: bool = False,
    ):
        """
        Initialize the Claude Agent SDK agent.
        Uses ClaudeAgentOptions + create_sdk_mcp_server() (aligned with
        claude-agents-sdk/ folder patterns).
        Note: Claude Agent SDK requires the Claude Code CLI and uses Anthropic models.
        """
        self.name = "Claude Agent SDK"
        self.verbose = verbose
        self.tokens = tokens
        self.memory = memory
        self.messages = []

        self.system_prompt = "\n".join([role, goal, instructions, knowledge])

        # Extras:
        self.tools_descriptions = get_tools_descriptions(
            [
                ("date_tool", "Gets the current date"),
                ("web_search_tool", "Searches the web for information"),
            ]
        )

    def chat(self, message):
        """
        Send a message and get a response.
        Uses ClaudeAgentOptions with MCP servers (per framework patterns).

        Args:
            message (str): User's input message

        Returns:
            tuple: (response_text, exec_time, tokens_dict)
        """
        try:
            start = time.perf_counter()

            async def _run():
                final_text = ""
                prompt_tokens = 0
                completion_tokens = 0

                options = ClaudeAgentOptions(
                    system_prompt=self.system_prompt,
                    mcp_servers={"agent_tools": tools_server},
                    allowed_tools=["mcp__agent_tools__*"],
                )

                async for msg in query(
                    prompt=message,
                    options=options,
                ):
                    if isinstance(msg, ResultMessage) and msg.subtype == "success":
                        final_text = msg.result
                        # ResultMessage carries usage as a dict with
                        # input_tokens, output_tokens, etc.
                        if hasattr(msg, "usage") and msg.usage:
                            usage = msg.usage
                            if isinstance(usage, dict):
                                prompt_tokens += usage.get("input_tokens", 0)
                                completion_tokens += usage.get("output_tokens", 0)
                            else:
                                prompt_tokens += getattr(usage, "input_tokens", 0)
                                completion_tokens += getattr(usage, "output_tokens", 0)

                return final_text, prompt_tokens, completion_tokens

            response_text, prompt_tok, completion_tok = asyncio.run(_run())

            end = time.perf_counter()
            exec_time = end - start

            if self.tokens:
                tokens = {
                    "total_embedding_token_count": 0,
                    "prompt_llm_token_count": prompt_tok,
                    "completion_llm_token_count": completion_tok,
                    "total_llm_token_count": prompt_tok + completion_tok,
                }
            else:
                tokens = {}

            return response_text, exec_time, tokens

        except Exception as e:
            print(f"Error in chat: {e}")
            return "Sorry, I encountered an error processing your request.", 0.0, {}

    def clear_chat(self):
        """Reset the conversation context."""
        self.messages = []
        return True


def main():
    """
    Example usage demonstrating the agent interface.
    """
    args = parse_args()

    agent = Agent(
        provider=args.provider,
        memory=not args.no_memory,
        verbose=args.verbose,
        tokens=args.mode in ["metrics", "metrics-loop"],
    )

    execute_agent(agent, args)


if __name__ == "__main__":
    main()
