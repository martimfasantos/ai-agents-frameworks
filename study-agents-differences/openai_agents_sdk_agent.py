from datetime import date
from tavily import TavilyClient
import json
import time
import asyncio

# OpenAI Agents SDK imports
from agents import Agent as OAIAgent, Runner, function_tool

from settings import settings
from utils import get_tools_descriptions, parse_args, execute_agent

# Prompt components
from prompts import role, goal, instructions, knowledge


# --- Tool Definitions ---


@function_tool
def date_tool() -> str:
    """Gets the current date."""
    today = date.today()
    return today.strftime("%B %d, %Y")


@function_tool
def web_search_tool(query: str) -> str:
    """Searches the web for information using Tavily."""
    tavily_client = TavilyClient(api_key=settings.tavily_api_key.get_secret_value())
    search_response = tavily_client.search(query)
    results = json.dumps(search_response.get("results", []))
    return results


class Agent:
    def __init__(
        self,
        provider: str = "openai",
        memory: bool = True,
        verbose: bool = False,
        tokens: bool = False,
    ):
        """
        Initialize the OpenAI Agents SDK agent.
        Note: This uses the new OpenAI Agents SDK (openai-agents), not the raw OpenAI API.
        """
        self.name = "OpenAI Agents SDK"
        self.verbose = verbose
        self.tokens = tokens
        self.memory = memory

        # Determine model
        if provider == "openai" and settings.openai_api_key:
            model = settings.openai_model_name
        else:
            model = "gpt-4o-mini"

        # Create tools
        self.tools = [date_tool, web_search_tool]

        # Create the agent
        self.agent = OAIAgent(
            name="openai_agents_sdk_agent",
            instructions="\n".join([role, goal, instructions, knowledge]),
            model=model,
            tools=self.tools,
        )

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
        """
        try:
            start = time.perf_counter()

            # OpenAI Agents SDK uses Runner.run_sync
            result = Runner.run_sync(self.agent, message)

            end = time.perf_counter()
            exec_time = end - start

            response_text = result.final_output

            if self.tokens:
                # Extract usage from result.raw_responses — each has a Usage object
                # with input_tokens, output_tokens, total_tokens (not prompt_/completion_)
                raw_responses = getattr(result, "raw_responses", [])
                prompt_tokens = 0
                completion_tokens = 0
                for resp in raw_responses:
                    if hasattr(resp, "usage") and resp.usage:
                        prompt_tokens += getattr(resp.usage, "input_tokens", 0)
                        completion_tokens += getattr(resp.usage, "output_tokens", 0)

                tokens = {
                    "total_embedding_token_count": 0,
                    "prompt_llm_token_count": prompt_tokens,
                    "completion_llm_token_count": completion_tokens,
                    "total_llm_token_count": prompt_tokens + completion_tokens,
                }
            else:
                tokens = {}

            return response_text, exec_time, tokens

        except Exception as e:
            print(f"Error in chat: {e}")
            return "Sorry, I encountered an error processing your request.", 0.0, {}

    def clear_chat(self):
        """Reset the conversation context."""
        return True


def main():
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
