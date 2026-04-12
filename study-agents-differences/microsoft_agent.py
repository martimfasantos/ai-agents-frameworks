from datetime import date
from tavily import TavilyClient
import json
import time
import asyncio

# Microsoft Agent Framework imports (aligned with microsoft-agent-framework/ folder patterns)
from agent_framework import Agent as MSAgent, tool
from agent_framework.openai import OpenAIChatClient

from settings import settings
from utils import get_tools_descriptions, parse_args, execute_agent

# Prompt components
from prompts import role, goal, instructions, knowledge


# --- Tool Definitions (using @tool from agent_framework, per framework patterns) ---


@tool
def date_tool() -> str:
    """Gets the current date."""
    today = date.today()
    return today.strftime("%B %d, %Y")


@tool
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
        Initialize the Microsoft Agent Framework agent.
        Uses model= and imports from agent_framework.openai (aligned with
        agent-framework 1.0.0 API).
        """
        self.name = "Microsoft Agent Framework"
        self.verbose = verbose
        self.tokens = tokens
        self.memory = memory

        # Create the OpenAI chat client (model= per framework 1.0.0 patterns)
        if provider == "azure" and settings.azure_api_key:
            self.client = OpenAIChatClient(
                model=settings.azure_deployment_name,
                base_url=f"{settings.azure_endpoint}/deployments/{settings.azure_deployment_name}",
                api_key=settings.azure_api_key.get_secret_value(),
            )
        elif provider == "openai" and settings.openai_api_key:
            self.client = OpenAIChatClient(
                model=settings.openai_model_name,
                api_key=settings.openai_api_key.get_secret_value(),
            )
        else:
            self.client = OpenAIChatClient(
                model=settings.open_source_model_name,
            )

        # Create the agent
        self.agent = self.client.as_agent(
            name="ms_agent",
            instructions="\n".join([role, goal, instructions, knowledge]),
            tools=[date_tool, web_search_tool],
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

        Args:
            message (str): User's input message

        Returns:
            tuple: (response_text, exec_time, tokens_dict)
        """
        try:
            start = time.perf_counter()

            async def _run():
                result = await self.agent.run(message)
                return result

            result = asyncio.run(_run())

            end = time.perf_counter()
            exec_time = end - start

            # result.text is the correct accessor (per framework patterns)
            response_text = result.text

            if self.tokens:
                # Microsoft Agent Framework exposes usage on AgentResponse.usage_details
                # (a UsageDetails TypedDict with input_token_count, output_token_count,
                # total_token_count) OR on individual message Content items of type "usage".
                try:
                    usage = getattr(result, "usage_details", None)
                    if usage and isinstance(usage, dict):
                        tokens = {
                            "total_embedding_token_count": 0,
                            "prompt_llm_token_count": usage.get("input_token_count", 0)
                            or 0,
                            "completion_llm_token_count": usage.get(
                                "output_token_count", 0
                            )
                            or 0,
                            "total_llm_token_count": usage.get("total_token_count", 0)
                            or 0,
                        }
                    else:
                        # Fallback: scan message contents for usage content items
                        prompt_tok = 0
                        completion_tok = 0
                        for msg in getattr(result, "messages", []):
                            for content_item in getattr(msg, "contents", []):
                                ct = getattr(content_item, "type", "")
                                if ct == "usage":
                                    ud = getattr(content_item, "usage_details", None)
                                    if ud and isinstance(ud, dict):
                                        prompt_tok += (
                                            ud.get("input_token_count", 0) or 0
                                        )
                                        completion_tok += (
                                            ud.get("output_token_count", 0) or 0
                                        )
                        tokens = {
                            "total_embedding_token_count": 0,
                            "prompt_llm_token_count": prompt_tok,
                            "completion_llm_token_count": completion_tok,
                            "total_llm_token_count": prompt_tok + completion_tok,
                        }
                except Exception:
                    tokens = {}
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
