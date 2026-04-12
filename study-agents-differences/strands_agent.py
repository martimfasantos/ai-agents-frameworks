from datetime import date
from tavily import TavilyClient
import json
import time

# Strands Agents SDK imports
from strands import Agent as StrandsAgent, tool as strands_tool
from strands.models.openai import OpenAIModel

from settings import settings
from utils import get_tools_descriptions, parse_args, execute_agent

# Prompt components
from prompts import role, goal, instructions, knowledge


# --- Tool Definitions ---


@strands_tool
def date_tool() -> str:
    """Gets the current date."""
    today = date.today()
    return today.strftime("%B %d, %Y")


@strands_tool
def web_search_tool(query: str) -> str:
    """Searches the web for information using Tavily.

    Args:
        query: The search query string.
    """
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
        Initialize the Strands Agents SDK agent.
        """
        self.name = "Strands Agent"
        self.verbose = verbose
        self.tokens = tokens
        self.memory = memory

        # Determine the model
        if provider == "azure" and settings.azure_api_key:
            self.model = OpenAIModel(
                model_id=settings.azure_deployment_name,
                client_args={
                    "base_url": f"{settings.azure_endpoint}/deployments/{settings.azure_deployment_name}",
                    "api_key": settings.azure_api_key.get_secret_value(),
                    "default_headers": {"api-version": settings.azure_api_version},
                },
            )
        elif provider == "openai" and settings.openai_api_key:
            self.model = OpenAIModel(
                model_id=settings.openai_model_name,
                client_args={
                    "api_key": settings.openai_api_key.get_secret_value(),
                },
            )
        else:
            self.model = OpenAIModel(model_id=settings.open_source_model_name)

        # Create tools
        self.tools = [date_tool, web_search_tool]

        # Create the agent
        self.agent = StrandsAgent(
            model=self.model,
            system_prompt="\n".join([role, goal, instructions, knowledge]),
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
            result = self.agent(message)
            end = time.perf_counter()
            exec_time = end - start

            # result.message is the correct accessor (per framework patterns)
            response_text = (
                str(result.message) if hasattr(result, "message") else str(result)
            )

            if self.tokens:
                # Strands provides metrics via result.metrics.get_summary()
                # Token usage is under "accumulated_usage" with camelCase keys
                prompt_tokens = 0
                completion_tokens = 0
                total_tokens = 0
                try:
                    metrics = getattr(result, "metrics", None)
                    if metrics and hasattr(metrics, "get_summary"):
                        summary = metrics.get_summary()
                        usage = summary.get("accumulated_usage", {})
                        prompt_tokens = usage.get("inputTokens", 0)
                        completion_tokens = usage.get("outputTokens", 0)
                        total_tokens = usage.get("totalTokens", 0)
                except Exception:
                    pass

                tokens = {
                    "total_embedding_token_count": 0,
                    "prompt_llm_token_count": prompt_tokens,
                    "completion_llm_token_count": completion_tokens,
                    "total_llm_token_count": total_tokens
                    or (prompt_tokens + completion_tokens),
                }
            else:
                tokens = {}

            # Clear conversation if memory is disabled
            if not self.memory:
                self.agent.messages = []

            return response_text, exec_time, tokens

        except Exception as e:
            print(f"Error in chat: {e}")
            return "Sorry, I encountered an error processing your request.", 0.0, {}

    def clear_chat(self):
        """Reset the conversation context."""
        try:
            self.agent.messages = []
            return True
        except Exception as e:
            print(f"Error clearing chat: {e}")
            return False


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
