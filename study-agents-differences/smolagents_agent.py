from datetime import date
from tavily import TavilyClient
import json
import time

# Smolagents imports (aligned with smolagents/ folder patterns)
from smolagents import (
    CodeAgent,
    tool as smolagent_tool,
    OpenAIModel,
    LiteLLMModel,
)

from settings import settings
from utils import get_tools_descriptions, parse_args, execute_agent

# Prompt components
from prompts import role, goal, instructions, knowledge


# --- Tool Definitions (using bare @tool decorator, per framework patterns) ---


@smolagent_tool
def date_tool() -> str:
    """Gets the current date.

    Returns:
        str: The current date formatted as 'Month Day, Year'.
    """
    today = date.today()
    return today.strftime("%B %d, %Y")


@smolagent_tool
def web_search_tool(query: str) -> str:
    """Searches the web for information using Tavily.

    Args:
        query: The search query string.

    Returns:
        str: JSON string of search results.
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
        Initialize the Smolagents agent.
        Uses CodeAgent + OpenAIModel (aligned with smolagents/ folder patterns,
        replacing ToolCallingAgent + OpenAIServerModel).
        """
        self.name = "Smolagents Agent"
        self.verbose = verbose
        self.tokens = tokens
        self.memory = memory

        # Determine the model (OpenAIModel for OpenAI, LiteLLMModel for Azure/other)
        if provider == "azure" and settings.azure_api_key:
            self.model = LiteLLMModel(
                model_id=f"azure/{settings.azure_deployment_name}",
                api_base=settings.azure_endpoint,
                api_key=settings.azure_api_key.get_secret_value(),
            )
        elif provider == "openai" and settings.openai_api_key:
            self.model = OpenAIModel(
                model_id=settings.openai_model_name,
                api_key=settings.openai_api_key.get_secret_value(),
            )
        else:
            self.model = LiteLLMModel(
                model_id=settings.open_source_model_name,
            )

        # Create tools
        self.tools = [date_tool, web_search_tool]

        # Create the agent - CodeAgent writes and executes Python code (per framework patterns)
        # prompt_templates replaces the removed system_prompt kwarg in newer smolagents
        custom_instructions = "\n".join([role, goal, instructions, knowledge])
        self.agent = CodeAgent(
            model=self.model,
            tools=self.tools,
            max_steps=5,
            verbosity_level=2 if verbose else 0,
        )
        # Prepend custom instructions to default system prompt
        self.agent.prompt_templates["system_prompt"] = (
            custom_instructions + "\n\n" + self.agent.prompt_templates["system_prompt"]
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
            response = self.agent.run(message)
            end = time.perf_counter()
            exec_time = end - start

            if self.tokens:
                # Smolagents tracks token usage per step in agent.memory.steps
                # Each ActionStep has a token_usage attr with input_tokens,
                # output_tokens, total_tokens
                input_tokens = 0
                output_tokens = 0
                try:
                    for step in self.agent.memory.steps:
                        token_usage = getattr(step, "token_usage", None)
                        if token_usage is not None:
                            input_tokens += getattr(token_usage, "input_tokens", 0)
                            output_tokens += getattr(token_usage, "output_tokens", 0)
                except Exception:
                    pass

                tokens = {
                    "total_embedding_token_count": 0,
                    "prompt_llm_token_count": input_tokens,
                    "completion_llm_token_count": output_tokens,
                    "total_llm_token_count": input_tokens + output_tokens,
                }
            else:
                tokens = {}

            # Reset memory if not persistent
            if not self.memory:
                self.agent.memory.steps = []

            return str(response), exec_time, tokens

        except Exception as e:
            print(f"Error in chat: {e}")
            return "Sorry, I encountered an error processing your request.", 0.0, {}

    def clear_chat(self):
        """Reset the conversation context."""
        try:
            self.agent.memory.steps = []
            return True
        except Exception as e:
            print(f"Error clearing chat: {e}")
            return False


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
