from datetime import date
from tavily import TavilyClient
import json
import time

# Agno imports (aligned with agno/ folder patterns)
from agno.models.openai import OpenAIChat
from agno.models.azure import AzureOpenAI
from agno.agent import Agent as AgnoAgent
from agno.tools import tool
from agno.tools.tavily import TavilyTools
from agno.models.huggingface import HuggingFace

from settings import settings
from utils import get_tools_descriptions, parse_args, execute_agent

# Prompt components
from prompts import role, goal, instructions, knowledge


class Agent:
    def __init__(
        self,
        provider: str = "openai",
        memory: bool = True,
        verbose: bool = False,
        tokens: bool = False,
    ):
        """
        Initialize the Agno agent.
        Uses instructions= and bare @tool (aligned with agno/ folder patterns).
        """
        self.name = "Agno Agent"

        self.model = (
            AzureOpenAI(
                base_url=f"{settings.azure_endpoint}/deployments/{settings.azure_deployment_name}",
                api_version=settings.azure_api_version,
                api_key=settings.azure_api_key.get_secret_value(),
            )
            if provider == "azure" and settings.azure_api_key
            else OpenAIChat(
                api_key=settings.openai_api_key.get_secret_value(),
                id=settings.openai_model_name,
            )
            if provider == "openai" and settings.openai_api_key
            else HuggingFace(
                model_name=settings.open_source_model_name,
            )
        )

        # Create tools
        self.tools = self._create_tools()

        # Create the Agent (using instructions= per framework patterns)
        self.agent = AgnoAgent(
            name="Agno Agent",
            model=self.model,
            tools=self.tools,
            instructions="\n".join(
                [
                    role,
                    goal,
                    instructions,
                    "You have access to two primary tools: date_tool and web_search_tool.",
                    knowledge,
                ]
            ),
            add_history_to_context=True if memory else False,
            read_chat_history=True if memory else False,
            markdown=True,
            debug_mode=True if verbose else False,
        )

        self.tokens = tokens

        # Extras:
        self.tools_descriptions = get_tools_descriptions(
            [(t.name, getattr(t, "description", str(t))) for t in self.tools]
        )

    @staticmethod
    @tool
    def date_tool():
        """Gets the current date.

        Returns:
            str: The current date formatted as 'Month Day, Year'.
        """
        today = date.today()
        return today.strftime("%B %d, %Y")

    @staticmethod
    @tool
    def web_search_tool(query: str):
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

    def _create_tools(self):
        """
        Create tools for the agent.

        Returns:
            List of tools
        """
        return [
            self.date_tool,
            TavilyTools(api_key=settings.tavily_api_key.get_secret_value()),
        ]

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
                # Agno 2.5+ returns RunOutput with a RunMetrics dataclass at
                # response.metrics (fields: input_tokens, output_tokens, total_tokens)
                try:
                    metrics = response.metrics
                    if metrics is not None:
                        tokens = {
                            "total_embedding_token_count": 0,
                            "prompt_llm_token_count": getattr(
                                metrics, "input_tokens", 0
                            ),
                            "completion_llm_token_count": getattr(
                                metrics, "output_tokens", 0
                            ),
                            "total_llm_token_count": getattr(
                                metrics, "total_tokens", 0
                            ),
                        }
                    else:
                        tokens = {}
                except Exception:
                    tokens = {}
            else:
                tokens = {}

            return response.content, exec_time, tokens

        except Exception as e:
            print(f"Error in chat: {e}")
            return "Sorry, I encountered an error processing your request.", 0.0, {}

    def clear_chat(self):
        """
        Reset the conversation context.

        Returns:
            bool: True if reset was successful
        """
        try:
            # Clear session history by resetting the agent's run history
            if hasattr(self.agent, "run_response"):
                self.agent.run_response = None
            return True
        except Exception as e:
            print(f"Error in clearing memory: {e}")
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
