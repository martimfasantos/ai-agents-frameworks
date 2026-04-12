from datetime import date
from tavily import TavilyClient
import json
import time

# Pydantic AI imports (aligned with pydantic-ai/ folder patterns)
from pydantic_ai import Agent as PydanticAgent
from pydantic_ai.tools import Tool

# Prompt components
from prompts import role, goal, instructions, knowledge

from utils import get_tools_descriptions, parse_args, execute_agent

# Load environment variables
from settings import settings

# Initialize Tavily client
tavily_client = TavilyClient(api_key=settings.tavily_api_key.get_secret_value())


class Agent:
    def __init__(
        self,
        provider: str = "openai",
        memory: bool = True,
        verbose: bool = False,
        tokens: bool = False,
    ):
        """
        Initialize the Pydantic AI agent.
        Uses string model names and run_sync() (aligned with pydantic-ai/ folder patterns).
        """
        self.name = "PydanticAI Agent"
        self.tokens = tokens

        # Determine the model name string (pydantic-ai uses plain string model names)
        if provider == "azure" and settings.azure_api_key:
            # Azure requires OpenAIModel with explicit base_url
            from pydantic_ai.models.openai import OpenAIModel

            self.model = OpenAIModel(
                settings.azure_deployment_name,
                base_url=f"{settings.azure_endpoint}/deployments/{settings.azure_deployment_name}",
                api_key=settings.azure_api_key.get_secret_value(),
            )
        elif provider == "openai" and settings.openai_api_key:
            # For OpenAI, use the string model name directly
            self.model = settings.openai_model_name
        else:
            self.model = settings.open_source_model_name

        # Create tools
        self.tools = self._create_tools()

        # Create the agent with instructions (not system_prompt)
        self.agent = PydanticAgent(
            model=self.model,
            tools=self.tools,
            instructions="\n".join(
                [
                    role,
                    goal,
                    instructions,
                    "You have access to two primary tools: date and web_search.",
                    knowledge,
                ]
            ),
            output_type=str,
        )

        # Conversation history
        self.messages = [] if memory else None
        self.memory = memory

        # Extras:
        self.tools_descriptions = get_tools_descriptions(
            [(tool.name, tool.description) for tool in self.tools]
        )

    def _create_tools(self):
        """
        Create and register tools for the agent.
        """

        def date_tool() -> str:
            """Get the current date"""
            today = date.today()
            return today.strftime("%B %d, %Y")

        def web_search_tool(query: str) -> str:
            """Search the web for information"""
            search_response = tavily_client.search(query)
            results = json.dumps(search_response.get("results", []))
            return results

        return [
            Tool(date_tool, name="date_tool", description="Gets the current date"),
            Tool(
                web_search_tool,
                name="web_search_tool",
                description="Searches the web for information",
            ),
        ]

    def chat(self, message):
        """
        Send a message and get a response.
        Uses run_sync() (aligned with pydantic-ai/ folder patterns).

        Args:
            message (str): User's input message

        Returns:
            tuple: (response_text, exec_time, tokens_dict)
        """
        try:
            start = time.perf_counter()
            result = self.agent.run_sync(
                message, message_history=self.messages if self.memory else None
            )
            end = time.perf_counter()
            exec_time = end - start

            # Maintain conversation history
            if self.memory and self.messages is not None:
                self.messages.extend(result.new_messages())

            if self.tokens:
                try:
                    usage = result.usage()
                    tokens = {
                        "total_embedding_token_count": 0,
                        "prompt_llm_token_count": getattr(usage, "request_tokens", 0)
                        or 0,
                        "completion_llm_token_count": getattr(
                            usage, "response_tokens", 0
                        )
                        or 0,
                        "total_llm_token_count": getattr(usage, "total_tokens", 0) or 0,
                    }
                except Exception:
                    tokens = {}
            else:
                tokens = {}

            # result.output is the new API (replacing result.data)
            return result.output, exec_time, tokens

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
            self.messages = []
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
