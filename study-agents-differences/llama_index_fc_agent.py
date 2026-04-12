import asyncio
import tiktoken
from datetime import date
from tavily import TavilyClient
import json
import time

# Llama-Index imports (aligned with llama-index/ folder patterns)
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.core.tools import FunctionTool
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler

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
        Initialize the Llama-Index Function Calling agent.
        Uses the new FunctionAgent from llama_index.core.agent.workflow (replacing
        the deprecated FunctionCallingAgentWorker).
        """
        self.name = "Llama-Index Function Calling Agent"
        self.tokens = tokens

        # Initialize token counter per instance
        self.token_counter = TokenCountingHandler(
            tokenizer=tiktoken.encoding_for_model("gpt-4").encode
        )
        callback_manager = CallbackManager([self.token_counter]) if tokens else None

        # Initialize the language model
        if provider == "azure" and settings.azure_api_key:
            self.model = AzureOpenAI(
                engine=settings.azure_deployment_name,
                api_base=f"{settings.azure_endpoint}/deployments/{settings.azure_deployment_name}",
                api_version=settings.azure_api_version,
                api_key=settings.azure_api_key.get_secret_value(),
                callback_manager=callback_manager,
            )
        elif provider == "openai" and settings.openai_api_key:
            self.model = OpenAI(
                model=settings.openai_model_name,
                api_key=settings.openai_api_key.get_secret_value(),
                callback_manager=callback_manager,
            )
        else:
            self.model = HuggingFaceInferenceAPI(
                model_name=settings.open_source_model_name,
                callback_manager=callback_manager,
            )

        # Create tools
        self.tools = self._create_tools()

        # Create the FunctionAgent (new API, replacing FunctionCallingAgentWorker)
        self.agent = FunctionAgent(
            name="llama_index_fc_agent",
            description="A function-calling agent with web search and date tools.",
            llm=self.model,
            tools=self.tools,
            system_prompt="\n".join(
                [
                    role,
                    goal,
                    instructions,
                    "You have access to two primary tools: date_tool and web_search_tool.",
                    knowledge,
                ]
            ),
        )

        # Extras:
        self.tools_descriptions = get_tools_descriptions(
            [(tool.metadata.name, tool.metadata.description) for tool in self.tools]
        )

    @staticmethod
    def date_tool():
        """
        Function to get the current date.
        """
        today = date.today()
        return today.strftime("%B %d, %Y")

    @staticmethod
    def web_search_tool(query: str):
        """
        This function searches the web for the given query and returns the results.
        """
        search_response = tavily_client.search(query)
        results = json.dumps(search_response.get("results", []))
        return results

    def _create_tools(self):
        """
        Create tools for the agent.

        Returns:
            List of FunctionTool instances
        """
        return [
            FunctionTool.from_defaults(
                fn=self.date_tool,
                name="date_tool",
                description="Useful for getting the current date",
            ),
            FunctionTool.from_defaults(
                fn=self.web_search_tool,
                name="web_search_tool",
                description="Useful for searching the web for information",
            ),
        ]

    def chat(self, message):
        """
        Send a message and get a response.
        Uses async agent.run() internally via an event loop.

        Args:
            message (str): User's input message

        Returns:
            tuple: (response_text, exec_time, tokens_dict)
        """
        try:
            start = time.perf_counter()

            async def _run():
                handler = self.agent.run(message)
                return await handler

            response = asyncio.run(_run())
            end = time.perf_counter()
            exec_time = end - start

            if self.tokens:
                tokens = {
                    "total_embedding_token_count": self.token_counter.total_embedding_token_count,
                    "prompt_llm_token_count": self.token_counter.prompt_llm_token_count,
                    "completion_llm_token_count": self.token_counter.completion_llm_token_count,
                    "total_llm_token_count": self.token_counter.total_llm_token_count,
                }
                self.token_counter.reset_counts()
            else:
                tokens = {}

            return str(response), exec_time, tokens

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
            # FunctionAgent doesn't maintain persistent state between runs
            # by default, so clearing is a no-op.
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
