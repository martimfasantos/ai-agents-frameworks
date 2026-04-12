from datetime import date
from tavily import TavilyClient
import json
import time

# LangChain imports (aligned with langchain/ folder patterns)
from langchain.agents import create_agent
from langchain.tools import tool as langchain_tool
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

from settings import settings
from utils import get_tools_descriptions, parse_args, execute_agent

# Prompt components
from prompts import role, goal, instructions, knowledge


# --- Tool Definitions ---


@langchain_tool
def date_tool() -> str:
    """Gets the current date."""
    today = date.today()
    return today.strftime("%B %d, %Y")


@langchain_tool
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
        Initialize the LangChain agent (v1.0+ API with create_agent).
        """
        self.name = "LangChain Agent"
        self.verbose = verbose
        self.tokens = tokens
        self.memory = memory

        # Initialize the language model
        if provider == "azure" and settings.azure_api_key:
            self.model = AzureChatOpenAI(
                base_url=f"{settings.azure_endpoint}/deployments/{settings.azure_deployment_name}",
                api_version=settings.azure_api_version,
                api_key=settings.azure_api_key.get_secret_value(),
            )
        elif provider == "openai" and settings.openai_api_key:
            self.model = ChatOpenAI(
                api_key=settings.openai_api_key.get_secret_value(),
                model=settings.openai_model_name,
            )
        else:
            self.model = ChatOpenAI(model=settings.open_source_model_name)

        # Create tools
        self.tools = [date_tool, web_search_tool]

        # Create memory
        self.checkpointer = InMemorySaver() if memory else None
        self.thread_id = 1

        # Create the agent using LangChain v1.0+ API
        self.agent = create_agent(
            model=self.model,
            tools=self.tools,
            system_prompt="\n".join([role, goal, instructions, knowledge]),
            checkpointer=self.checkpointer,
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
            config = {"configurable": {"thread_id": str(self.thread_id)}}
            inputs = {"messages": [{"role": "user", "content": message}]}

            start = time.perf_counter()
            result = self.agent.invoke(inputs, config=config)
            end = time.perf_counter()
            exec_time = end - start

            # Extract the final response
            messages = result.get("messages", [])
            response_text = messages[-1].content if messages else ""

            if self.tokens:
                tokens = {
                    "total_embedding_token_count": 0,
                    "prompt_llm_token_count": 0,
                    "completion_llm_token_count": 0,
                    "total_llm_token_count": 0,
                }
                for msg in messages:
                    if hasattr(msg, "response_metadata") and msg.response_metadata:
                        token_usage = msg.response_metadata.get("token_usage", {})
                        tokens["prompt_llm_token_count"] += token_usage.get(
                            "prompt_tokens", 0
                        )
                        tokens["completion_llm_token_count"] += token_usage.get(
                            "completion_tokens", 0
                        )
                        tokens["total_llm_token_count"] += token_usage.get(
                            "total_tokens", 0
                        )
            else:
                tokens = {}

            return response_text, exec_time, tokens

        except Exception as e:
            print(f"Error in chat: {e}")
            return "Sorry, I encountered an error processing your request.", 0.0, {}

    def clear_chat(self):
        """Reset the conversation context."""
        try:
            self.thread_id += 1
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
