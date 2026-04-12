from datetime import date
from tavily import TavilyClient
import json
import time

# CrewAI imports
from crewai import Agent as CrewAgent, Task, Crew
from crewai.tools import tool as crewai_tool

from settings import settings
from utils import get_tools_descriptions, parse_args, execute_agent

# Prompt components
from prompts import role, goal, instructions, knowledge


# --- Tool Definitions ---


@crewai_tool("date_tool")
def date_tool() -> str:
    """Gets the current date."""
    today = date.today()
    return today.strftime("%B %d, %Y")


@crewai_tool("web_search_tool")
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
        Initialize the CrewAI agent.
        """
        self.name = "CrewAI Agent"
        self.verbose = verbose
        self.tokens = tokens
        self.memory = memory

        # Determine the LLM model string
        if provider == "azure" and settings.azure_api_key:
            self.llm = f"azure/{settings.azure_deployment_name}"
        elif provider == "openai" and settings.openai_api_key:
            self.llm = settings.openai_model_name
        else:
            self.llm = settings.open_source_model_name

        # Create tools
        self.tools = [date_tool, web_search_tool]

        # Create the CrewAI agent
        self.crew_agent = CrewAgent(
            role="Information Retrieval Specialist",
            goal=goal,
            backstory="\n".join([role, instructions, knowledge]),
            tools=self.tools,
            llm=self.llm,
            verbose=verbose,
            memory=memory,
        )

        # Extras:
        self.tools_descriptions = get_tools_descriptions(
            [(t.name, t.description) for t in self.tools]
        )

    def chat(self, message):
        """
        Send a message and get a response.
        """
        try:
            task = Task(
                description=message,
                expected_output="A concise and accurate response to the user's query.",
                agent=self.crew_agent,
            )

            crew = Crew(
                agents=[self.crew_agent],
                tasks=[task],
                verbose=self.verbose,
                memory=self.memory,
            )

            start = time.perf_counter()
            result = crew.kickoff()
            end = time.perf_counter()
            exec_time = end - start

            if self.tokens:
                usage = result.token_usage if hasattr(result, "token_usage") else None
                tokens = {
                    "total_embedding_token_count": 0,
                    "prompt_llm_token_count": getattr(usage, "prompt_tokens", 0)
                    if usage
                    else 0,
                    "completion_llm_token_count": getattr(usage, "completion_tokens", 0)
                    if usage
                    else 0,
                    "total_llm_token_count": getattr(usage, "total_tokens", 0)
                    if usage
                    else 0,
                }
            else:
                tokens = {}

            return str(result), exec_time, tokens

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
