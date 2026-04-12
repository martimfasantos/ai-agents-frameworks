import os
import time
from datetime import date

# CrewAI imports
from crewai import Agent as CrewAgent, Task, Crew
from crewai.tools import tool as crewai_tool
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource

from settings import settings
from utils import get_tools_descriptions, parse_args, execute_agent

# Prompt components
from prompts import knowledge, role, goal, instructions

# Tools
from shared_functions import F1API, MetroAPI


# --- Tool Definitions ---


@crewai_tool("date_tool")
def get_date() -> str:
    """Gets the current date."""
    today = date.today()
    return today.strftime("%B %d, %Y")


@crewai_tool("get_driver_info")
def get_driver_info(driver_number: int, session_key: int = 9158) -> str:
    """Useful function to get F1 driver information."""
    return F1API.get_driver_info(driver_number, session_key)


@crewai_tool("get_state_subway")
def get_state_subway() -> str:
    """Useful function to get the information about the state of the subway."""
    return MetroAPI.get_state_subway()


@crewai_tool("get_times_next_two_subways_in_station")
def get_times_next_two_subways_in_station(station: str) -> str:
    """Useful to get the time (in seconds) of the next two subways in a station."""
    return MetroAPI.get_times_next_two_subways_in_station(station)


class CrewAIRAGandAPIAgent:
    def __init__(
        self,
        provider: str = "openai",
        memory: bool = True,
        verbose: bool = False,
        tokens: bool = False,
    ):
        """
        Initialize the CrewAI RAG & API agent.
        """
        self.name = "CrewAI RAG & API Agent"
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

        # Load knowledge base documents for RAG
        self.knowledge_sources = self._load_knowledge_sources()

        # Create tools
        self.tools = [
            get_date,
            get_driver_info,
            get_state_subway,
            get_times_next_two_subways_in_station,
        ]

        # Create the CrewAI agent with knowledge sources (RAG)
        self.crew_agent = CrewAgent(
            role="Information Retrieval Specialist",
            goal=goal,
            backstory="\n".join(
                [
                    role,
                    instructions,
                    knowledge,
                    "You have access to the knowledge base about the matches of the 2025 UEFA Champions League "
                    "and the provided tools to get information about F1 drivers, the state of the subway, and the times "
                    "of the next two subways in a station.",
                ]
            ),
            tools=self.tools,
            llm=self.llm,
            verbose=verbose,
            memory=memory,
            knowledge_sources=self.knowledge_sources,
        )

        # Extras:
        self.tools_descriptions = get_tools_descriptions(
            [("RAG_tool", "Knowledge base for the 2025 UEFA Champions League matches")]
            + [(t.name, t.description) for t in self.tools]
        )

    @staticmethod
    def _load_knowledge_sources():
        """Load knowledge base documents as StringKnowledgeSources."""
        docs_path = "knowledge_base/cl_matches"
        sources = []
        for file_name in os.listdir(docs_path):
            if file_name.endswith(".md"):
                file_path = os.path.join(docs_path, file_name)
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()
                    sources.append(
                        StringKnowledgeSource(
                            content=content,
                            metadata={"file_name": file_name},
                        )
                    )
        return sources

    def chat(self, message):
        """Send a message and get a response."""
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

    agent = CrewAIRAGandAPIAgent(
        provider=args.provider,
        memory=not args.no_memory,
        verbose=args.verbose,
        tokens=args.mode in ["metrics", "metrics-loop"],
    )

    execute_agent(agent, args)


if __name__ == "__main__":
    main()
