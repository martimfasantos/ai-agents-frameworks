import os
import time
from datetime import date

# LangChain imports (aligned with langchain/ folder patterns)
from langchain.agents import create_agent
from langchain.tools import tool as langchain_tool
from langchain_openai import (
    AzureChatOpenAI,
    ChatOpenAI,
    OpenAIEmbeddings,
    AzureOpenAIEmbeddings,
)
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langgraph.checkpoint.memory import InMemorySaver

from settings import settings
from utils import get_tools_descriptions, parse_args, execute_agent

# Prompt components
from prompts import knowledge, role, goal, instructions

# Tools
from shared_functions import F1API, MetroAPI


class LangChainRAGandAPIAgent:
    def __init__(
        self,
        provider: str = "openai",
        memory: bool = True,
        verbose: bool = False,
        tokens: bool = False,
    ):
        """
        Initialize the LangChain RAG & API agent.
        """
        self.name = "LangChain RAG & API Agent"
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

        # Create tools (including RAG)
        self.tools = self._create_tools()

        # Create memory
        self.checkpointer = InMemorySaver() if memory else None
        self.thread_id = 1

        # Create the agent
        self.agent = create_agent(
            model=self.model,
            tools=self.tools,
            system_prompt="\n".join(
                [
                    knowledge,
                    role,
                    goal,
                    instructions,
                    "You have access to the knowledge base about the matches of the 2025 UEFA Champions League "
                    "and the provided tools to get information about F1 drivers, the state of the subway, and the times "
                    "of the next two subways in a station.",
                ]
            ),
            checkpointer=self.checkpointer,
        )

        # Extras:
        self.tools_descriptions = get_tools_descriptions(
            [(t.name, t.description) for t in self.tools]
        )

    @staticmethod
    def _create_vectorstore():
        """Load documents and create a vectorstore for RAG."""
        docs = DirectoryLoader(
            "knowledge_base/cl_matches/", glob="*.md", show_progress=True
        ).load()

        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=(
                AzureOpenAIEmbeddings(
                    base_url=f"{settings.azure_endpoint}/deployments/{settings.embeddings_model_name}",
                    api_version=settings.embeddings_api_version,
                    api_key=settings.azure_api_key.get_secret_value(),
                )
                if settings.azure_api_key
                else FastEmbedEmbeddings()
            ),
            collection_name="langchain-local-rag",
        )
        return vectorstore

    def _create_tools(self):
        """Create tools including RAG retriever tool."""

        # Create the vectorstore and retriever
        vectorstore = self._create_vectorstore()
        retriever = vectorstore.as_retriever()

        @langchain_tool
        def rag_tool(query: str) -> str:
            """Search and retrieve information from the knowledge base about the matches of the 2025 UEFA Champions League."""
            docs = retriever.invoke(query)
            return "\n\n".join([doc.page_content for doc in docs])

        @langchain_tool
        def get_driver_info(driver_number: int, session_key: int = 9158) -> str:
            """Useful function to get F1 driver information."""
            return F1API.get_driver_info(driver_number, session_key)

        @langchain_tool
        def get_state_subway() -> str:
            """Useful function to get the information about the state of the subway."""
            return MetroAPI.get_state_subway()

        @langchain_tool
        def get_times_next_two_subways_in_station(station: str) -> str:
            """Useful to get the time (in seconds) of the next two subways in a station."""
            return MetroAPI.get_times_next_two_subways_in_station(station)

        return [
            rag_tool,
            get_driver_info,
            get_state_subway,
            get_times_next_two_subways_in_station,
        ]

    def chat(self, message):
        """Send a message and get a response."""
        try:
            config = {"configurable": {"thread_id": str(self.thread_id)}}
            inputs = {"messages": [{"role": "user", "content": message}]}

            start = time.perf_counter()
            result = self.agent.invoke(inputs, config=config)
            end = time.perf_counter()
            exec_time = end - start

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
        self.thread_id += 1
        return True


def main():
    args = parse_args()

    agent = LangChainRAGandAPIAgent(
        provider=args.provider,
        memory=not args.no_memory,
        verbose=args.verbose,
        tokens=args.mode in ["metrics", "metrics-loop"],
    )

    execute_agent(agent, args)


if __name__ == "__main__":
    main()
