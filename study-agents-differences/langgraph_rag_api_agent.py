import time

# LangGraph and LangChain imports (aligned with langgraph/ folder patterns)
from langchain.tools.retriever import create_retriever_tool
from langchain_openai import (
    AzureChatOpenAI,
    ChatOpenAI,
    AzureOpenAIEmbeddings,
)
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_huggingface import ChatHuggingFace

from utils import get_tools_descriptions, parse_args, execute_agent

# Prompt components
from prompts import knowledge, role, goal, instructions

# Tools
from shared_functions import F1API, MetroAPI

# Load environment variables
from settings import settings


class LangGraphRAGandAPIAgent:
    def __init__(
        self,
        provider: str = "openai",
        memory: bool = True,
        verbose: bool = False,
        tokens: bool = False,
    ):
        """
        Initialize the LangGraph RAG & API agent using StateGraph with ToolNode
        and tools_condition (aligned with langgraph/ folder patterns).
        """
        self.name = "LangGraph RAG & API Agent"

        # Create tools
        self.tools = self._create_tools()

        # Create memory
        if memory:
            self.memory = MemorySaver()
        else:
            self.memory = None
        self.thread_id = 1

        self.tokens = tokens

        # System prompt
        self.system_prompt = "\n".join([knowledge, role, goal, instructions])

        # Initialize the language model
        self.model = (
            AzureChatOpenAI(
                base_url=f"{settings.azure_endpoint}/deployments/{settings.azure_deployment_name}",
                api_version=settings.azure_api_version,
                api_key=settings.azure_api_key.get_secret_value(),
            )
            if provider == "azure" and settings.azure_api_key
            else ChatOpenAI(
                api_key=settings.openai_api_key.get_secret_value(),
                model=settings.openai_model_name,
            )
            if provider == "openai" and settings.openai_api_key
            else ChatHuggingFace(model=settings.open_source_model_name)
        )

        # Bind tools to the LLM (per framework patterns)
        self.llm_with_tools = self.model.bind_tools(self.tools)

        # Build the ReAct agent graph using StateGraph (per framework patterns)
        self.graph = self._build_graph()

        # Extras:
        self.tools_descriptions = get_tools_descriptions(
            [(t.name, t.description) for t in self.tools]
        )

    def _build_graph(self):
        """
        Build the StateGraph with chatbot node, ToolNode, and tools_condition.
        """
        system_prompt = self.system_prompt
        agent_tools = self.tools

        def chatbot(state: MessagesState):
            """Call the LLM — it may request tool calls or produce a final answer."""
            messages = state["messages"]
            if (
                not messages
                or not hasattr(messages[0], "type")
                or messages[0].type != "system"
            ):
                messages = [SystemMessage(content=system_prompt)] + messages
            return {"messages": [self.llm_with_tools.invoke(messages)]}

        builder = StateGraph(MessagesState)

        builder.add_node("chatbot", chatbot)
        builder.add_node("tools", ToolNode(agent_tools))

        builder.add_edge(START, "chatbot")
        builder.add_conditional_edges("chatbot", tools_condition)
        builder.add_edge("tools", "chatbot")

        return builder.compile(checkpointer=self.memory)

    @staticmethod
    def load_documents(url: str) -> list[Document]:
        """
        Load all docs from a folder and return them as a list of Document.
        """
        return DirectoryLoader(url, glob="*.md", show_progress=True).load()

    @staticmethod
    def create_vectorstore(documents: list[Document]) -> Chroma:
        """
        Create a simple vectorstore using the in memory Chroma.
        """
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=(
                AzureOpenAIEmbeddings(
                    base_url=f"{settings.azure_endpoint}/deployments/{settings.embeddings_model_name}",
                    api_version=settings.embeddings_api_version,
                    api_key=settings.azure_api_key.get_secret_value(),
                )
                if settings.azure_api_key
                else FastEmbedEmbeddings()
            ),
            collection_name="local-rag",
        )
        return vectorstore

    @staticmethod
    def create_rag_tool():
        """
        RAG tool that loads documents, creates a vectorstore, and returns a retriever tool.
        """
        docs = LangGraphRAGandAPIAgent.load_documents("knowledge_base/cl_matches/")
        vectorstore = LangGraphRAGandAPIAgent.create_vectorstore(docs)

        return create_retriever_tool(
            vectorstore.as_retriever(),
            name="RAG_tool",
            description="Search and retrieve information from the knowledge base about the matches of the 2025 UEFA Champions League.",
        )

    @staticmethod
    @tool
    def get_driver_info(driver_number: int, session_key: int = 9158) -> str:
        """Useful function to get F1 drivers information."""
        return F1API.get_driver_info(driver_number, session_key)

    @staticmethod
    @tool
    def get_state_subway() -> str:
        """Useful function to get state subway information."""
        return MetroAPI.get_state_subway()

    @staticmethod
    @tool
    def get_times_next_two_subways_in_station(station: str) -> str:
        """Useful to get the time (in seconds) of the next two subways in a station."""
        return MetroAPI.get_times_next_two_subways_in_station(station)

    def _create_tools(self):
        """
        Create tools for the agent.
        """
        return [
            self.create_rag_tool(),
            self.get_driver_info,
            self.get_state_subway,
            self.get_times_next_two_subways_in_station,
        ]

    def _inc_thread_id(self):
        """Simply increments the thread id and returns the new id."""
        new_thread_id = self.thread_id + 1
        self.thread_id = new_thread_id
        return new_thread_id

    def chat(self, message):
        """
        Send a message and get a response.

        Args:
            message (str): User's input message

        Returns:
            tuple: (response_text, exec_time, tokens_dict)
        """
        try:
            inputs = {"messages": [("user", message)]}
            config = {"configurable": {"thread_id": str(self.thread_id)}}

            full_response = ""
            start = time.perf_counter()
            for event in self.graph.stream(inputs, config=config, stream_mode="values"):
                if event and "messages" in event:
                    last_message = event["messages"][-1]
                    if hasattr(last_message, "content"):
                        full_response = last_message.content
            end = time.perf_counter()
            exec_time = end - start

            if self.tokens:
                tokens = {
                    "total_embedding_token_count": 0,
                    "prompt_llm_token_count": 0,
                    "completion_llm_token_count": 0,
                    "total_llm_token_count": 0,
                }
                try:
                    for msg in event["messages"]:
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
                except Exception:
                    pass
            else:
                tokens = {}

            return full_response, exec_time, tokens

        except Exception as e:
            print(f"Error in chat: {e}")
            return "Sorry, I encountered an error processing your request.", 0.0, {}

    def clear_chat(self):
        """
        Reset the conversation context.
        """
        try:
            self._inc_thread_id()
            return True
        except Exception as e:
            print(f"Error clearing chat: {e}")
            return False


def main():
    """
    Example usage demonstrating the agent interface.
    """

    args = parse_args()

    agent = LangGraphRAGandAPIAgent(
        provider=args.provider,
        memory=not args.no_memory,
        verbose=args.verbose,
        tokens=args.mode in ["metrics", "metrics-loop"],
    )

    execute_agent(agent, args)


if __name__ == "__main__":
    main()
