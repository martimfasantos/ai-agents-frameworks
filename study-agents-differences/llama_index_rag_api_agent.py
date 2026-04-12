import asyncio
import tiktoken
import time

# Llama-Index imports (aligned with llama-index/ folder patterns)
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document
from llama_index.core.tools import FunctionTool, RetrieverTool
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.embeddings.openai import OpenAIEmbedding

# Prompt components
from prompts import role, goal, instructions, knowledge

from utils import get_tools_descriptions, parse_args, execute_agent

# Tools
from shared_functions import F1API, MetroAPI

# Load environment variables
from settings import settings


class LlamaIndexRAGandAPIAgent:
    def __init__(
        self,
        provider: str = "openai",
        memory: bool = True,
        verbose: bool = False,
        tokens: bool = False,
    ):
        """
        Initialize the Llama-Index RAG & API agent.
        Uses the new FunctionAgent from llama_index.core.agent.workflow (replacing
        the deprecated ReActAgent).
        """
        self.name = "Llama-Index RAG & API Agent"
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
                api_key=settings.openai_api_key.get_secret_value(),
                model=settings.openai_model_name,
                callback_manager=callback_manager,
            )
        else:
            self.model = HuggingFaceInferenceAPI(
                model_name=settings.open_source_model_name,
                callback_manager=callback_manager,
            )

        # Create tools
        self.tools = self._create_tools()

        # Create the FunctionAgent (new API, replacing ReActAgent)
        self.agent = FunctionAgent(
            name="llama_index_rag_api_agent",
            description="A RAG & API agent with knowledge base search, F1, and Metro tools.",
            llm=self.model,
            tools=self.tools,
            system_prompt="\n".join(
                [
                    role,
                    goal,
                    instructions,
                    knowledge,
                    "You have access to the knowledge base about the matches of the 2025 UEFA Champions League "
                    "and the provided tools to get information about F1 drivers, the state of the subway, and the times "
                    "of the next two subways in a station.",
                ]
            ),
        )

        # Extras:
        self.tools_descriptions = get_tools_descriptions(
            [(tool.metadata.name, tool.metadata.description) for tool in self.tools]
        )

    @staticmethod
    def load_documents(docs_path: str) -> list[Document]:
        """
        Load all docs from a folder and return them as a list of Document.
        Args:
            docs_path (str): A path to the directory containing the documents.
        Returns:
            List[Document]: A list of Document objects.
        """
        return SimpleDirectoryReader(docs_path).load_data()

    @staticmethod
    def create_vectorstore_index(documents: list[Document]) -> VectorStoreIndex:
        """
        Create a simple vectorstore using VectorStoreIndex
        """
        splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=50)
        nodes = splitter.get_nodes_from_documents(documents)

        return VectorStoreIndex(
            nodes,
            embed_model=(
                OpenAIEmbedding(
                    api_key=settings.openai_api_key.get_secret_value(),
                    model_name=settings.embeddings_model_name,
                )
                if settings.openai_api_key
                else f"local:{settings.local_embeddings_model_name}"
            ),
        )

    @staticmethod
    def create_rag_tool() -> RetrieverTool:
        """
        RAG tool that loads documents, creates a vectorstore, and returns a retriever tool.
        """
        docs = LlamaIndexRAGandAPIAgent.load_documents("knowledge_base/cl_matches")
        vector_index = LlamaIndexRAGandAPIAgent.create_vectorstore_index(docs)

        return RetrieverTool.from_defaults(
            retriever=vector_index.as_retriever(),
            name="RAG_tool",
            description="Search and retrieve information about the matches of the 2025 UEFA Champions League.",
        )

    def _create_tools(self):
        return [
            # RAG Tool
            self.create_rag_tool(),
            # API Tools
            FunctionTool.from_defaults(
                F1API.get_driver_info,
                name="get_driver_info",
                description="This tool is used to search for information about a driver",
            ),
            FunctionTool.from_defaults(
                MetroAPI.get_state_subway,
                name="get_state_subway",
                description="This tool is used to search for information about the state of the subway",
            ),
            FunctionTool.from_defaults(
                MetroAPI.get_times_next_two_subways_in_station,
                name="get_times_next_two_subways_in_station",
                description="This tool is used to search for information about the times (in seconds) of the next two subways in a station",
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
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            start = time.perf_counter()
            response = loop.run_until_complete(self.agent.run(message))
            end = time.perf_counter()
            exec_time = end - start

            loop.close()

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
            return True
        except Exception as e:
            print(f"Error clearing chat: {e}")
            return False


def main():
    """
    Example usage demonstrating the agent interface.
    """

    args = parse_args()

    agent = LlamaIndexRAGandAPIAgent(
        provider=args.provider,
        memory=not args.no_memory,
        verbose=args.verbose,
        tokens=args.mode in ["metrics", "metrics-loop"],
    )

    execute_agent(agent, args)


if __name__ == "__main__":
    main()
