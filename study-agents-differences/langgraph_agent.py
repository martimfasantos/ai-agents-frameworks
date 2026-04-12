from datetime import date
from tavily import TavilyClient
import json
import time

# LangGraph and LangChain imports (aligned with langgraph/ folder patterns)
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_huggingface import ChatHuggingFace
from langchain_core.tools import tool as langchain_tool
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

from utils import get_tools_descriptions, parse_args, execute_agent

# Prompt components
from prompts import role, goal, instructions, knowledge

# Load environment variables
from settings import settings

# Initialize Tavily client
tavily_client = TavilyClient(api_key=settings.tavily_api_key.get_secret_value())


# --- Tool Definitions (using @tool decorator, per framework patterns) ---


@langchain_tool
def date_tool() -> str:
    """Gets the current date."""
    today = date.today()
    return today.strftime("%B %d, %Y")


@langchain_tool
def web_search_tool(query: str) -> str:
    """Searches the web for information using Tavily."""
    search_response = tavily_client.search(query)
    results = json.dumps(search_response.get("results", []))
    return results


tools = [date_tool, web_search_tool]


class Agent:
    def __init__(
        self,
        provider: str = "openai",
        memory: bool = True,
        verbose: bool = False,
        tokens: bool = False,
    ):
        """
        Initialize the LangGraph agent using StateGraph with ToolNode and tools_condition
        (aligned with langgraph/ folder patterns, replacing create_react_agent).
        """
        self.name = "LangGraph Agent"

        # Create memory
        if memory:
            self.memory = MemorySaver()
        else:
            self.memory = None
        # Memory will be checkpointed per thread. We will start with thread id 1.
        self.thread_id = 1

        self.tokens = tokens

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
        self.llm_with_tools = self.model.bind_tools(tools)

        # System prompt
        self.system_prompt = "\n".join([role, goal, instructions, knowledge])

        # Build the ReAct agent graph using StateGraph (per framework patterns)
        self.graph = self._build_graph()

        # Extras:
        self.tools_descriptions = get_tools_descriptions(
            [(t.name, t.description) for t in tools]
        )

    def _build_graph(self):
        """
        Build the StateGraph with chatbot node, ToolNode, and tools_condition.
        This replaces the prebuilt create_react_agent.
        """
        system_prompt = self.system_prompt

        def chatbot(state: MessagesState):
            """Call the LLM — it may request tool calls or produce a final answer."""
            # Prepend system message if it's the start of conversation
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
        builder.add_node("tools", ToolNode(tools))

        builder.add_edge(START, "chatbot")
        builder.add_conditional_edges(
            "chatbot", tools_condition
        )  # Routes to "tools" or END
        builder.add_edge("tools", "chatbot")  # After tool execution, go back to chatbot

        return builder.compile(checkpointer=self.memory)

    def _inc_thread_id(self):
        """
        Simply increments the thread id and returns the new id.
        """
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
            # Prepare input
            inputs = {"messages": [("user", message)]}
            config = {"configurable": {"thread_id": str(self.thread_id)}}

            # Stream the graph updates and collect the final response
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
                    for msg in event["messages"]:  # last event contains all messages
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

        Returns:
            bool: True if reset was successful
        """
        try:
            self._inc_thread_id()  # Incrementing the thread ID basically resets the memory
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
