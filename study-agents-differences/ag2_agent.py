from datetime import date
from tavily import TavilyClient
import json
import time

# AG2 imports (aligned with ag2/ folder patterns — import from autogen)
from autogen import ConversableAgent, LLMConfig

from settings import settings
from utils import get_tools_descriptions, parse_args, execute_agent

# Prompt components
from prompts import role, goal, instructions, knowledge


# --- Tool Definitions ---


def date_tool() -> str:
    """Gets the current date."""
    today = date.today()
    return today.strftime("%B %d, %Y")


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
        Initialize the AG2 agent.
        Uses dict-based LLMConfig and imports from autogen (aligned with ag2/ folder patterns).
        """
        self.name = "AG2 Agent"
        self.verbose = verbose
        self.tokens = tokens
        self.memory = memory

        # Configure the LLM (LLMConfig with keyword args per framework 0.8+ API)
        if provider == "azure" and settings.azure_api_key:
            self.llm_config = LLMConfig(
                {
                    "model": settings.azure_deployment_name,
                    "api_type": "azure",
                    "base_url": f"{settings.azure_endpoint}/deployments/{settings.azure_deployment_name}",
                    "api_key": settings.azure_api_key.get_secret_value(),
                    "api_version": settings.azure_api_version,
                },
            )
        elif provider == "openai" and settings.openai_api_key:
            self.llm_config = LLMConfig(
                {
                    "model": settings.openai_model_name,
                    "api_key": settings.openai_api_key.get_secret_value(),
                },
            )
        else:
            self.llm_config = LLMConfig(
                {"model": settings.open_source_model_name},
            )

        # Create the AG2 agent (llm_config= param per framework patterns)
        self.agent = ConversableAgent(
            name="ag2_agent",
            system_message="\n".join([role, goal, instructions, knowledge]),
            human_input_mode="NEVER",
            llm_config=self.llm_config,
        )

        # Register tools via register_for_llm / register_for_execution (AG2 0.8+ API)
        self.agent.register_for_llm(
            name="date_tool", description="Gets the current date"
        )(date_tool)
        self.agent.register_for_execution(name="date_tool")(date_tool)
        self.agent.register_for_llm(
            name="web_search_tool",
            description="Searches the web for information",
        )(web_search_tool)
        self.agent.register_for_execution(name="web_search_tool")(web_search_tool)

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

        Args:
            message (str): User's input message

        Returns:
            tuple: (response_text, exec_time, tokens_dict)
        """
        try:
            start = time.perf_counter()

            result = self.agent.run(message=message, max_turns=3, user_input=False)

            # Extract the final response from events
            # AG2 0.8+ uses an event-based RunResponse; iterate events to find
            # the first non-empty TextEvent from the agent (not the user).
            # Note: events is a generator — iterating it drives the actual API calls.
            response_text = ""
            for event in result.events:
                event_type = type(event).__name__
                if event_type == "TextEvent":
                    inner = event.content
                    sender = getattr(inner, "sender", "")
                    content = getattr(inner, "content", "")
                    if content and sender != "user":
                        response_text = str(content)
                        break  # Take the first substantive agent response
            # Fallback to summary if no text event found
            if not response_text:
                response_text = (
                    str(result.summary)
                    if hasattr(result, "summary") and result.summary
                    else str(result)
                )

            end = time.perf_counter()
            exec_time = end - start

            if self.tokens:
                # AG2 tracks usage on the ConversableAgent's client via
                # get_total_usage() / get_actual_usage(), which return a dict
                # like {"model": {"prompt_tokens": N, "completion_tokens": N, ...}}
                try:
                    usage_summary = self.agent.get_total_usage() or {}
                    prompt_tok = 0
                    completion_tok = 0
                    total_tok = 0
                    for _model, model_usage in usage_summary.items():
                        if isinstance(model_usage, dict):
                            prompt_tok += model_usage.get("prompt_tokens", 0)
                            completion_tok += model_usage.get("completion_tokens", 0)
                            total_tok += model_usage.get("total_tokens", 0)
                    tokens = {
                        "total_embedding_token_count": 0,
                        "prompt_llm_token_count": prompt_tok,
                        "completion_llm_token_count": completion_tok,
                        "total_llm_token_count": total_tok
                        if total_tok
                        else prompt_tok + completion_tok,
                    }
                except Exception:
                    tokens = {}
            else:
                tokens = {}

            return response_text, exec_time, tokens

        except Exception as e:
            print(f"Error in chat: {e}")
            return "Sorry, I encountered an error processing your request.", 0.0, {}

    def clear_chat(self):
        """Reset the conversation context."""
        try:
            self.agent.clear_history()
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
