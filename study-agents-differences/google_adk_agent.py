from datetime import date
from tavily import TavilyClient
import json
import time
import asyncio

# Google ADK imports (aligned with google-adk/ folder patterns)
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import FunctionTool
from google.adk.models.lite_llm import LiteLlm
from google.genai import types

from settings import settings
from utils import get_tools_descriptions, parse_args, execute_agent

# Prompt components
from prompts import role, goal, instructions, knowledge


# --- Tool Definitions (plain functions for ADK) ---


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
        Initialize the Google ADK agent.
        """
        self.name = "Google ADK Agent"
        self.verbose = verbose
        self.tokens = tokens

        # Determine the model - Google ADK uses LiteLlm for non-Google models
        if provider == "azure" and settings.azure_api_key:
            self.model_name = LiteLlm(model=f"azure/{settings.azure_deployment_name}")
        elif provider == "openai" and settings.openai_api_key:
            self.model_name = LiteLlm(model=f"openai/{settings.openai_model_name}")
        else:
            self.model_name = "gemini-2.0-flash"

        # Create the ADK agent (using FunctionTool wrapping per framework patterns)
        self.adk_agent = LlmAgent(
            name="google_adk_agent",
            model=self.model_name,
            instruction="\n".join([role, goal, instructions, knowledge]),
            tools=[FunctionTool(func=date_tool), FunctionTool(func=web_search_tool)],
        )

        # Session management
        self.session_service = InMemorySessionService()
        self.runner = Runner(
            agent=self.adk_agent,
            app_name="study_comparison",
            session_service=self.session_service,
        )
        self.session_id = "session_1"
        self.user_id = "user_1"

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
            start = time.perf_counter()

            # Run the agent asynchronously
            async def _run():
                session = await self.session_service.create_session(
                    app_name="study_comparison",
                    user_id=self.user_id,
                )
                content = types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=message)],
                )
                final_response = ""
                prompt_tokens = 0
                completion_tokens = 0

                async for event in self.runner.run_async(
                    user_id=self.user_id,
                    session_id=session.id,
                    new_message=content,
                ):
                    if event.is_final_response():
                        for part in event.content.parts:
                            if part.text:
                                final_response += part.text
                    # Collect token usage from usage_metadata on events
                    if hasattr(event, "usage_metadata") and event.usage_metadata:
                        um = event.usage_metadata
                        prompt_tokens += getattr(um, "prompt_token_count", 0) or 0
                        completion_tokens += (
                            getattr(um, "candidates_token_count", 0) or 0
                        )

                return final_response, prompt_tokens, completion_tokens

            response_text, prompt_tok, completion_tok = asyncio.run(_run())

            end = time.perf_counter()
            exec_time = end - start

            if self.tokens:
                tokens = {
                    "total_embedding_token_count": 0,
                    "prompt_llm_token_count": prompt_tok,
                    "completion_llm_token_count": completion_tok,
                    "total_llm_token_count": prompt_tok + completion_tok,
                }
            else:
                tokens = {}

            return response_text, exec_time, tokens

        except Exception as e:
            print(f"Error in chat: {e}")
            return "Sorry, I encountered an error processing your request.", 0.0, {}

    def clear_chat(self):
        """Reset the conversation context."""
        self.session_id = f"session_{int(time.time())}"
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
