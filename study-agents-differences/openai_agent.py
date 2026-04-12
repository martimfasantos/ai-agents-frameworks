from datetime import date
from tavily import TavilyClient
import json
import time

# OpenAI imports
from openai import OpenAI, AzureOpenAI

from settings import settings
from utils import get_tools_descriptions, parse_args, execute_agent

# Prompt components
from prompts import role, goal, instructions, knowledge
from prompts import openai_completion_after_tool_call_prompt


class Agent:
    def __init__(
        self,
        provider: str = "openai",
        memory: bool = True,
        verbose: bool = False,
        tokens: bool = False,
    ):
        """
        Initialize the OpenAI agent.
        """
        self.name = "OpenAI Agent"
        self.provider = provider
        self.verbose = verbose
        self.tokens = tokens
        self.messages = []

        if provider == "azure" and settings.azure_api_key:
            self.model = AzureOpenAI(
                base_url=f"{settings.azure_endpoint}/deployments/{settings.azure_deployment_name}",
                api_version=settings.azure_api_version,
                api_key=settings.azure_api_key.get_secret_value(),
            )
            self.model_name = settings.azure_deployment_name
        elif provider == "openai" and settings.openai_api_key:
            self.model = OpenAI(
                api_key=settings.openai_api_key.get_secret_value(),
            )
            self.model_name = settings.openai_model_name
        else:
            self.model = OpenAI(
                api_key=settings.openai_api_key.get_secret_value(),
            )
            self.model_name = settings.openai_model_name

        # Create tools
        self.tools = self._create_tools()

        # Tool dispatch map
        self._tool_dispatch = {
            "date_tool": self.date_tool,
            "web_search_tool": self.web_search_tool,
        }

        # Create prompt
        self.prompt = self._create_prompt()

        # Create the Agent (completions API)
        self.agent = self.model.chat.completions

        # Tool descriptions for UI
        self.tools_descriptions = get_tools_descriptions(
            [
                ("date_tool", "Get the current date"),
                ("web_search_tool", "Search the web for information"),
            ]
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
        tavily_client = TavilyClient(api_key=settings.tavily_api_key.get_secret_value())
        search_response = tavily_client.search(query)
        results = json.dumps(search_response.get("results", []))
        return results

    def call_function(self, name, args):
        """Dispatch a tool call by name using the dispatch map."""
        func = self._tool_dispatch.get(name)
        if func is None:
            return f"Unknown tool: {name}"
        return func(**args) if args else func()

    def _create_tools(self):
        """
        Create tools for the agent.

        Returns:
            List of tools
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "date_tool",
                    "description": "Useful for getting the current date.",
                },
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "web_search_tool",
                    "description": "Useful for searching the web for information.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Provide a query to search the web for information.",
                            }
                        },
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            },
        ]

    def _create_prompt(self):
        """
        Create a prompt for the agent.

        Returns:
            str: Prompt for the agent
        """
        return {
            "role": "system",
            "content": f"{role}\n{goal}\n{instructions}\n{knowledge}",
        }

    def chat(self, message):
        """
        Send a message and get a completion.

        Args:
            message (str): User's input message

        Returns:
            tuple: (response_text, exec_time, tokens_dict)
        """
        try:
            start = time.perf_counter()
            messages = [self.prompt, {"role": "user", "content": message}]

            # Send prompt + user_message to the agent
            completion = self.agent.create(
                model=self.model_name, messages=messages, tools=self.tools
            )

            tokens = {
                "total_embedding_token_count": 0,
                "prompt_llm_token_count": 0,
                "completion_llm_token_count": 0,
                "total_llm_token_count": 0,
            }

            if self.tokens and completion.usage:
                tokens["prompt_llm_token_count"] = completion.usage.prompt_tokens or 0
                tokens["completion_llm_token_count"] = (
                    completion.usage.completion_tokens or 0
                )
                tokens["total_llm_token_count"] = (
                    tokens["prompt_llm_token_count"]
                    + tokens["completion_llm_token_count"]
                )

            response_message = completion.choices[0].message
            tool_calls = response_message.tool_calls

            # Agent returns tool calls, we need to process them and manually call the functions
            if tool_calls:
                messages.append(response_message)

                for tool_call in tool_calls:
                    name = tool_call.function.name
                    args = json.loads(tool_call.function.arguments)

                    if self.verbose:
                        print(f"Tool call name: {name}")
                        print(f"Tool call args: {args}")

                    # Call the chosen tool using dispatch map
                    tool_result = self.call_function(name, args)

                    # Append the tool result to the messages
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": str(tool_result),
                        }
                    )

            # Append a prompt to the messages to get the final completion
            messages.append(
                {"role": "system", "content": openai_completion_after_tool_call_prompt}
            )

            # Send the messages with the tool results back to the agent to get the final completion
            completion2 = self.agent.create(
                model=self.model_name,
                messages=messages,
            )

            end = time.perf_counter()
            exec_time = end - start

            # Add the tokens from the second completion
            if self.tokens and completion2.usage:
                tokens["prompt_llm_token_count"] += completion2.usage.prompt_tokens or 0
                tokens["completion_llm_token_count"] += (
                    completion2.usage.completion_tokens or 0
                )
                tokens["total_llm_token_count"] += (
                    completion2.usage.prompt_tokens or 0
                ) + (completion2.usage.completion_tokens or 0)

            return completion2.choices[0].message.content, exec_time, tokens

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
            self.messages = []
            return True
        except Exception as e:
            print(f"Error in clearing memory: {e}")
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
