from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.duckduckgo import DuckDuckGoTools
from settings import settings

# ---------------------------------------------------------
# In this example, we explore Agno's Agent class with the following features:
# - Using tools
# - Pretty-printing responses
# ---------------------------------------------------------

agent = Agent(
    model=OpenAIChat(
        id=settings.OPENAI_MODEL_NAME,
        api_key=settings.OPENAI_API_KEY.get_secret_value(),
    ),
    tools=[DuckDuckGoTools()],
    description="You are an enthusiastic news reporter with a flair for storytelling!",
    show_tool_calls=True,
    markdown=True,
)

agent.print_response("Tell me about a breaking news story from New York.", stream=False)

