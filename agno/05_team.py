from dotenv import load_dotenv

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.team import Team
from agno.team.team import TeamMode
from agno.utils.pprint import pprint_run_response

from settings import settings

load_dotenv()

"""
-------------------------------------------------------
In this example, we explore Agno with the following features:
- Creating a Team of specialized agents
- Using TeamMode.coordinate for collaborative task execution
- Model inheritance from team to members
- Agent role specialization with descriptions and instructions

Teams let you orchestrate multiple agents that each have a
specific role. In coordinate mode, a team leader delegates
sub-tasks to the most suitable member agents, combines
their outputs, and returns a unified response.

For more details, visit:
https://docs.agno.com/teams/introduction
-------------------------------------------------------
"""

# --- 1. Create specialized agents ---
researcher = Agent(
    name="Researcher",
    description="An expert at finding and summarizing factual information.",
    instructions=[
        "You research topics thoroughly and provide factual summaries.",
        "Always cite specific details like dates, names, and numbers.",
    ],
)

writer = Agent(
    name="Writer",
    description="A skilled writer who creates engaging, well-structured content.",
    instructions=[
        "You write clear, engaging content based on provided information.",
        "Use a professional but approachable tone.",
        "Structure your output with headers and bullet points when appropriate.",
    ],
)

critic = Agent(
    name="Critic",
    description="A sharp editor who reviews content for accuracy and clarity.",
    instructions=[
        "You review content for factual accuracy, clarity, and tone.",
        "Suggest specific improvements where needed.",
        "Be constructive but honest in your feedback.",
    ],
)

# --- 2. Create the team ---
# Members inherit the team's model when they don't specify their own
content_team = Team(
    name="Content Team",
    mode=TeamMode.coordinate,
    model=OpenAIChat(id=settings.OPENAI_MODEL_NAME),
    members=[researcher, writer, critic],
    instructions=[
        "You are a content production team.",
        "The Researcher finds facts, the Writer drafts content,",
        "and the Critic reviews it for quality.",
    ],
    markdown=True,
)

# --- 3. Run the team ---
run_output = content_team.run(
    "Write a short article about the history of the Python programming language."
)

# --- 4. Print the result ---
pprint_run_response(run_output)
