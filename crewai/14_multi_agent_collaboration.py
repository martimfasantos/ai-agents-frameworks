import os

from crewai import Agent, Crew, Task, Process

from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore CrewAI's agents with the following features:
- Multi-agent crews
- Agent collaboration and delegation
- Sequential vs. Hierarchical collaboration patterns
- Task decomposition and assignment

When delegation is enabled, CrewAI agents automatically gain
the ability to delegate tasks and ask questions to their teammates.
Their given 2 powerful tools to facilitate collaboration:
1. Delegate Work tool
2. Ask Question tool

For more details, visit:
https://docs.crewai.com/en/concepts/collaboration
-------------------------------------------------------
"""

# --- 1. Create collaborative agents ---
researcher = Agent(
    role="Research Specialist",
    goal="Find accurate, up-to-date information on any topic",
    backstory=(
        "You're a meticulous researcher with expertise in finding "
        "reliable sources and fact-checking information across various domains."
    ),
    # 🔑 Key setting for collaboration,
    # - Enables delegation work to other agents
    # - Enables asking questions to other agents
    llm=settings.OPENAI_MODEL_NAME,
    allow_delegation=True,
)

# When allow_delegation=True, CrewAI automatically provides agents
# with two powerful tools:
#
# - `Delegate Work tool`:
# Allows agents to assign tasks to teammates with specific expertise.
# Agent automatically gets this tool:
#   Delegate work to coworker(task: str, context: str, coworker: str)
#
# - `Ask Question tool`:
# Enables agents to ask specific questions to gather information from colleagues.
# Agent automatically gets this tool:
#   Ask question to coworker(question: str, coworker: str)

writer = Agent(
    role="Content Writer",
    goal="Create engaging, well-structured content",
    backstory=(
        "You're a skilled content writer who excels at transforming "
        "research into compelling, readable content for different audiences."
    ),
    llm=settings.OPENAI_MODEL_NAME,
    allow_delegation=True,
)

editor = Agent(
    role="Content Editor",
    goal="Ensure content quality and consistency",
    backstory=(
        "You're an experienced editor with an eye for detail, "
        "ensuring content meets high standards for clarity and accuracy."
    ),
    llm=settings.OPENAI_MODEL_NAME,
    allow_delegation=True,
)

# 1.1. Manager agent can coordinate the team (used in hierarchical process)
manager = Agent(
    role="Project Manager",
    goal="Coordinate team efforts and ensure project success",
    backstory="Experienced project manager skilled at delegation and quality control",
    llm=settings.OPENAI_MODEL_NAME,
    allow_delegation=True,
)

# --- 2. Create single tasks that encourages collaboration ---
article_task = Task(
    description=(
        """
        Write a comprehensive 1000-word article about 'The Future of AI in Healthcare'.

        The article should include:
        - Current AI applications in healthcare
        - Emerging trends and technologies
        - Potential challenges and ethical considerations
        - Expert predictions for the next 5 years

        Collaborate with your teammates to ensure accuracy and quality.
        """
    ),
    expected_output=(
        "A well-researched, engaging 1000-word article "
        "with proper structure and citations"
    ),
    agent=writer,  # Writer leads, but can delegate research to researcher
)

project_task = Task(
    description=("Create a comprehensive market analysis report with recommendations"),
    expected_output=(
        "Executive summary, detailed analysis, and strategic recommendations"
    ),
    agent=manager,  # Manager will delegate to specialists
)

# --- 3. Create the crews ---
# 3.1. SEQUENTIAL pattern: Agents collaborate on the task without a manager
crew_sequential = Crew(
    agents=[researcher, writer, editor],
    tasks=[article_task],
    process=Process.sequential,
    verbose=True,
)

# 3.2. HIERARCHICAL pattern: Manager oversees the task and delegates to agents
crew_hierarchical = Crew(
    agents=[researcher, writer],
    tasks=[project_task],
    process=Process.hierarchical,
    manager_agent=manager,
    # manager_llm="gpt-4o",  # Specify LLM for manager
    verbose=True,
)

# --- 4. Kickoff the crews ---
result1 = crew_sequential.kickoff()
print("\n" + "=" * 100 + "\n")
result2 = crew_hierarchical.kickoff()
