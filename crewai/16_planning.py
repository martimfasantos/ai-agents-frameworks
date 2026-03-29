import os

from crewai import Agent, Task, Crew

from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore CrewAI with the following features:
- Crew planning mode with planning=True
- Automatic step-by-step plan generation before execution
- Custom planning LLM configuration

When planning is enabled, CrewAI uses an AgentPlanner to create
a detailed step-by-step plan for each task before the crew begins
execution. This improves task quality and agent coordination.

For more details, visit:
https://docs.crewai.com/en/concepts/planning
-------------------------------------------------------
"""

# --- 1. Create agents ---
researcher = Agent(
    role="Research Analyst",
    goal="Research topics thoroughly and provide factual summaries",
    backstory="You are a meticulous researcher who values accuracy.",
    llm=settings.OPENAI_MODEL_NAME,
    verbose=True,
)

writer = Agent(
    role="Content Writer",
    goal="Transform research into engaging content",
    backstory="You are a skilled writer who creates clear, readable content.",
    llm=settings.OPENAI_MODEL_NAME,
    verbose=True,
)

# --- 2. Create tasks ---
research_task = Task(
    description="Research the current state of AI agents in 2025, focusing on key trends.",
    expected_output="A structured summary of AI agent trends in 2025.",
    agent=researcher,
)

writing_task = Task(
    description="Write a short blog post based on the research findings about AI agents.",
    expected_output="A concise blog post about AI agent trends.",
    agent=writer,
)

# --- 3. Create a crew with planning enabled ---
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    planning=True,  # Enables automatic planning before execution
    # planning_llm="gpt-4o",  # Optionally specify a different LLM for planning
    verbose=True,
)

# --- 4. Run the crew (planning happens automatically before task execution) ---
result = crew.kickoff()
print("Result:", result.raw[:500])
