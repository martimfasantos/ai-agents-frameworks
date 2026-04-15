import os

from crewai import Agent, Task, Crew, Process

from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore CrewAI with the following features:
- Token usage tracking via CrewOutput.token_usage (UsageMetrics)
- Prompt, completion, cached prompt, and total token counts
- Per-task output inspection with task-level details
- Successful request counting

CrewAI automatically tracks token usage across all agent
interactions during a crew execution. The CrewOutput object
returned by crew.kickoff() contains a token_usage field with
a UsageMetrics dataclass providing prompt_tokens,
completion_tokens, cached_prompt_tokens, total_tokens, and
successful_requests.

For more details, visit:
https://docs.crewai.com/concepts/crews#crew-output
-------------------------------------------------------
"""

# --- 1. Create agents ---
researcher = Agent(
    role="Research Analyst",
    goal="Find and summarize key facts about a topic",
    backstory="You are an experienced research analyst who provides concise, factual summaries.",
    llm=settings.OPENAI_MODEL_NAME,
    verbose=False,
)

writer = Agent(
    role="Technical Writer",
    goal="Write a clear, concise summary from research findings",
    backstory="You are a technical writer who creates short, readable summaries.",
    llm=settings.OPENAI_MODEL_NAME,
    verbose=False,
)

# --- 2. Create tasks ---
research_task = Task(
    description="Research the main benefits of renewable energy. List 3 key points.",
    expected_output="A list of 3 key benefits of renewable energy, each in one sentence.",
    agent=researcher,
)

writing_task = Task(
    description="Using the research findings, write a 2-3 sentence summary about renewable energy benefits.",
    expected_output="A concise 2-3 sentence summary of renewable energy benefits.",
    agent=writer,
)

# --- 3. Create and run the crew ---
print("=== CrewAI Token Usage Tracking ===\n")
print("--- Running crew (2 agents, 2 tasks) ---\n")

crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    process=Process.sequential,
    verbose=False,
)

result = crew.kickoff()

# --- 4. Display crew output ---
print(f"Final output:\n{result.raw}\n")

# --- 5. Display token usage ---
print("--- Token Usage (CrewOutput.token_usage) ---")
usage = result.token_usage
print(f"  Prompt tokens:        {usage.prompt_tokens}")
print(f"  Completion tokens:    {usage.completion_tokens}")
print(f"  Cached prompt tokens: {usage.cached_prompt_tokens}")
print(f"  Total tokens:         {usage.total_tokens}")
print(f"  Successful requests:  {usage.successful_requests}")

# --- 6. Display per-task outputs ---
print("\n--- Per-Task Outputs ---")
for i, task_output in enumerate(result.tasks_output):
    print(f"\n  Task {i + 1}: {task_output.description[:60]}...")
    print(f"    Agent: {task_output.agent}")
    print(f"    Output preview: {task_output.raw[:100]}...")

print("\n=== Token Usage Demo Complete ===")
