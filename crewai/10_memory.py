import os

from crewai import Agent, Task, Crew, Process

from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore CrewAI with the following memory features:
- Short-Term Memory: RAG-based context for the current session
- Long-Term Memory: Persistent storage of task results across sessions
- Entity Memory: RAG-based tracking of entities (people, places, concepts)
- Enabling memory with memory=True on a Crew

The CrewAI framework provides a sophisticated memory system designed
to significantly enhance AI agent capabilities. Enabling memory=True
activates short-term, long-term, and entity memory by default.

For more details, visit:
https://docs.crewai.com/en/concepts/memory
-------------------------------------------------------
"""

# --- 1. Create a simple agent ---
agent = Agent(
    role="Simple Agent",
    goal="Respond to tasks",
    backstory="You are a simple agent that responds to tasks.",
    llm=settings.OPENAI_MODEL_NAME,
    verbose=True,
)

# --- 2. Define a task that uses memory ---
stock_price = Task(
    description="Search for the current stock price of {company}",
    expected_output="The current stock price of {company}",
    agent=agent,
)

# --- 3. Create a crew with memory enabled ---
# Enable Basic Memory System (Short-Term, Long-Term, Entity Memory)
crew = Crew(
    agents=[agent],
    tasks=[stock_price],
    process=Process.sequential,
    memory=True,  # Enables short-term, long-term, and entity memory
    verbose=True,
)

# --- 4. Run the crew once ---
# This will fetch the stock price of Apple and stores it in memory
result1 = crew.kickoff(inputs={"company": "Apple"})

# --- 5. Run the crew again ---
# This time, the agent should recall the previous result from memory
result2 = crew.kickoff(inputs={"company": "Apple"})

# --- 6. Check the storage directory (optional) ---
# Linux: ~/.local/share/CrewAI/{project_name}/
#   --> this example: ~/.local/share/crewai/
# MacOS: ~/Library/Application Support/CrewAI/{project_name}/
#   --> this example: ~/Library/Application Support/crewai/
# Windows: C:\Users\{username}\AppData\Local\CrewAI\{project_name}\
#   --> this example: C:\Users\{username}\AppData\Local\crewai\
