import os
from typing import List

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent

from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore CrewAI's Crew simplification features:
- Declarative simple crew, agent, and task definitions using CrewBase
- YAML-based configuration for agents and tasks
- Seamless integration of agents, tasks, and crews
- Simplified code for building and running multi-agent teams
- Support for both Python and YAML agent/task definitions

This demonstrates how to:
1. Use YAML files for agent and task configuration
2. Build a crew with minimal Python code using CrewBase
3. Integrate agents and tasks from config files into a working crew
4. Maintain flexibility for custom Python logic if needed

For more details, visit:
https://docs.crewai.com/en/concepts/crews#yaml-configuration-recommended
-------------------------------------------------------
"""


# --- 1. Define a Crew using CrewAI simplifications ---
@CrewBase
class ResearchCrew:
    """Research crew for comprehensive topic analysis and reporting"""

    agents: List[BaseAgent]
    tasks: List[Task]

    # Paths to your YAML configuration files
    # To see an example agent and task defined in YAML, checkout the following:
    # - Task: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    # - Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config["researcher"],
            tools=[],
            llm=settings.OPENAI_MODEL_NAME,
            verbose=True,
        )

    @agent
    def analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["analyst"],
            llm=settings.OPENAI_MODEL_NAME,
            verbose=True,
        )

    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config["research_task"]
            # you can add more custom parameters here if needed
        )

    @task
    def analysis_task(self) -> Task:
        return Task(
            config=self.tasks_config["analysis_task"],
            # output_file='output/report.md'
        )

    @crew
    def crew(self) -> Crew:
        """Creates the research crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            manager_llm=settings.OPENAI_MODEL_NAME,
        )


# --- 2. Run the crew with a given topic ---
if __name__ == "__main__":
    ResearchCrew().crew().kickoff(inputs={"topic": "AI in Portugal"})
