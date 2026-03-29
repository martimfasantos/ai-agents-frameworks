import os
import asyncio
from typing import Optional
from pydantic import BaseModel, Field

from crewai import Agent, Crew, Task
from crewai.flow import Flow, listen, start

from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore CrewAI's Flows with integrated agents:
- Integration of agents and crews within flow steps
- Structured state management using Pydantic models
- @start() decorator for flow entry points
- @listen() decorator for method chaining and event listening
- Passing structured data between agents and flow steps
- Asynchronous execution of flows
- Automatic flow visualization (see "res/" folder)

This demonstrates how to:
1. Orchestrate multi-agent workflows using Flows
2. Pass structured outputs between agents and flow steps
3. Use Pydantic models for robust state and output management

For more details, visit:
https://docs.crewai.com/en/concepts/flows
-------------------------------------------------------
"""


# --- 1. Define Pydantic models for structured outputs ---
# 1.1 Structured outputs for agent outputs
class ResearchSummary(BaseModel):
    key_findings: list[str] = Field(description="List of main research findings")
    statistics: list[str] = Field(description="Important statistics and data points")
    trends: list[str] = Field(description="Current trends and developments")


class BlogPost(BaseModel):
    title: str = Field(description="Engaging title for the blog post")
    content: str = Field(description="Well-structured blog post content")
    word_count: int = Field(description="Approximate word count")


# 1.2 Define the overall workflow state
class WorkflowState(BaseModel):
    topic: Optional[str] = None
    research: Optional[ResearchSummary] = None
    blog: Optional[BlogPost] = None


# --- 2. Create a Flow with integrated agents ---
class SimpleAgentFlow(Flow[WorkflowState]):
    # 2.1 Start method using
    @start()
    def start_workflow(self):
        """Start the workflow with a defined topic"""
        print(f"🚀 Starting workflow for: {self.state.topic}")
        return self.state.topic

    # 2.2 Listener method using @listen()
    @listen(start_workflow)
    def research_phase(self):
        """Research agent conducts analysis"""
        print("🔍 Research agent working...")

        researcher = Agent(
            role="Research Analyst",
            goal="Provide structured research findings",
            backstory="Expert in analyzing and summarizing complex topics",
            llm=settings.OPENAI_MODEL_NAME,
            verbose=True,
        )

        research_task = Task(
            description=f"Research {self.state.topic} and provide structured findings",
            agent=researcher,
            expected_output=(
                "Structured research summary with key findings, statistics, and trends"
            ),
            output_pydantic=ResearchSummary,
        )

        research_crew = Crew(agents=[researcher], tasks=[research_task])
        result = research_crew.kickoff()

        if result.pydantic:
            self.state.research = result.pydantic
            print(
                "✅ Research completed with structured data - "
                f"{type(self.state.research)}"
            )
        return result

    # 2.3 Listener method for writing phase - dependent on research_phase
    @listen(research_phase)
    def writing_phase(self):
        """Writing agent creates content"""
        print("✍️ Writing agent working...")

        writer = Agent(
            role="Content Writer",
            goal="Create engaging blog posts from research",
            backstory="Skilled at transforming research into readable content",
            llm=settings.OPENAI_MODEL_NAME,
            verbose=True,
        )

        writing_task = Task(
            description=(
                f"Write a blog post about {self.state.topic} "
                f"using the following data: {self.state.research}"
            ),
            agent=writer,
            expected_output="Structured blog post with title, content and word count",
            output_pydantic=BlogPost,
        )

        writing_crew = Crew(agents=[writer], tasks=[writing_task])
        result = writing_crew.kickoff()

        if result.pydantic:
            self.state.blog = result.pydantic
            print("✅ Blog post created with structured data")
        return result

    # 2.4 Final listener method to complete the workflow
    @listen(writing_phase)
    def complete_workflow(self):
        """Finalize and show results"""
        print("🎉 Workflow completed!")
        print(f"\n📝 Blog Title: {self.state.blog.title if self.state.blog else 'N/A'}")
        if self.state.research:
            print(
                f"📊 Research Findings: {len(self.state.research.key_findings)} key points\n"
                + "\n- ".join(self.state.research.key_findings)
            )
        else:
            print("📊 Research Findings: 0 key points")
        return self.state


# --- 3. Run the flow asynchronously ---
async def run_flow():
    # 3.1 Create flow instance
    flow = SimpleAgentFlow()

    # 3.2 Generate visualization
    flow.plot("res/simple_flow_with_agents_plot")

    # 3.3 Execute the flow asynchronously
    result = await flow.kickoff_async(inputs={"topic": "AI in Healthcare"})
    return result


if __name__ == "__main__":
    asyncio.run(run_flow())
