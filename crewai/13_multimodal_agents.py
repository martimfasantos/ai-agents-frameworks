import os

from crewai import Agent, Task, Crew

from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore CrewAI with the following features:
- Multimodal agents that can process images
- The multimodal=True flag on agents
- Passing image URLs in task descriptions

Multimodal agents can analyze images alongside text. When
multimodal=True is set, CrewAI automatically adds image processing
capabilities to the agent, enabling vision-based tasks.

For more details, visit:
https://docs.crewai.com/en/learn/multimodal-agents
-------------------------------------------------------
"""

# --- 1. Create a multimodal agent ---
image_analyst = Agent(
    role="Image Analyst",
    goal="Analyze images and provide detailed descriptions",
    backstory="You are an expert at analyzing visual content and extracting insights.",
    llm=settings.OPENAI_MODEL_NAME,
    multimodal=True,  # Enables image processing capabilities
    verbose=True,
)

# --- 2. Create a task that includes an image URL ---
# Using a public domain image for demonstration
image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/300px-PNG_transparency_demonstration_1.png"

analysis_task = Task(
    description=(
        f"Analyze the following image and describe what you see in detail. "
        f"Image: {image_url}"
    ),
    expected_output="A detailed description of the image content, colors, and composition.",
    agent=image_analyst,
)

# --- 3. Create and run the crew ---
crew = Crew(
    agents=[image_analyst],
    tasks=[analysis_task],
    verbose=True,
)

result = crew.kickoff()
print("Analysis result:", result.raw[:500])
