import os

from crewai import Agent, Task, Crew, LLM
from crewai.events import LLMStreamChunkEvent, BaseEventListener

from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore CrewAI's agents with the following features:
- Real-time output streaming during task execution
- Step-by-step execution visibility with listeners

CrewAI supports streaming by setting stream=True in the LLM configuration.
Additionally, custom event listeners can be created to handle streaming events.
This enables real-time monitoring of agent outputs as they are generated.

For more details, visit:
https://docs.crewai.com/en/concepts/llms#streaming-responses
-------------------------------------------------------
"""

# --- 1. Create an LLM with streaming enable ---
llm = LLM(
    model=settings.OPENAI_MODEL_NAME,
    api_key=settings.OPENAI_API_KEY.get_secret_value(),
    temperature=0,
    stream=True,  # Enable streaming
)


# --- 2. Define a custom event listener for streaming events (optional) ---
class MyCustomListener(BaseEventListener):
    def setup_listeners(self, crewai_event_bus):
        @crewai_event_bus.on(LLMStreamChunkEvent)
        def on_llm_stream_chunk(source, event):
            if researcher.id == event.agent_id:
                print("\n==============\n Got event:", event, "\n==============\n")


my_listener = MyCustomListener()

# --- 3. Define the agent ---
researcher = Agent(
    role="About User",
    goal="You know everything about the user.",
    backstory=("You are a master at understanding people and their preferences."),
    llm=llm,
    verbose=True,
)

# --- 4. Define the task ---
search = Task(
    description="Answer the following questions about the user: {question}",
    expected_output="An answer to the question.",
    agent=researcher,
)

# --- 5. Create the crew with the event listener ---
crew = Crew(
    agents=[researcher],
    tasks=[search],
)

# --- 6. Run the crew (will print streaming events in real-time) ---
result = crew.kickoff(inputs={"question": "What is the capital of Portugal?"})
