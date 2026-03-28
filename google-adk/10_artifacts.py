import os
import asyncio
import logging

from google.adk.agents import LlmAgent
from google.adk.artifacts import InMemoryArtifactService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import ToolContext  # type: ignore[attr-defined]
from google.genai import types

from settings import settings

os.environ["GOOGLE_API_KEY"] = settings.GOOGLE_API_KEY.get_secret_value()

# Suppress the SDK's "non-text parts in response" informational warning
logging.getLogger("google_genai.types").setLevel(logging.ERROR)

"""
-------------------------------------------------------
In this example, we explore Google ADK with the following features:
- Artifacts: save and load versioned binary data within a session
- InMemoryArtifactService: the in-process artifact backend
- ToolContext.save_artifact / load_artifact: accessing artifacts from tools
- Versioning: each save creates a new version; load retrieves the latest

Artifacts let agents store file-like binary data (text, PDFs, images) that
persists across tool calls in a session. They are separate from session state
and designed for larger or binary payloads. This example shows a report-writer
tool that saves a text report as an artifact, and a report-reader tool that
loads it back.

For more details, visit:
https://google.github.io/adk-docs/artifacts/
-------------------------------------------------------
"""

APP_NAME = "artifact_demo"
USER_ID = "user"
SESSION_ID = "session_1"

REPORT_FILENAME = "report.txt"


# --- 1. Define tools that use the artifact service ---


async def save_report(topic: str, tool_context: ToolContext) -> dict:
    """Generate and save a short report on the given topic as an artifact."""
    content = (
        f"Report on: {topic}\n"
        f"{'=' * 40}\n"
        f"This report covers key aspects of {topic}. "
        f"It was generated and stored as a versioned artifact "
        f"so it can be retrieved in later tool calls or sessions.\n"
    )
    artifact = types.Part.from_bytes(
        data=content.encode("utf-8"),
        mime_type="text/plain",
    )
    version = await tool_context.save_artifact(
        filename=REPORT_FILENAME, artifact=artifact
    )
    return {"status": "saved", "filename": REPORT_FILENAME, "version": version}


async def load_report(tool_context: ToolContext) -> dict:
    """Load the latest saved report artifact and return its text content."""
    artifact = await tool_context.load_artifact(filename=REPORT_FILENAME)
    if artifact is None:
        return {"status": "not_found", "content": None}
    text = artifact.inline_data.data.decode("utf-8")  # type: ignore[union-attr]
    return {"status": "loaded", "content": text}


async def list_artifacts(tool_context: ToolContext) -> dict:
    """List all artifact filenames available in the current session."""
    filenames = await tool_context.list_artifacts()
    return {"artifacts": filenames}


# --- 2. Create the agent with artifact-aware tools ---

artifact_agent = LlmAgent(
    name="ArtifactAgent",
    model=settings.GOOGLE_MODEL_NAME,
    instruction=(
        "You are a report assistant. "
        "When asked to save a report, call save_report with the topic. "
        "When asked to load or read the report, call load_report. "
        "When asked to list artifacts, call list_artifacts. "
        "Be concise — summarise what you did in one sentence."
    ),
    tools=[save_report, load_report, list_artifacts],
)


# --- 3. Set up runner with InMemoryArtifactService ---


async def run_demo() -> None:
    session_service = InMemorySessionService()
    artifact_service = InMemoryArtifactService()

    await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
    )

    runner = Runner(
        agent=artifact_agent,
        app_name=APP_NAME,
        session_service=session_service,
        artifact_service=artifact_service,
    )

    async def send(query: str) -> str:
        message = types.Content(role="user", parts=[types.Part(text=query)])
        response_text = ""
        async for event in runner.run_async(
            user_id=USER_ID, session_id=SESSION_ID, new_message=message
        ):
            if event.is_final_response() and event.content and event.content.parts:
                response_text = event.content.parts[0].text or ""
        return response_text

    # --- 4. Demonstrate save, list, and load ---

    print("\n" + "-" * 65)
    print("  Step 1: Save a report artifact")
    print("-" * 65)
    reply = await send("Save a report on renewable energy.")
    print(f"  Agent: {reply.strip()}")

    print("\n" + "-" * 65)
    print("  Step 2: List artifacts in the session")
    print("-" * 65)
    reply = await send("What artifacts are stored in this session?")
    print(f"  Agent: {reply.strip()}")

    print("\n" + "-" * 65)
    print("  Step 3: Load and display the saved report")
    print("-" * 65)
    reply = await send("Load the report and tell me what it says.")
    print(f"  Agent: {reply.strip()}")

    # --- 5. Save a second version of the artifact to show versioning ---

    print("\n" + "-" * 65)
    print("  Step 4: Save an updated report (versioning)")
    print("-" * 65)
    reply = await send("Save an updated report on solar energy specifically.")
    print(f"  Agent: {reply.strip()}")
    print()


if __name__ == "__main__":
    asyncio.run(run_demo())
