from dotenv import load_dotenv

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIChat
from agno.utils.pprint import pprint_run_response

from settings import settings

load_dotenv()

"""
-------------------------------------------------------
In this example, we explore Agno with the following features:
- Session storage with SqliteDb
- Persisting conversation history to a SQLite database
- Restoring context from previous sessions via session_id
- add_history_to_context for automatic context injection

Storage lets agents maintain conversation history across
process restarts. By attaching a SqliteDb and providing a
session_id, the agent automatically saves each interaction
and restores previous messages when resuming a session.

For more details, visit:
https://docs.agno.com/agents/storage
-------------------------------------------------------
"""

# --- 1. Configure SQLite storage ---
db = SqliteDb(
    db_file="/tmp/agno_storage_example.db",
)

# --- 2. Create the agent with storage ---
agent = Agent(
    model=OpenAIChat(id=settings.OPENAI_MODEL_NAME),
    db=db,
    add_history_to_context=True,
    num_history_runs=5,
    instructions=[
        "You are a helpful coding tutor.",
        "Remember what the student has been learning.",
        "Build on previous topics in the conversation.",
    ],
    markdown=True,
)

SESSION_ID = "coding-tutorial-session-001"

# --- 3. First interaction ---
print("=== Turn 1 ===\n")
run_output = agent.run(
    "I want to learn Python. Let's start with variables.",
    session_id=SESSION_ID,
)
pprint_run_response(run_output)

# --- 4. Second interaction (same session — history is preserved) ---
print("\n=== Turn 2 ===\n")
run_output = agent.run(
    "Great, now teach me about loops.",
    session_id=SESSION_ID,
)
pprint_run_response(run_output)

# --- 5. Third interaction (demonstrates context from history) ---
print("\n=== Turn 3 ===\n")
run_output = agent.run(
    "Can you give me an exercise that combines what we've covered so far?",
    session_id=SESSION_ID,
)
pprint_run_response(run_output)
