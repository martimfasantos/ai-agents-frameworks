import os
import sqlite3
import tempfile

import autogen.runtime_logging as runtime_logging
from autogen import ConversableAgent, LLMConfig

from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore AG2 with the following features:
- Runtime logging to SQLite for observability
- Capturing chat completions, agent events, and metadata
- Querying logged data for analysis after execution

AG2 provides built-in runtime logging that records agent
interactions, LLM calls, and events to a SQLite database.
This enables post-hoc analysis, debugging, and monitoring
of agent behavior without modifying agent code.

For more details, visit:
https://docs.ag2.ai/latest/docs/user-guide/observability/logging_events/
-------------------------------------------------------
"""

# --- 1. Configure LLM ---
llm_config = LLMConfig({"model": settings.OPENAI_MODEL_NAME})

# --- 2. Set up runtime logging ---
db_path = os.path.join(tempfile.gettempdir(), "ag2_logging_demo.db")
if os.path.exists(db_path):
    os.remove(db_path)

print("=== Observability: Runtime Logging ===\n")
session_id = runtime_logging.start(
    logger_type="sqlite",
    config={"dbname": db_path},
)
print(f"Logging session started: {session_id}")
print(f"Database: {db_path}\n")

# --- 3. Create agents ---
assistant = ConversableAgent(
    name="assistant",
    system_message=(
        "You are a helpful assistant. Answer questions concisely in 1-2 sentences."
    ),
    llm_config=llm_config,
    human_input_mode="NEVER",
)

user = ConversableAgent(
    name="user",
    human_input_mode="NEVER",
    llm_config=False,
    is_termination_msg=lambda x: True,
)

# --- 4. Run a conversation (logged automatically) ---
print("--- Running conversation 1 ---")
user.initiate_chat(
    assistant,
    message="What is the speed of light?",
    max_turns=1,
)

print("\n--- Running conversation 2 ---")
user.initiate_chat(
    assistant,
    message="What is the largest ocean on Earth?",
    max_turns=1,
)

# --- 5. Stop logging ---
runtime_logging.stop()
print("\nLogging session stopped.")

# --- 6. Analyze the logged data ---
print("\n=== Log Analysis ===\n")

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# List all tables and their row counts
tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()

for (table_name,) in tables:
    count = cursor.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    if count > 0:
        print(f"  {table_name}: {count} record(s)")

# Show chat completions
print("\n--- Chat Completions ---")
completions = cursor.execute(
    "SELECT id, source_name, start_time, end_time FROM chat_completions"
).fetchall()
for comp_id, source, start, end in completions:
    print(f"  [{comp_id}] Agent: {source} | Start: {start} | End: {end}")

# Show registered agents
print("\n--- Registered Agents ---")
agents = cursor.execute("SELECT name, class FROM agents").fetchall()
for name, cls in agents:
    print(f"  {name} ({cls})")

# Show events
print("\n--- Events ---")
events = cursor.execute("SELECT source_name, event_name FROM events").fetchall()
for source, event in events:
    preview = event[:80] + "..." if len(event) > 80 else event
    print(f"  [{source or 'system'}] {preview}")

conn.close()

# --- 7. Clean up ---
os.remove(db_path)
print(f"\n=== Observability Demo Complete ===")
print(
    f"Session {session_id} logged {len(completions)} completions and {len(events)} events."
)
