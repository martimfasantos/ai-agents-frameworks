import asyncio
import os

from agno.agent import Agent
from agno.db.migrations.manager import MigrationManager
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIChat

from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore Agno with the following features:
- The agno_runs table introduced in Agno v3.0
- Direct run APIs: db.get_runs(), db.get_run(), db.delete_run()
- Filtering runs by status and paging them
- MigrationManager, which a 2.x database must run before v3 serves traffic

Through Agno 2.x every run was appended to a JSON blob on the session row,
so writing run N rewrote runs 1..N-1 with it — O(N^2) write amplification,
and on DynamoDB or Firestore an eventual collision with the item-size limit.
In v3 each run is its own row in agno_runs with real columns, and the runs
re-attach on read so session APIs are unchanged. A database written by 2.x
must be migrated first: MigrationManager(db).up() is non-destructive and
idempotent, and an unmigrated database raises a typed error rather than
misbehaving quietly.

For more details, visit:
https://docs.agno.com/sessions/persisting-sessions/overview
-------------------------------------------------------
"""

DB_FILE = "/tmp/agno_runs_example.db"

# Start from a clean file so the run counts below are the example's own.
if os.path.exists(DB_FILE):
    os.remove(DB_FILE)

# --- 1. A database, and the schema version it targets ---
db = SqliteDb(db_file=DB_FILE)
manager = MigrationManager(db)

print("=== Runs table (Agno v3) ===\n")
print(f"Latest schema version: {manager.latest_schema_version}")
print(f"Available migrations:  {manager.available_versions}")
# up() is what a database written by 2.x needs before v3 serves traffic. On a
# fresh database it is a no-op, which is why it is safe to call unconditionally.
# It is a coroutine even on a sync db, so it has to be awaited.
asyncio.run(manager.up())
print("MigrationManager.up() applied (no-op on a fresh database)\n")

# --- 2. Three runs in one session ---
agent = Agent(
    model=OpenAIChat(id=settings.OPENAI_MODEL_NAME),
    db=db,
    instructions="Reply with one short sentence.",
    session_id="trip-planning",
)

for question in (
    "Name one thing to do in Lisbon.",
    "And one in Porto?",
    "Which of the two is further north?",
):
    agent.run(question)

# --- 3. Each run is now its own row, queryable directly ---
runs = db.get_runs(session_id="trip-planning")
print(f"Runs recorded for session 'trip-planning': {len(runs)}")
for index, run in enumerate(runs):
    print(f"  {index}: run_id={run.run_id[:8]}… status={run.status}")

# --- 4. Filters and paging work on the table, not on a decoded blob ---
completed = db.get_runs(session_id="trip-planning", status="COMPLETED")
first_page = db.get_runs(session_id="trip-planning", limit=2, page=1)
print(f"\nFiltered by status=COMPLETED: {len(completed)}")
print(f"First page (limit=2):         {len(first_page)}")

# --- 5. A single run fetched by id ---
one = db.get_run(run_id=runs[0].run_id)
print(f"\nFetched run {one.run_id[:8]}… directly")
print(f"  content: {str(one.content).strip()[:70]}")

# --- 6. The session API is unchanged — runs re-attach on read ---
session = db.get_session(session_id="trip-planning")
print(f"\nSession still reports {len(session.runs)} runs and")
print(f"{len(session.get_messages())} messages, exactly as it did in 2.x.")
