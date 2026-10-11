import os

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIChat
from agno.offload.store import ResultStore
from agno.tools import tool

from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore Agno with the following features:
- Tool result offloading, added in Agno v3.0
- ResultStore to tune the threshold and preview size
- The read_result / search_result tools the agent gets in exchange

A tool that returns a large payload — a file dump, a big query result, a
scraped page — puts all of it in the transcript, where it is re-sent on
every subsequent model call for the rest of the run. With
offload_tool_results the oversized result is written to AgentFS and the
message keeps a short envelope: a preview, the size, and a result_id. The
agent gets read_result and search_result to fetch the rest only if it
actually needs it, and nothing extra is spent on the write path.

For more details, visit:
https://docs.agno.com/agents/tool-result-offloading
-------------------------------------------------------
"""

DB_FILE = "/tmp/agno_offload_example.db"

if os.path.exists(DB_FILE):
    os.remove(DB_FILE)

db = SqliteDb(db_file=DB_FILE)


# --- 1. A tool with a deliberately oversized result ---
@tool
def fetch_server_log(service: str) -> str:
    """Fetch the recent log for a service."""
    lines = [
        f"2026-09-12T10:{minute:02d}:00Z {service} INFO heartbeat ok latency=12ms"
        for minute in range(60)
    ]
    # The answer is buried in the middle, so a preview alone cannot supply it.
    lines[37] = "2026-09-12T10:37:00Z payments ERROR gateway timeout after 30s"
    payload = "\n".join(lines * 6)
    print(f"  [tool] returning {len(payload):,} characters")
    return payload


# --- 2. Offload anything over the threshold ---
# The default threshold is 16,000 characters; lowered here so a single
# example-sized payload crosses it.
agent = Agent(
    model=OpenAIChat(id=settings.OPENAI_MODEL_NAME),
    db=db,
    tools=[fetch_server_log],
    offload_tool_results=ResultStore(threshold_chars=2_000),
    instructions=(
        "Use fetch_server_log, then report the single ERROR line verbatim. "
        "If the result was offloaded, use search_result or read_result to find it."
    ),
)

print("=== Tool result offloading ===\n")

response = agent.run("What went wrong in the payments service?")

# --- 3. The transcript kept an envelope, not the payload ---
tool_messages = [m for m in response.messages if m.role == "tool"]
for message in tool_messages:
    content = str(message.content)
    print(f"\n  tool message in transcript: {len(content):,} characters")
    print(f"  starts: {content[:100].strip()}…")

# --- 4. The agent reached for the offload tools on its own ---
called = [
    call.get("function", {}).get("name")
    for message in response.messages
    for call in (message.tool_calls or [])
]
print(f"\nTools called: {called}")
print(f"\nAnswer: {str(response.content).strip()}")
