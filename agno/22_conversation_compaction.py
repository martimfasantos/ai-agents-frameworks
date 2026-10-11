import os

from dotenv import load_dotenv

from agno.agent import Agent
from agno.compaction import Compaction
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIChat

from settings import settings

load_dotenv()

"""
-------------------------------------------------------
In this example, we explore Agno with the following features:
- Conversation compaction, added in Agno v3.1.2
- agent.compact() to fold older turns into a summary on demand
- The archived transcript and the search_compacted_history tool

A long session eventually outgrows the context window. With compaction, older
turns are replaced in the model's context by a short summary, while the
original messages are archived in the database. Because the archive is
searchable, the summary acts as an index rather than a lossy replacement: when
a question needs a detail the summary dropped, the agent searches the archive.

For more details, visit:
https://github.com/agno-agi/agno/blob/v3.1.2/cookbook/02_agents/03_context_management/compaction/compaction_searchable_archive.py
-------------------------------------------------------
"""

DB_FILE = "/tmp/agno_compaction_example.db"
SESSION_ID = "inventory-session-001"
OWNERS = ["ana", "rui", "ines", "joao", "marta"]

if os.path.exists(DB_FILE):
    os.remove(DB_FILE)


# --- 1. Mock data with values a model cannot guess ---
def port_for(index: int) -> int:
    return 10_000 + index * 7919 % 50_000


inventory = "\n".join(
    f"service-{i:02d}: port {port_for(i)}, owner {OWNERS[i % 5]}" for i in range(1, 26)
)

# --- 2. Configure compaction ---
compaction = Compaction(
    # Fold only when agent.compact() is called, so every run folds at the same point.
    compact_at_tokens=None,
    uncompacted_runs=1,
    # Too small to carry a 25-row inventory, so the rows can only come back from the archive.
    compacted_token_budget=150,
    search_compacted_messages=True,
)

agent = Agent(
    model=OpenAIChat(id=settings.OPENAI_MODEL_NAME),
    db=SqliteDb(db_file=DB_FILE),
    session_id=SESSION_ID,
    add_history_to_context=True,
    compaction=compaction,
    instructions="Keep replies to one short sentence.",
)

# --- 3. Build up history worth compacting ---
print("=== Turn 1: share a 25-row service inventory ===")
response = agent.run(f"Here is our service inventory for later. Just say noted.\n{inventory}")
print(f"Agent: {response.content}\n")

print("=== Turn 2: an unrelated question ===")
response = agent.run("What is a service mesh?")
print(f"Agent: {response.content}\n")

# --- 4. Fold everything before the last turn into a summary ---
print("=== Compacting the session ===")
result = agent.compact(session_id=SESSION_ID)
print(f"Status:          {result.status.value}")
if result.record is not None:
    record = result.record
    summary = " ".join(record.summary.split())
    print(f"Messages folded: {record.messages_compacted}")
    print(f"Context tokens:  {record.tokens_before} -> {record.tokens_after}")
    print(f"Summary:         {summary[:160]}…")
    print(f"Summary holds service-17's port: {str(port_for(17)) in record.summary}\n")

# --- 5. Ask for a detail the summary could not keep ---
print("=== Turn 3: recall a single row ===")
response = agent.run("What port does service-17 use, and who owns it?")
for tool in response.tools or []:
    print(f"Tool call: {tool.tool_name}({tool.tool_args})")
print(f"Agent:     {response.content}")
print(f"Expected:  port {port_for(17)}, owner {OWNERS[17 % 5]}")
