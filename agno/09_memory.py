from dotenv import load_dotenv

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.memory.manager import MemoryManager
from agno.models.openai import OpenAIChat
from agno.utils.pprint import pprint_run_response

from settings import settings

load_dotenv()

"""
-------------------------------------------------------
In this example, we explore Agno with the following features:
- Agentic memory with MemoryManager
- enable_agentic_memory for automatic memory management
- Persistent user context across multiple interactions
- Memory recall in follow-up questions

Agentic memory lets an agent remember facts from previous
interactions and recall them when relevant. The MemoryManager
automatically extracts and stores important information from
conversations. On subsequent runs, the agent retrieves
relevant memories to provide personalized responses.

For more details, visit:
https://docs.agno.com/agents/memory
-------------------------------------------------------
"""

# --- 1. Configure storage for memory persistence ---
memory_db = SqliteDb(
    db_file="/tmp/agno_memory_example.db",
)

# --- 2. Create a memory manager with storage ---
memory = MemoryManager(
    model=OpenAIChat(id=settings.OPENAI_MODEL_NAME),
    db=memory_db,
)

# --- 3. Create the agent with agentic memory ---
# The Agent needs its own db= parameter for memory persistence.
# During initialization, agent.db is propagated to memory_manager.db
# if the memory manager's db is not set.
agent = Agent(
    model=OpenAIChat(id=settings.OPENAI_MODEL_NAME),
    memory_manager=memory,
    db=memory_db,
    enable_agentic_memory=True,
    instructions=[
        "You are a helpful personal assistant.",
        "Remember details the user shares about themselves.",
        "Use what you remember to personalize your responses.",
    ],
    markdown=True,
)

# --- 4. First interaction — share some facts ---
print("=== Interaction 1: Sharing information ===\n")
run_output = agent.run(
    "My name is Alice. I'm a software engineer who loves hiking and Portuguese food.",
    user_id="alice_123",
)
pprint_run_response(run_output)

# --- 5. Second interaction — the agent should recall details ---
print("\n=== Interaction 2: Testing recall ===\n")
run_output = agent.run(
    "Can you suggest a weekend activity for me?",
    user_id="alice_123",
)
pprint_run_response(run_output)

# --- 6. Third interaction — more specific recall ---
print("\n=== Interaction 3: Specific recall ===\n")
run_output = agent.run(
    "What do you remember about me?",
    user_id="alice_123",
)
pprint_run_response(run_output)
