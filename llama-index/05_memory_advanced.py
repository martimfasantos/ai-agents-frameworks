import asyncio
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.llms import ChatMessage
from llama_index.core.memory import Memory, StaticMemoryBlock, FactExtractionMemoryBlock
from llama_index.llms.openai import OpenAI
from settings import settings


"""
-------------------------------------------------------
In this example, we explore LlamaIndex with the following features:
- Short-term memory with token limits and FIFO queue
- Long-term memory with Memory Blocks (Static, Fact Extraction)
- Manual memory manipulation with put/get operations
- Multi-turn conversations with context retention

LlamaIndex's Memory class provides both short-term memory (a FIFO queue of
messages) and long-term memory (using Memory Blocks). Short-term memory
manages recent conversation history with token limits, while long-term memory
stores important information extracted from conversations.
This enables agents to maintain conversation context while respecting token
limits and preserving important information.

For more details, visit:
https://developers.llamaindex.ai/python/framework/module_guides/deploying/agents/memory/
https://developers.llamaindex.ai/python/examples/memory/memory
-------------------------------------------------------
"""


# --- 1. Define tools ---
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    # Simulated weather data
    weather_data = {
        "lisbon": "Sunny, 22°C",
        "london": "Cloudy, 15°C",
        "new york": "Rainy, 18°C",
        "tokyo": "Clear, 25°C",
    }
    return weather_data.get(city.lower(), f"Weather data not available for {city}")

def get_time() -> str:
    """Get the current time."""
    from datetime import datetime
    return datetime.now().strftime("%I:%M %p")


# --- 2. Short-Term Memory Demo ---
async def demo_short_term_memory():
    """Demonstrate short-term memory configuration and behavior."""
    print("\n" + "=" * 60)
    print("DEMO 1: Short-Term Memory")
    print("=" * 60)
    
    # --- 2.1 Configure short-term memory ---
    memory = Memory.from_defaults(
        session_id="short_term_demo",
        token_limit=500,
        token_flush_size=100,
        chat_history_token_ratio=0.7,
    )
    
    # --- 2.2 Manually add messages to memory ---
    await memory.aput(ChatMessage(role="user", content="Hello! My name is Alice."))
    await memory.aput(ChatMessage(role="assistant", content="Hello Alice! Nice to meet you."))
    await memory.aput(ChatMessage(role="user", content="I live in Lisbon, Portugal."))
    await memory.aput(ChatMessage(role="assistant", content="Lisbon is a beautiful city!"))
    
    # --- 2.3 Retrieve current chat history ---
    chat_history = await memory.aget()
    print(f"Stored {len(chat_history)} messages in memory")
    
    # --- 2.4 Get all messages (including archived) ---
    all_messages = await memory.aget_all()
    print(f"Total messages (including archived): {len(all_messages)}")
    
    # --- 2.5 Reset memory ---
    await memory.areset()
    all_messages = await memory.aget_all()
    print(f"Messages after reset: {len(all_messages)}")


# --- 3. Long-Term Memory Demo ---
async def demo_long_term_memory():
    """Demonstrate long-term memory with Memory Blocks."""
    print("\n" + "=" * 60)
    print("DEMO 2: Long-Term Memory with Memory Blocks")
    print("=" * 60)
    
    # --- 3.1 Create LLM ---
    llm = OpenAI(
        model=settings.OPENAI_MODEL_NAME,
        api_key=settings.OPENAI_API_KEY.get_secret_value()
    )
    
    # --- 3.2 Define Memory Blocks ---
    # Memory blocks process flushed messages and store long-term information
    blocks = [
        # Static block: Always present information (priority=0 means always kept)
        StaticMemoryBlock(
            name="user_profile",
            static_content="The user is a software developer interested in AI and lives in Portugal.",
            priority=0,  # Always kept in memory
        ),
        # Fact extraction block: Extracts and stores facts from conversations
        FactExtractionMemoryBlock(
            name="extracted_facts",
            llm=llm,
            max_facts=20,  # Maximum facts to store
            priority=1,  # Can be truncated if needed
        ),
    ]
    
    # --- 3.3 Create memory with blocks ---
    memory = Memory.from_defaults(
        session_id="long_term_demo",
        token_limit=2000,
        chat_history_token_ratio=0.5,  # 50% short-term, 50% long-term
        token_flush_size=200,
        memory_blocks=blocks,
        insert_method="system",  # Insert long-term memory into system message
    )
    
    # --- 3.4 Create agent with memory ---
    agent = FunctionAgent(
        name="memory_agent",
        description="An agent with advanced memory capabilities.",
        llm=llm,
        tools=[get_weather, get_time],
        system_prompt="You are a helpful assistant with excellent memory.",
    )
    
    # --- 3.5 Have a conversation that builds memory ---
    conversations = [
        "Hi! I'm working on a new Python project using LlamaIndex.",
        "What's the weather like in Lisbon today?",
        "I'm planning to add memory features to my AI agent.",
        "Can you remind me what project I'm working on?",
    ]
    
    for user_msg in conversations:
        response = await agent.run(user_msg=user_msg, memory=memory)
        print(f"User: {user_msg}")
        print(f"Agent: {response}\n")
    
    # --- 3.6 Inspect memory state ---
    chat_history = await memory.aget()
    print(f"Memory contains {len(chat_history)} messages")
    if chat_history and chat_history[0].role == "system":
        if hasattr(chat_history[0], 'blocks') and chat_history[0].blocks:
            print(f"Long-term memory blocks: {len(chat_history[0].blocks)}")


# --- 4. Run the examples ---
async def main():
    """Run the memory demonstrations."""
    
    # 4.1 Run demonstrations
    await demo_short_term_memory()
    await demo_long_term_memory()
    
    print("\n" + "=" * 60)
    
    """
    Expected outputs:
    
    DEMO 1 - Short-Term Memory:
    - Shows memory configuration with token limits
    - Manual message addition and retrieval
    - Memory reset functionality
    
    DEMO 2 - Long-Term Memory:
    - StaticMemoryBlock provides constant context
    - FactExtractionMemoryBlock extracts facts from conversations
    - Memory blocks have priorities for truncation
    """

if __name__ == "__main__":
    asyncio.run(main())
