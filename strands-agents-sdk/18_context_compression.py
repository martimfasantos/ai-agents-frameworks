import os

from strands import Agent, tool
from strands.agent.conversation_manager import SummarizingConversationManager
from strands.agent.conversation_manager.conversation_manager import ProactiveCompressionConfig
from strands.experimental.context_manager import STRATEGY_PRESET_NAMES, Offload
from strands.handlers.callback_handler import null_callback_handler
from strands.models.openai import OpenAIModel

from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore Strands Agents with the following features:
- Proactive context compression (new in v1.40.0)
- SummarizingConversationManager with ProactiveCompressionConfig
- Automatic conversation summarization to stay within token limits
- The context_manager="auto" / "agentic" built-in modes
- A custom strategy pipeline from named presets and raw Offload strategies

Proactive compression monitors token usage during conversation
and automatically compresses older messages when a threshold is
reached, without waiting for context window overflow errors.
The context_manager parameter goes further: it registers a ContextManager
that runs an ordered pipeline of summarize / truncate / drop strategies
and replaces the conversation manager entirely (a co-provided
conversation_manager is ignored with a warning).

For more details, visit:
https://strandsagents.com/docs/user-guide/sdk/context-management/
-------------------------------------------------------
"""


# --- 1. Conversation manager with proactive compression ---
compression_config: ProactiveCompressionConfig = {
    "compression_threshold": 0.7,  # Compress when 70% of context window used
}

conversation_manager = SummarizingConversationManager(
    summary_ratio=0.3,  # Summarize to 30% of original content
    preserve_recent_messages=4,  # Always keep last 4 messages intact
    proactive_compression=compression_config,
)

print("=== Proactive Context Compression ===")
print(f"  compression_threshold: {compression_config['compression_threshold']}")
print(f"  summary_ratio: {conversation_manager.summary_ratio}")
print(f"  preserve_recent_messages: {conversation_manager.preserve_recent_messages}")
print()

# --- 2. Create agent with compression-enabled conversation manager ---
openai_model = OpenAIModel(
    client_args={
        "api_key": settings.OPENAI_API_KEY.get_secret_value()
        if settings.OPENAI_API_KEY
        else ""
    },
    model_id=settings.OPENAI_MODEL_NAME,
)

agent = Agent(
    model=openai_model,
    system_prompt="You are a helpful assistant. Be concise.",
    conversation_manager=conversation_manager,
    callback_handler=null_callback_handler,
)

# --- 3. Simulate a multi-turn conversation ---
questions = [
    "What is the capital of Portugal?",
    "What is its population?",
    "What language do they speak there?",
    "What's the weather like in summer?",
]

print("=== Multi-Turn Conversation ===")
print()

for q in questions:
    print(f"Q: {q}")
    response = agent(q)
    print(f"A: {response}")
    print()

# --- 4. Show conversation state ---
messages = agent.messages
print(f"=== Conversation State ===")
print(f"  Total messages in context: {len(messages)}")
print(f"  (Proactive compression keeps context manageable across long conversations)")
print()

# --- 5. context_manager="auto": a strategy-driven ContextManager ---
print("=== context_manager Shortcut ===")

auto_agent = Agent(
    model=openai_model,
    context_manager="auto",
    callback_handler=null_callback_handler,
)
print('context_manager="auto" registers a plugin that owns all context reduction:')
print(f"  context_manager: {type(auto_agent.context_manager).__name__}")
print(f"  conversation_manager: {type(auto_agent.conversation_manager).__name__} (a no-op)")
print(f"  tools: {auto_agent.tool_names}")
print()

# "agentic" hands the compression decisions to the model instead
agentic_agent = Agent(
    model=openai_model,
    context_manager="agentic",
    callback_handler=null_callback_handler,
)
print('context_manager="agentic" lets the model manage its own context:')
print(f"  tools: {agentic_agent.tool_names}")
print()

# --- 6. A custom pipeline: named presets mixed with a raw Offload strategy ---
BUILD_LOG = "\n".join(
    [f"step {i:03d} compiled module_{i:03d}.py in {10 + i % 7}ms" for i in range(1, 400)]
    + ["step 400 FAILED: test_checkout.py::test_refund timed out after 30s"]
)


@tool
def fetch_build_log(build_id: str) -> str:
    """Fetch the full CI log for a build."""
    return BUILD_LOG


print("=== Custom Strategy Pipeline ===")
print(f"Named presets: {list(STRATEGY_PRESET_NAMES)}")

pipeline_agent = Agent(
    model=openai_model,
    tools=[fetch_build_log],
    system_prompt="Answer in one short sentence.",
    # Strategies run in order; each sees the output of the previous one
    context_manager={
        "strategies": [
            "large_tool_offloading",  # tool results over 2,500 tokens -> 1,000-token preview
            "stale_tool_cleanup",
            # A pinned config, unlike the preset "proactive_summarization" (whose values may change)
            Offload.summarize("*").when(utilization=0.9, preserve_recent=2),
            "overflow_protection",
        ],
    },
    callback_handler=null_callback_handler,
)
result = pipeline_agent("Fetch the log for build b-1042. Which step failed?")

tool_result = next(
    block["toolResult"] for message in pipeline_agent.messages
    for block in message["content"] if "toolResult" in block
)
seen = tool_result["content"][0]["text"]
print(f"  fetch_build_log returned {len(BUILD_LOG):,} chars; the model saw {len(seen):,}")
print(f"  {seen.splitlines()[0]}")
print(f"  original stashed for retrieve_context: {'[Stashed]' in seen}")
print(f"A: {str(result).strip()}")
