import os

from crewai import LLM, Agent, Crew, Task
from crewai.events import (
    BaseEventListener,
    ContextWindowSummarizationCompletedEvent,
    ContextWindowSummarizationStartedEvent,
    LLMCallStartedEvent,
    ToolUsageFinishedEvent,
)
from crewai.tools import tool

from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore CrewAI with the following features:
- llm.get_context_window_size(): the usable window, 85% of the advertised one
- llm.get_input_tokens_count(text): a provider-side token count
- respect_context_window + use_better_compaction on an Agent
- Oversized tool results deferred to a temp file, read back with DeferredToolResultReader
- ContextWindowSummarizationStartedEvent / CompletedEvent on the event bus

A tool that returns a huge payload (a log dump, a full table export) can crowd
everything else out of the conversation. With use_better_compaction, any tool
result above 25% of the usable window is written to a temporary artifact: the
model sees a short preview plus the path, and a built-in reader tool lets it
page through the file. If the conversation itself nears the window,
respect_context_window summarizes it before the next model call.

For more details, visit:
https://docs.crewai.com/en/concepts/agents#context-window-management
-------------------------------------------------------
"""

# --- 1. Inspect the model's context budget ---
llm = LLM(model=settings.OPENAI_MODEL_NAME)
usable_window = llm.get_context_window_size()
# use_better_compaction defers any tool result above 25% of the usable window
deferral_threshold = usable_window // 4

print("=== 1. Context budget ===")
print(f"Usable context window for {llm.model}: {usable_window:,} tokens (85% of advertised)")
print(f"Tool results above 25% of that are deferred: > {deferral_threshold:,} tokens")


# --- 2. Build a large log dump with one ERROR line buried inside ---
SERVICES = ["checkout-svc", "cart-svc", "auth-svc", "search-svc", "inventory-svc"]
TOTAL_LINES = 700
ERROR_LINE = 431


def build_log_dump() -> str:
    lines = []
    for i in range(1, TOTAL_LINES + 1):
        timestamp = f"2026-10-23T08:{(i // 60) % 60:02d}:{i % 60:02d}.{i % 1000:03d}Z"
        service = SERVICES[i % len(SERVICES)]
        if i == ERROR_LINE:
            lines.append(
                f"{timestamp} ERROR [payments-svc] request_id=req-{i:05d} "
                "card authorization timed out after 30000ms (gateway=stripe-eu)"
            )
        else:
            lines.append(
                f"{timestamp} INFO  [{service}] request_id=req-{i:05d} "
                f"GET /api/v1/items/{i * 7 % 997} status=200 latency={i % 90 + 5}ms"
            )
    return "\n".join(lines)


LOG_DUMP = build_log_dump()
log_tokens = llm.get_input_tokens_count(LOG_DUMP)
print(f"\nLog dump: {TOTAL_LINES:,} lines, {log_tokens:,} tokens (counted by the provider)")
print(f"Over the deferral threshold: {log_tokens > deferral_threshold}")


@tool("fetch_service_logs")
def fetch_service_logs(window: str) -> str:
    """Fetch the raw production logs for a time window, e.g. 'last_hour'."""
    return LOG_DUMP


# --- 3. Watch tool calls, the deferral notice, and summarization events ---
class ContextWatcher(BaseEventListener):
    def __init__(self) -> None:
        super().__init__()
        self.tools_used: list[str] = []
        self.summarizations = 0
        self.deferral_notice_shown = False

    def setup_listeners(self, crewai_event_bus):
        @crewai_event_bus.on(ToolUsageFinishedEvent)
        def on_tool_finish(source, event):
            self.tools_used.append(event.tool_name)
            print(
                f"[TOOL] {event.tool_name} args={event.tool_args} "
                f"-> {len(str(event.output)):,} chars"
            )

        @crewai_event_bus.on(LLMCallStartedEvent)
        def on_llm_call(source, event):
            # Show what the model received instead of the raw log dump, once.
            if self.deferral_notice_shown or not isinstance(event.messages, list):
                return
            for message in event.messages:
                content = str(message.get("content") or "")
                if message.get("role") == "tool" and "temporary artifact" in content:
                    self.deferral_notice_shown = True
                    first_line = content.splitlines()[0]
                    last_line = content.splitlines()[-1]
                    print(f"[DEFERRED] Model sees ({len(content):,} chars): {first_line}")
                    print(f"[DEFERRED] ...{last_line}")

        @crewai_event_bus.on(ContextWindowSummarizationStartedEvent)
        def on_summarization_start(source, event):
            self.summarizations += 1
            print(f"[SUMMARIZE] Started for {event.agent_role} on {event.model}")

        @crewai_event_bus.on(ContextWindowSummarizationCompletedEvent)
        def on_summarization_done(source, event):
            print(f"[SUMMARIZE] Completed for {event.agent_role}")


watcher = ContextWatcher()

# --- 4. An agent with both context-window features enabled (the defaults) ---
agent = Agent(
    role="SRE On-Call",
    goal="Find the root-cause error line in production logs",
    backstory=(
        "You fetch logs once. If the result was stored as an artifact, you read "
        "it from line 1 in consecutive sections of the suggested size and stop "
        "at the first section that contains an ERROR line. You answer in one sentence."
    ),
    tools=[fetch_service_logs],
    llm=llm,
    respect_context_window=True,
    use_better_compaction=True,
)

task = Task(
    description=(
        "Fetch the logs for the last hour and find the single ERROR line. "
        "Report its timestamp, service and message."
    ),
    expected_output="One sentence with the timestamp, service and error message.",
    agent=agent,
)

# --- 5. Run the crew and show how the oversized result was handled ---
print("\n=== 2. Agent run with use_better_compaction=True ===")
crew = Crew(agents=[agent], tasks=[task])
result = crew.kickoff()

print(f"\nAnswer: {result.raw}")
print(
    f"fetch_service_logs calls: {watcher.tools_used.count('fetch_service_logs')}, "
    f"deferred-result reads: {watcher.tools_used.count('read_a_deferred_tool_result')}"
)
print(f"Summarization events: {watcher.summarizations}")
if watcher.summarizations == 0:
    print(
        "(Summarization only runs when a pending request exceeds the usable window; "
        "deferral kept this run well below it.)"
    )
