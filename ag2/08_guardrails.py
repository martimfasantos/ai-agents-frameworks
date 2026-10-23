import asyncio
import os
import re

from ag2 import Agent, Context, Middleware, tool
from ag2.config import OpenAIConfig
from ag2.events import (
    BaseEvent,
    HaltEvent,
    ObserverAlert,
    Severity,
    ToolCallEvent,
    ToolResultEvent,
)
from ag2.middleware import BaseMiddleware
from ag2.observers import BaseObserver
from ag2.policies import AlertPolicy
from ag2.stream import MemoryStream
from ag2.watch import EventWatch

from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore AG2 with the following features:
- A tool middleware that vetoes a tool call before it runs
- A custom BaseObserver raising ObserverAlert(severity=FATAL)
- AlertPolicy in assembly= turning that alert into a HaltEvent

The classic Maris safeguard policies were removed in AG2 1.0. A 1.0
guardrail has two layers: on_tool_execution middleware sits between
the model and the tool, so it can refuse to run the call at all, and
an observer plus AlertPolicy raises an alert and halts the turn. The
observer alone cannot stop the tool — it only sees the call — so the
middleware is what actually keeps the data out of the CRM.

For more details, visit:
https://docs.ag2.ai/docs/user-guide/tools/tool_middleware/
-------------------------------------------------------
"""

SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
crm_records: list[str] = []


def contains_ssn(event: ToolCallEvent) -> bool:
    return bool(SSN_PATTERN.search(event.arguments))


# --- 1. The tool under supervision (simulated CRM) ---
@tool
def send_to_crm(record: str) -> str:
    """Push a customer record to the external CRM system."""
    crm_records.append(record)
    return f"[ok] pushed {len(record)} chars to the CRM"


# --- 2. Prevention: middleware that refuses to run a call carrying an SSN ---
class PiiVeto(BaseMiddleware):
    async def on_tool_execution(self, call_next, event, context):
        if contains_ssn(event):
            return ToolResultEvent.from_call(event, "BLOCKED: record contains an SSN.")
        return await call_next(event, context)


# --- 3. Alerting: an observer that flags the same calls as FATAL ---
class PiiGuardian(BaseObserver):
    """Emits a FATAL alert when a tool call carries an SSN-like value."""

    def __init__(self) -> None:
        super().__init__("pii-guardian", watch=EventWatch(ToolCallEvent))

    async def process(
        self, events: list[BaseEvent], ctx: Context
    ) -> ObserverAlert | None:
        for event in events:
            if isinstance(event, ToolCallEvent) and contains_ssn(event):
                return ObserverAlert(
                    source=self.name,
                    severity=Severity.FATAL,
                    message=f"PII detected in call to {event.name}",
                )
        return None


async def main() -> None:
    # --- 4. Subscribe to the guardrail's own events so we can prove it fired ---
    alerts: list[ObserverAlert] = []
    halts: list[HaltEvent] = []
    stream = MemoryStream()
    stream.where(ObserverAlert).subscribe(lambda e: alerts.append(e))
    stream.where(HaltEvent).subscribe(lambda e: halts.append(e))

    agent = Agent(
        "crm_operator",
        prompt=(
            "You are a CRM operator. Always push records with send_to_crm "
            "exactly as given; a guardrail screens every call. Confirm the "
            "result in one sentence."
        ),
        config=OpenAIConfig(model=settings.OPENAI_MODEL_NAME),
        tools=[send_to_crm],
        middleware=[Middleware(PiiVeto)],
        observers=[PiiGuardian()],
        assembly=[AlertPolicy()],
    )

    # --- 5. A safe record: the guardrail stays silent ---
    print("=== Request 1: safe record ===")
    reply = await agent.ask(
        "Push this record to the CRM: 'Ada Lovelace, London, tier=gold'.",
        stream=stream,
    )
    print(f"Agent: {reply.body}\n")

    # --- 6. A record with PII: the call is vetoed and the turn halted ---
    print("=== Request 2: record containing an SSN ===")
    reply2 = await agent.ask(
        "Push this record to the CRM: 'Ada Lovelace, SSN 123-45-6789'.",
        stream=stream,
    )
    print(f"Agent: {reply2.body}\n")

    # --- 7. Confirm the guardrail fired and the SSN never reached the CRM ---
    print("=== Guardrail activity ===")
    print(f"ObserverAlerts: {len(alerts)}")
    for alert in alerts:
        print(f"  - [{alert.severity.upper()}] {alert.source}: {alert.message}")
    print(f"HaltEvents:     {len(halts)}")
    for halt in halts:
        print(f"  - source={halt.source} reason={halt.reason!r}")
    print(f"CRM received:   {crm_records}")


if __name__ == "__main__":
    asyncio.run(main())
