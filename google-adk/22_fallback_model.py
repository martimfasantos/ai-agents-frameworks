import os
import asyncio

from google.adk.agents import LlmAgent
from google.adk.models import FallbackModel

from settings import settings
from utils import call_agent_async

os.environ["GOOGLE_API_KEY"] = settings.GOOGLE_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore Google ADK with the following features:
- FallbackModel to keep an agent up when a model call fails
- retriable_status_codes to control which failures move on
- Failures outside that set propagating unchanged

A model endpoint can be overloaded, rate-limited or briefly down.
FallbackModel tries a list of models in order, moving to the next one when a
call fails with a retriable status. Each model is tried exactly once —
retrying a single model is that model's own retry_options, so one failure is
never retried twice over by two layers. Once a model has produced its first
response the turn belongs to it, so a mid-stream failure propagates rather
than splicing a second model onto a half-finished answer.

For more details, visit:
https://google.github.io/adk-docs/agents/models/
-------------------------------------------------------
"""

BROKEN_MODEL = "gemini-does-not-exist-9000"


async def main() -> None:
    # --- 1. A primary that will fail, with a working backup behind it ---
    # A bad model name answers 404, which is NOT in the default retriable set
    # (429/500/502/503/504), so 404 is added here to make it fall through.
    resilient_model = FallbackModel(
        models=[BROKEN_MODEL, settings.GOOGLE_MODEL_NAME],
        retriable_status_codes=frozenset({404, 429, 500, 502, 503, 504}),
    )

    agent = LlmAgent(
        name="resilient_agent",
        model=resilient_model,
        instruction="Answer in one short sentence.",
    )

    print("=== FallbackModel ===\n")
    print(f"Primary: {BROKEN_MODEL}  (will 404)")
    print(f"Backup:  {settings.GOOGLE_MODEL_NAME}\n")

    await call_agent_async(agent, "What is the capital of Portugal?")

    # --- 2. The same failure without 404 in the retriable set ---
    # Anything outside retriable_status_codes is the caller's problem, so the
    # backup is never reached.
    strict_model = FallbackModel(models=[BROKEN_MODEL, settings.GOOGLE_MODEL_NAME])

    strict_agent = LlmAgent(
        name="strict_agent",
        model=strict_model,
        instruction="Answer in one short sentence.",
    )

    print("\nSame pair, default retriable codes (404 not included):")
    try:
        await call_agent_async(strict_agent, "What is the capital of Spain?")
    except Exception as exc:
        print(f"  raised {type(exc).__name__} — the 404 propagated, no fallback")


if __name__ == "__main__":
    asyncio.run(main())
