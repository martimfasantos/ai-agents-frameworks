import asyncio

from dotenv import load_dotenv

from pydantic_ai import (
    Agent,
    ModelAPIError,
    ModelContextWindowExceededError,
    ModelHTTPError,
    ModelMessage,
    ModelOverloadedError,
    ModelQuotaExceededError,
    ModelRateLimitError,
    ModelServerError,
    ModelUnavailableError,
)
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import AgentInfo, FunctionModel

from settings import settings

load_dotenv()

"""
-----------------------------------------------------------------------
In this example, we explore Pydantic AI with the following features:
- Provider-agnostic error categories: ModelRateLimitError, ModelQuotaExceededError,
  ModelUnavailableError/ModelOverloadedError, ModelServerError,
  ModelContextWindowExceededError
- ModelHTTPError.for_category() to raise a categorized error from a custom model
- FallbackModel(fallback_on=(ModelRateLimitError,)) to fall back on one category only
- ModelResponse.failed_attempts, the record of the models a fallback moved on from

Every model adapter raises the same categories whichever provider failed,
so "back off on a rate limit" or "compact the history on a context
overflow" is one except clause. Each HTTP error is both a ModelHTTPError
and its category, so status-code handlers keep working. Provider errors
can't be triggered on demand, so a FunctionModel raises them here: every
[SIMULATED] error comes from it; only the fallback answer is real.

For more details, visit:
https://pydantic.dev/docs/ai/models/overview/#handling-model-api-errors
-----------------------------------------------------------------------
"""


# --- 1. A fake model that fails the way a real provider would ---
def failing_model(category: type[ModelAPIError], status_code: int, **kwargs) -> FunctionModel:
    """A FunctionModel that always raises the given categorized HTTP error."""

    def respond(messages: list[ModelMessage], info: AgentInfo):
        raise ModelHTTPError.for_category(
            category, status_code=status_code, model_name="flaky-model", **kwargs
        )

    return FunctionModel(respond, model_name="flaky-model")


# --- 2. One handler for every provider, keyed on the category ---
def advice_for(exc: ModelAPIError) -> str:
    # Order matters: ModelOverloadedError is a subclass of ModelUnavailableError
    if isinstance(exc, ModelRateLimitError):
        return f"back off for {exc.retry_after}s and retry"
    if isinstance(exc, ModelQuotaExceededError):
        return "top up billing -- waiting won't help"
    if isinstance(exc, ModelOverloadedError):
        return "provider overloaded -- try another model"
    if isinstance(exc, ModelUnavailableError):
        return "provider down -- try another model"
    if isinstance(exc, ModelServerError):
        return "provider bug -- retry once, then alert"
    if isinstance(exc, ModelContextWindowExceededError):
        return "compact the message history and retry"
    return "unhandled"


async def main() -> None:

    # ------------------------------------------------------------------
    # Example 1: One handler covers every category
    # ------------------------------------------------------------------
    print("=== Example 1: One handler for all categories [SIMULATED] ===")

    simulated = [
        (ModelRateLimitError, 429, {"retry_after": 2.0}),
        (ModelQuotaExceededError, 402, {}),
        (ModelOverloadedError, 529, {}),
        (ModelUnavailableError, 503, {}),
        (ModelServerError, 500, {}),
        (ModelContextWindowExceededError, 400, {"provider_error_code": "context_length_exceeded"}),
    ]
    for category, status_code, kwargs in simulated:
        try:
            await Agent(failing_model(category, status_code, **kwargs)).run("Hi")
        except ModelHTTPError as exc:
            # A plain ModelHTTPError handler still receives the specific category
            print(f"  HTTP {exc.status_code} {type(exc).__name__:<32} -> {advice_for(exc)}")
    print()

    # ------------------------------------------------------------------
    # Example 2: Fall back to a real model only on rate limits
    # ------------------------------------------------------------------
    print("=== Example 2: FallbackModel(fallback_on=(ModelRateLimitError,)) ===")

    fallback = FallbackModel(
        failing_model(ModelRateLimitError, 429, retry_after=2.0),
        settings.OPENAI_MODEL_NAME,
        fallback_on=(ModelRateLimitError,),
    )
    result = await Agent(fallback, instructions="Reply in one sentence.").run(
        "What is the capital of Portugal?"
    )
    print(f"Response (REAL, from {result.response.model_name}): {result.output}")
    # The answering response records every model the fallback skipped
    for attempt in result.response.failed_attempts or []:
        print(f"  failed attempt: model={attempt.model_name!r} outcome={attempt.outcome!r}")
        print(f"    error: {attempt.error}")
    print()

    # ------------------------------------------------------------------
    # Example 3: Categories outside fallback_on are raised as-is
    # ------------------------------------------------------------------
    print("=== Example 3: A quota error does not fall back [SIMULATED 402] ===")

    # Same fallback chain, but the primary now fails with exhausted quota
    fallback = FallbackModel(
        failing_model(ModelQuotaExceededError, 402),
        settings.OPENAI_MODEL_NAME,
        fallback_on=(ModelRateLimitError,),
    )
    try:
        await Agent(fallback).run("What is the capital of Portugal?")
    except ModelQuotaExceededError as exc:
        print(f"Caught {type(exc).__name__} from {exc.model_name!r}")
        print(f"  The real model was never called -> {advice_for(exc)}")


if __name__ == "__main__":
    asyncio.run(main())
