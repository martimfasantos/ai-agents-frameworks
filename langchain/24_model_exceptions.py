import os

from langchain_core.exceptions import (
    ModelAuthenticationError,
    ModelError,
    ModelNotFoundError,
    ModelRateLimitError,
)
from langchain_openai import ChatOpenAI

from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-----------------------------------------------------------------------
In this example, we explore LangChain with the following features:
- Standard model exception types, added in LangChain 1.3.16
- Catching ModelNotFoundError and ModelAuthenticationError by type
- The shared ModelError base for provider-agnostic handling

Before these existed, telling a bad API key from a bad model name from a
rate limit meant matching on provider-specific error strings, and the
matching broke whenever a provider reworded a message. The standard
types are raised by every integration, so one except block handles the
same failure across OpenAI, Anthropic and the rest.

For more details, visit:
https://docs.langchain.com/oss/python/langchain/models
-----------------------------------------------------------------------
"""

PROMPT = "Reply with the single word: ok"


# --- 1. A model name that does not exist ---
print("=== Standard model exceptions ===\n")

try:
    ChatOpenAI(model="gpt-does-not-exist-9000").invoke(PROMPT)
except ModelNotFoundError as exc:
    print("ModelNotFoundError")
    print(f"  is a ModelError:  {isinstance(exc, ModelError)}")
    print(f"  message:          {str(exc).splitlines()[0][:110]}")

# --- 2. A malformed API key ---
try:
    ChatOpenAI(
        model=settings.OPENAI_MODEL_NAME, api_key="sk-not-a-real-key"
    ).invoke(PROMPT)
except ModelAuthenticationError as exc:
    print("\nModelAuthenticationError")
    print(f"  is a ModelError:  {isinstance(exc, ModelError)}")
    print(f"  message:          {str(exc).splitlines()[0][:110]}")

# --- 3. One handler for any model failure ---
# Catching the base class keeps provider-agnostic retry/fallback logic in
# one place; the subclass still says which failure it was.
def ask(model_name: str, api_key: str | None = None) -> str:
    """Call a model, reporting failures by their standard type."""
    try:
        model = ChatOpenAI(model=model_name, api_key=api_key) if api_key else ChatOpenAI(model=model_name)
        return model.invoke(PROMPT).text
    except ModelRateLimitError:
        return "rate limited — back off and retry"
    except ModelError as exc:
        return f"unavailable ({type(exc).__name__}) — fall back to another model"


print("\nprovider-agnostic handler:")
print(f"  bad model name: {ask('gpt-does-not-exist-9000')}")
print(f"  bad api key:    {ask(settings.OPENAI_MODEL_NAME, 'sk-not-a-real-key')}")
print(f"  working model:  {ask(settings.OPENAI_MODEL_NAME)}")
