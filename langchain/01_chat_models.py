import os

from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI

from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore LangChain with the following features:
- Initializing chat models with init_chat_model() and ChatOpenAI()
- Using invoke() for single completions
- Working with message types (system, human, AI)
- Building multi-turn conversations manually

LangChain supports two ways to create chat models: the universal
init_chat_model() factory that works with any provider, and
provider-specific classes like ChatOpenAI. Both read the API key
from the environment.

For more details, visit:
https://docs.langchain.com/oss/python/langchain/models
-------------------------------------------------------
"""

# --------------------------------------------------------------
# Example 1: init_chat_model (universal factory)
# --------------------------------------------------------------
print("=== Example 1: init_chat_model ===")

# --- 1. Create a model using the universal factory ---
model = init_chat_model(f"openai:{settings.OPENAI_MODEL_NAME}")

# --- 2. Invoke with a simple string (becomes a HumanMessage) ---
response = model.invoke("What is the capital of Portugal?")
print(f"Response: {response.content}\n")

# --------------------------------------------------------------
# Example 2: ChatOpenAI (provider-specific)
# --------------------------------------------------------------
print("=== Example 2: ChatOpenAI ===")

# --- 3. Create a model using the provider-specific class ---
model_openai = ChatOpenAI(model=settings.OPENAI_MODEL_NAME)

# --- 4. Invoke with message dicts ---
response = model_openai.invoke(
    [
        {"role": "system", "content": "You are a geography expert. Be concise."},
        {"role": "user", "content": "What is the capital of Japan?"},
    ]
)
print(f"Response: {response.content}\n")

# --------------------------------------------------------------
# Example 3: Multi-turn conversation
# --------------------------------------------------------------
print("=== Example 3: Multi-turn Conversation ===")

# --- 5. Build a multi-turn conversation manually ---
messages = [
    {"role": "system", "content": "You are a helpful math tutor. Be concise."},
    {"role": "user", "content": "What is 2 + 2?"},
]

response_1 = model.invoke(messages)
print(f"Turn 1: {response_1.content}")

# Append the AI response and a follow-up question
messages.append({"role": "assistant", "content": response_1.content})
messages.append({"role": "user", "content": "Now multiply that by 10."})

response_2 = model.invoke(messages)
print(f"Turn 2: {response_2.content}")
