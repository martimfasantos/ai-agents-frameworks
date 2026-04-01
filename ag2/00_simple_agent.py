import os

from autogen import ConversableAgent, LLMConfig

from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore AG2 with the following features:
- Creating a simple ConversableAgent
- Using LLMConfig with the new dict-based configuration
- Running a single-turn conversation with run() and process()

AG2 (formerly AutoGen) is a multi-agent framework where
ConversableAgent is the core building block. This example
shows the simplest possible agent interaction.

For more details, visit:
https://docs.ag2.ai/latest/docs/user-guide/basic-concepts/conversable-agent/
-------------------------------------------------------
"""

# --- 1. Configure the LLM ---
llm_config = LLMConfig({"model": settings.OPENAI_MODEL_NAME})

# --- 2. Create a simple agent ---
agent = ConversableAgent(
    name="assistant",
    system_message="You are a helpful assistant. Be concise, reply in 1-2 sentences.",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

# --- 3. Run the agent and print the conversation ---
result = agent.run(
    message="Where does the phrase 'hello world' come from?",
    max_turns=1,
    user_input=False,
)

# process() prints the conversation to stdout
result.process()

# --- 4. Print the summary ---
print("\n=== Summary ===")
print(result.summary)
