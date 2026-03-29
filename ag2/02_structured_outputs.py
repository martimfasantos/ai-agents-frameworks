import os

from pydantic import BaseModel

from autogen import ConversableAgent, LLMConfig

from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore AG2 with the following features:
- Structured outputs using Pydantic models
- response_format parameter in LLMConfig
- Guaranteed schema-conforming JSON responses

Structured outputs force the LLM to return data matching
a Pydantic model schema, making agent outputs reliable
for downstream processing without manual parsing.

For more details, visit:
https://docs.ag2.ai/latest/docs/user-guide/basic-concepts/structured-output/
-------------------------------------------------------
"""


# --- 1. Define the Pydantic output model ---
class CityInfo(BaseModel):
    name: str
    country: str
    population: str
    famous_for: str
    best_time_to_visit: str


# --- 2. Create LLM config with response_format ---
llm_config = LLMConfig(
    {"model": settings.OPENAI_MODEL_NAME},
    response_format=CityInfo,
)

# --- 3. Create the agent ---
agent = ConversableAgent(
    name="city_expert",
    system_message=(
        "You are a city information expert. "
        "When asked about a city, provide accurate structured information."
    ),
    llm_config=llm_config,
    human_input_mode="NEVER",
)

# --- 4. Run the agent ---
result = agent.run(
    message="Tell me about Tokyo.",
    max_turns=1,
    user_input=False,
)
result.process()

# --- 5. Parse and display structured output ---
last_message = result.messages[-1]["content"]
city = CityInfo.model_validate_json(last_message)

print("\n=== Parsed Structured Output ===")
print(f"  City: {city.name}")
print(f"  Country: {city.country}")
print(f"  Population: {city.population}")
print(f"  Famous for: {city.famous_for}")
print(f"  Best time to visit: {city.best_time_to_visit}")
