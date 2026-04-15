import asyncio

from dotenv import load_dotenv

from agent_framework import AgentResponse, tool
from agent_framework.openai import OpenAIChatClient

from settings import settings

load_dotenv()

"""
-------------------------------------------------------
In this example, we explore Microsoft Agent Framework with the following features:
- Token usage tracking via AgentResponse.usage_details
- UsageDetails TypedDict with input_token_count, output_token_count, total_token_count
- Fallback: scanning message contents for usage Content items

Microsoft Agent Framework exposes token usage on AgentResponse
through the usage_details property, which is a UsageDetails
TypedDict. This provides input, output, and total token counts
for cost analysis. Usage data may also appear as Content items
within response messages.

For more details, visit:
https://github.com/microsoft/agent-framework
-------------------------------------------------------
"""


# --- 1. Define tools ---
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: The city name.

    Returns:
        A weather report string.
    """
    weather_data = {
        "london": "Cloudy, 14°C, light rain expected",
        "tokyo": "Sunny, 28°C, clear skies",
        "new york": "Partly cloudy, 22°C",
    }
    return weather_data.get(city.lower(), f"No weather data for {city}")


# --- 2. Create the client and agent ---
client = OpenAIChatClient(
    model_id=settings.OPENAI_MODEL_NAME,
    api_key=settings.OPENAI_API_KEY.get_secret_value(),
)

agent = client.as_agent(
    name="weather_agent",
    instructions="You are a concise weather assistant. Answer in 1-2 sentences.",
    tools=[get_weather],
)


# --- 3. Run and extract usage ---
async def main():
    print("=== Microsoft Agent Framework Token Usage ===\n")

    print("--- Running agent ---")
    result: AgentResponse = await agent.run("What's the weather in London and Tokyo?")

    print(f"Response: {result.text}\n")

    # --- 4. Extract usage from AgentResponse.usage_details ---
    print("--- Usage Details (AgentResponse.usage_details) ---")
    usage = getattr(result, "usage_details", None)

    if usage and isinstance(usage, dict):
        input_tok = usage.get("input_token_count", 0) or 0
        output_tok = usage.get("output_token_count", 0) or 0
        total_tok = usage.get("total_token_count", 0) or 0
        print(f"  Input tokens:  {input_tok}")
        print(f"  Output tokens: {output_tok}")
        print(f"  Total tokens:  {total_tok}")

        # Show any extra provider-specific fields
        for key, value in usage.items():
            if key not in (
                "input_token_count",
                "output_token_count",
                "total_token_count",
            ):
                print(f"  {key}: {value}")
    else:
        print("  No usage_details on response")

    # --- 5. Fallback: scan message contents for usage items ---
    print("\n--- Scanning Message Contents for Usage ---")
    found_usage = False
    for msg in getattr(result, "messages", []):
        for content_item in getattr(msg, "contents", []):
            ct = getattr(content_item, "type", "")
            if ct == "usage":
                found_usage = True
                ud = getattr(content_item, "usage_details", None)
                if ud and isinstance(ud, dict):
                    print(f"  Content usage item found:")
                    print(f"    Input tokens:  {ud.get('input_token_count', 0)}")
                    print(f"    Output tokens: {ud.get('output_token_count', 0)}")
                    print(f"    Total tokens:  {ud.get('total_token_count', 0)}")

    if not found_usage:
        print("  No usage Content items in messages")

    # --- 6. Inspect message structure ---
    print("\n--- Response Message Structure ---")
    for i, msg in enumerate(getattr(result, "messages", [])):
        author = getattr(msg, "author_name", "unknown")
        content_count = len(getattr(msg, "contents", []))
        print(f"  Message {i}: author={author}, contents={content_count}")
        for j, ci in enumerate(getattr(msg, "contents", [])):
            ct = getattr(ci, "type", "unknown")
            print(f"    Content {j}: type={ct}")

    print("\n=== Token Usage Demo Complete ===")


if __name__ == "__main__":
    asyncio.run(main())
