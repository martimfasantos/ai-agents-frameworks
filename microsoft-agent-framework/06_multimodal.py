import asyncio

from dotenv import load_dotenv

from agent_framework import Agent, Content, Message
from agent_framework.openai import OpenAIChatClient

from settings import settings

load_dotenv()

"""
-------------------------------------------------------
In this example, we explore Microsoft Agent Framework
with the following features:
- Sending images to agents via Content.from_uri()
- Building multi-part messages with text and images
- Using the Message class for structured input

Multimodal input lets agents analyze images alongside
text — useful for visual Q&A, image captioning, and
document analysis workflows.

For more details, visit:
https://learn.microsoft.com/en-us/agent-framework/agents/multimodal?pivots=programming-language-python
-------------------------------------------------------
"""

# A publicly accessible sample image URL
SAMPLE_IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png"


async def main() -> None:
    # --- 1. Create the client and agent ---
    client = OpenAIChatClient(
        model=settings.OPENAI_MODEL_NAME,
        api_key=settings.OPENAI_API_KEY.get_secret_value(),
    )

    agent = client.as_agent(
        name="vision-agent",
        instructions="You are a helpful assistant that can analyze images. Be concise.",
    )

    # --- 2. Build a multimodal message with text and image ---
    message = Message(
        role="user",
        contents=[
            Content.from_text(
                "What objects are shown in this image? Describe them briefly."
            ),
            Content.from_uri(SAMPLE_IMAGE_URL, media_type="image/png"),
        ],
    )

    # --- 3. Run the agent with the multimodal message ---
    result = await agent.run(message)

    # --- 4. Print the result ---
    print("Image analysis:")
    print(result.text)


if __name__ == "__main__":
    asyncio.run(main())
