import asyncio
from typing import Sequence

from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import (
    BaseChatMessage,
    BaseTextChatMessage,
    TextMessage,
)
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken

from utils import print_new_section

"""
------------------------------------------------------------------------
In this example, we explore Autogen agents with the following features:
- Custom agent by subclassing BaseChatAgent
- Implementing on_messages() and on_reset()
- Defining produced_message_types property

This example shows how to create a fully custom agent by subclassing
BaseChatAgent. Custom agents can implement arbitrary logic in their
on_messages() method — from simple stateful counters to complex
reasoning chains. The framework only requires you to implement
on_messages(), on_reset(), and the produced_message_types property.

For more details, visit:
https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/custom-agents.html
------------------------------------------------------------------------
"""


# --- 1. Define a custom CountDownAgent ---
class CountDownAgent(BaseChatAgent):
    """An agent that counts down from a given number to zero."""

    def __init__(self, name: str, count: int) -> None:
        super().__init__(
            name=name, description=f"An agent that counts down from {count}."
        )
        self._count = count

    @property
    def produced_message_types(self) -> list[type[BaseChatMessage]]:
        return [TextMessage]

    async def on_messages(
        self,
        messages: Sequence[BaseChatMessage],
        cancellation_token: CancellationToken,
    ) -> Response:
        # Decrement the counter
        self._count -= 1
        if self._count > 0:
            content = f"{self._count}..."
        else:
            content = "Liftoff! 🚀"
        return Response(chat_message=TextMessage(content=content, source=self.name))

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        """Reset is not needed for this simple agent."""
        pass


# --- 2. Define a custom EchoAgent ---
class EchoAgent(BaseChatAgent):
    """An agent that echoes the last message it received, reversed."""

    def __init__(self, name: str) -> None:
        super().__init__(name=name, description="Echoes messages back reversed.")

    @property
    def produced_message_types(self) -> list[type[BaseChatMessage]]:
        return [TextMessage]

    async def on_messages(
        self,
        messages: Sequence[BaseChatMessage],
        cancellation_token: CancellationToken,
    ) -> Response:
        # Get the last text message
        last_text = ""
        for msg in reversed(messages):
            if isinstance(msg, BaseTextChatMessage):
                last_text = msg.content
                break
        reversed_text = last_text[::-1]
        return Response(
            chat_message=TextMessage(
                content=f"Echo (reversed): {reversed_text}", source=self.name
            )
        )

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        pass


async def main() -> None:
    # ============================================================
    # --- 3. Run the CountDownAgent ---
    # ============================================================

    print_new_section("1. CountDownAgent")
    countdown = CountDownAgent("countdown", count=5)

    # Run it multiple times to see the countdown
    for i in range(5):
        response = await countdown.on_messages([], CancellationToken())
        print(response.chat_message.content)

    # ============================================================
    # --- 4. Run the EchoAgent ---
    # ============================================================

    print_new_section("2. EchoAgent")
    echo = EchoAgent("echo")

    test_messages = [
        TextMessage(content="Hello, World!", source="user"),
        TextMessage(content="Autogen is great", source="user"),
    ]

    for msg in test_messages:
        response = await echo.on_messages([msg], CancellationToken())
        print(f"Input:  {msg.content}")
        print(f"Output: {response.chat_message.content}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
