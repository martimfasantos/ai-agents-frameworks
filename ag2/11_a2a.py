import asyncio
import os
import threading
import time

import uvicorn

from autogen import ConversableAgent, LLMConfig
from autogen.a2a.client import A2aRemoteAgent
from autogen.a2a.server import A2aAgentServer

from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore AG2 with the following features:
- A2A (Agent-to-Agent) protocol for remote agent communication
- A2aAgentServer to expose an agent as an A2A-compatible service
- A2aRemoteAgent to connect to a remote A2A agent

The A2A protocol enables agents to communicate across
network boundaries. A2aAgentServer wraps an AG2 agent as
an ASGI web service, and A2aRemoteAgent connects to it as
a client. This enables distributed multi-agent systems.

For more details, visit:
https://docs.ag2.ai/latest/docs/user-guide/a2a/
-------------------------------------------------------
"""

PORT = 18765


def run_server() -> None:
    """Run the A2A server in a background thread."""
    # --- 1. Create the server-side agent ---
    llm_config = LLMConfig({"model": settings.OPENAI_MODEL_NAME})

    server_agent = ConversableAgent(
        name="translator",
        system_message=(
            "You are a language translator. When given text, translate it "
            "to French. Only output the French translation, nothing else."
        ),
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    # --- 2. Wrap the agent as an A2A server ---
    a2a_server = A2aAgentServer(
        agent=server_agent,
        url=f"http://localhost:{PORT}",
    )

    app = a2a_server.build_starlette_app()

    # Run uvicorn in this thread (blocking)
    config = uvicorn.Config(app, host="0.0.0.0", port=PORT, log_level="warning")
    server = uvicorn.Server(config)
    server.run()


async def run_client() -> None:
    """Connect to the A2A server and send a request."""
    # --- 3. Create a remote agent pointing to the server ---
    remote_agent = A2aRemoteAgent(
        url=f"http://localhost:{PORT}",
        name="remote_translator",
    )

    # --- 4. Create a local user agent ---
    user = ConversableAgent(
        name="user",
        human_input_mode="NEVER",
        llm_config=False,
        is_termination_msg=lambda x: True,
    )

    # --- 5. Chat with the remote agent via A2A ---
    print("Sending request to remote A2A agent...\n")
    result = await user.a_initiate_chat(
        remote_agent,
        message="Hello, how are you today?",
        max_turns=1,
    )

    print(f"\n=== Translation Result ===")
    print(f"Original: Hello, how are you today?")
    print(f"French:   {result.summary}")


def main() -> None:
    """Run the A2A demo."""
    print("=== A2A: Agent-to-Agent Protocol ===\n")

    # Start the server in a background thread
    print(f"Starting A2A server on port {PORT}...")
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    # Wait for the server to start
    time.sleep(3)
    print("Server ready.\n")

    # Run the client
    asyncio.run(run_client())

    print("\n=== A2A Demo Complete ===")


if __name__ == "__main__":
    main()
