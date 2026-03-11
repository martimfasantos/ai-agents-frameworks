import os
import asyncio

from agents import Agent, RunHooks, Runner

from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore OpenAI's Agents SDK with the following features:
- Lifecycle hooks with RunHooks
- Observing agent start/end, LLM calls, and tool invocations

RunHooks let you attach callbacks that fire at key points during a
Runner.run() invocation — agent start/end, LLM start/end, tool
start/end, and handoffs.  This is useful for logging, metrics,
pre-fetching data, or recording usage.

For more details, visit:
https://openai.github.io/openai-agents-python/agents/#lifecycle-events-hooks
-------------------------------------------------------
"""


# --- 1. Define a custom RunHooks class ---
class LoggingHooks(RunHooks):
    """Logs lifecycle events to the console."""

    async def on_agent_start(self, context, agent) -> None:
        print(f"[hook] Agent '{agent.name}' started")

    async def on_agent_end(self, context, agent, output) -> None:
        print(f"[hook] Agent '{agent.name}' finished — usage so far: {context.usage}")

    async def on_llm_start(
        self, context, agent, system_prompt=None, input_items=None
    ) -> None:
        print(f"[hook] LLM call starting for '{agent.name}'")

    async def on_llm_end(self, context, agent, response) -> None:
        print(
            f"[hook] LLM call ended for '{agent.name}' "
            f"— {len(response.output)} output items"
        )

    async def on_tool_start(self, context, agent, tool) -> None:
        print(f"[hook] Tool '{tool.name}' starting on agent '{agent.name}'")

    async def on_tool_end(self, context, agent, tool, result) -> None:
        print(f"[hook] Tool '{tool.name}' finished on agent '{agent.name}'")

    async def on_handoff(self, context, from_agent, to_agent) -> None:
        print(f"[hook] Handoff from '{from_agent.name}' to '{to_agent.name}'")


# --- 2. Define a simple agent ---
agent = Agent(
    name="Assistant",
    instructions="Be concise. Reply with one sentence.",
    model=settings.OPENAI_MODEL_NAME,
)


async def main() -> None:
    # --- 3. Run the agent with hooks ---
    result = await Runner.run(
        agent,
        "What is the capital of Portugal?",
        hooks=LoggingHooks(),
    )
    print(f"\nFinal output: {result.final_output}")


if __name__ == "__main__":
    asyncio.run(main())
