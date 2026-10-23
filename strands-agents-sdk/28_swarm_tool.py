import asyncio

from strands import Agent
from strands.hooks import AfterToolCallEvent, BeforeToolCallEvent
from strands.models.openai import OpenAIModel
from strands.multiagent import Fixed, Preset
from strands.vended_tools import make_swarm

from settings import settings

"""
-------------------------------------------------------
In this example, we explore Strands Agents SDK with the following features:
- The vended swarm tool, configured with make_swarm() (new in v1.62.0)
- Presets: named roles the model picks for each agent via agent_type
- instructions=Fixed() so the model cannot override a preset's prompt
- Developer-pinned limits: max_agents, max_handoffs, execution_timeout
- Watching the swarm's handoffs stream out of the tool call

In 11_multi_agent_swarm.py the developer builds the team by hand. Here the
model designs the team at runtime: it calls the swarm tool with a list of
agents (name + preset), the tool builds them, runs a Swarm where they hand
off to each other, and returns each agent's final output. The developer still
controls which roles exist and how far the team can run.

For more details, visit:
https://strandsagents.com/docs/user-guide/sdk/tools/vended-tools/#swarm
-------------------------------------------------------
"""

# --- 1. The swarm tool: two roles and hard limits ---
swarm = make_swarm(
    presets={
        "researcher": Preset(
            instructions=(
                "You are the researcher. Note at most 2 short facts per option, then you MUST "
                "call handoff_to_agent to pass them to the writer. Never write the final answer."
            ),
            tools=[],
            description="gathers key facts, then hands off",
        ),
        "writer": Preset(
            instructions=(
                "You are the writer and the last agent: never call handoff_to_agent. Reply with "
                "exactly 3 one-line bullet points that compare the options."
            ),
            tools=[],
            description="writes the final short comparison",
        ),
    },
    instructions=Fixed(),  # hide the free-form prompt: children only get preset instructions
    max_agents=3,  # the model may create at most 3 agents per call
    max_handoffs=4,
    execution_timeout=120.0,
)


# --- 2. Hooks show the team the model asked for and what came back ---
def on_swarm_start(event: BeforeToolCallEvent) -> None:
    args = event.tool_use["input"]
    print(f"[swarm] task: {args['task']}")
    for spec in args["agents"]:
        agent_type = spec.get("agent_type") or "omitted -> default preset"
        print(f"  [agent] {spec['name']} (agent_type={agent_type})")
    print(f"  [entry] {args.get('entry_point') or args['agents'][0]['name']}\n")


def on_swarm_end(event: AfterToolCallEvent) -> None:
    print(f"\n[swarm] tool status: {event.result['status']}")


# --- 3. The manager agent only has the swarm tool ---
openai_model = OpenAIModel(
    client_args={
        "api_key": settings.OPENAI_API_KEY.get_secret_value()
        if settings.OPENAI_API_KEY
        else ""
    },
    model_id=settings.OPENAI_MODEL_NAME,
)
# Default: Agent() uses Amazon Bedrock (requires AWS credentials)
manager = Agent(
    model=openai_model,
    tools=[swarm],
    callback_handler=None,
    system_prompt=(
        "You coordinate work. For comparison tasks, call the swarm tool once with a small team: "
        "one researcher and one writer. Then reply with the writer's 3 bullet points only, "
        "no preamble."
    ),
)
manager.add_hook(on_swarm_start)
manager.add_hook(on_swarm_end)


# --- 4. Stream the run and print the swarm's node starts and handoffs ---
async def main() -> None:
    print("=== Swarm tool: a model-designed agent team ===\n")
    path: list[str] = []
    final = None
    async for event in manager.stream_async(
        "Compare SQLite and PostgreSQL for a small internal web app."
    ):
        data = event.get("tool_stream_event", {}).get("data")
        if isinstance(data, dict) and data.get("type") == "multiagent_node_start":
            path.append(data["node_id"])
            print(f"  [node start] {data['node_id']}")
        elif isinstance(data, dict) and data.get("type") == "multiagent_handoff":
            print(f"  [handoff]    {data['from_node_ids'][0]} -> {data['to_node_ids'][0]}")
        elif "result" in event:
            final = event["result"]

    print(f"Handoff path: {' -> '.join(path)}")
    print(f"\nManager:\n{str(final).strip()}")


if __name__ == "__main__":
    asyncio.run(main())
