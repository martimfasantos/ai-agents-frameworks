import os

from strands import Agent, tool
from strands.hooks import AfterToolCallEvent, BeforeToolCallEvent
from strands.models.openai import OpenAIModel
from strands.multiagent import AgentSpec, Choice, Option, Preset
from strands.vended_tools.subagent import make_subagent

from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore Strands Agents with the following features:
- The vended subagent tool, configured with make_subagent()
- Presets: named child roles with their own instructions and tool subset
- A model Choice whose Option values are model instances
- A custom builder and max_depth to control how children are built and nested

A subagent runs a self-contained task in a fresh context and returns only
its final report, so the manager's context is not flooded with the child's
intermediate tool calls. Axis policies decide what the manager may set on
each child: a preset fixes the role, a Choice exposes a closed set of
options, and anything left unset is inherited.

For more details, visit:
https://strandsagents.com/docs/user-guide/sdk/tools/vended-tools/#subagent
-------------------------------------------------------
"""


def openai_model(model_id: str) -> OpenAIModel:
    return OpenAIModel(
        client_args={"api_key": settings.OPENAI_API_KEY.get_secret_value()},
        model_id=model_id,
    )


fast_model = openai_model(settings.OPENAI_MODEL_NAME)
strong_model = openai_model(settings.OPENAI_STRONG_MODEL_NAME)

# --- 1. Mock order data, readable only by the analyst role ---
ORDERS = {
    "A-1001": {"customer": "Ana", "status": "delivered", "days_late": 0},
    "A-1002": {"customer": "Bruno", "status": "in transit", "days_late": 4},
    "A-1003": {"customer": "Carla", "status": "in transit", "days_late": 0},
    "A-1004": {"customer": "Diogo", "status": "lost in transit", "days_late": 9},
}


@tool
def list_orders() -> str:
    """List every order with its customer, status and how many days late it is."""
    return "\n".join(
        f"{order_id} ({o['customer']}): {o['status']}, {o['days_late']} days late"
        for order_id, o in ORDERS.items()
    )


# --- 2. A builder that turns each resolved spec into a child agent ---
# The default builder grants tools from the parent's registry; this one keeps
# the order tools out of the manager's hands entirely.
CHILD_TOOLS = {"list_orders": list_orders}


def build_child(spec: AgentSpec) -> Agent:
    model = spec.model or fast_model
    tools = [CHILD_TOOLS[name] for name in spec.tools or []]
    print(f"  [spawn]  {spec.agent_type} on {model.get_config()['model_id']}, tools={spec.tools}")
    return Agent(model=model, system_prompt=spec.instructions, tools=tools, callback_handler=None)


# --- 3. The subagent tool: two presets and a model choice ---
subagent = make_subagent(
    builder=build_child,
    presets={
        "order_analyst": Preset(
            instructions=(
                "Answer from the order tools in one plain-text line, no markdown, e.g. "
                "'A-1002 (Bruno): in transit, 4 days late'."
            ),
            tools=["list_orders"],
            description="looks up order data; the only role with data access",
        ),
        "support_writer": Preset(
            instructions="Write one paragraph of at most two sentences, no greeting or sign-off.",
            tools=[],
            description="drafts customer emails; has no data access, so put every fact in the task",
        ),
    },
    model=Choice([
        Option("fast", fast_model, "cheap model for lookups"),
        Option("strong", strong_model, "stronger model for customer-facing writing"),
    ]),
    max_depth=1,  # one level of delegation: children can never spawn their own subagents
)


# --- 4. Hooks that show what the manager delegated and what came back ---
def on_delegate(event: BeforeToolCallEvent) -> None:
    args = event.tool_use["input"]
    print(f"[delegate] agent_type={args.get('agent_type')} model={args.get('model', 'inherited')}")
    print(f"  [task]   {args['task']}")


def on_report(event: AfterToolCallEvent) -> None:
    print(f"  [report] {event.result['content'][0]['text'].strip()}\n")


manager = Agent(
    model=fast_model,
    tools=[subagent],
    callback_handler=None,
    system_prompt=(
        "You are a support manager with no data access. Delegate one task at a time: first an "
        "order_analyst on the fast model for the facts, then, once it reports, a support_writer "
        "on the strong model whose task includes the customer name, order id and delay. Keep "
        "each task to one sentence. Finish with one sentence saying what was done."
    ),
)
manager.add_hook(on_delegate)
manager.add_hook(on_report)

# --- 5. Run the manager ---
print("=== Subagent delegation ===")
print(f"Model options: fast -> {settings.OPENAI_MODEL_NAME}, "
      f"strong -> {settings.OPENAI_STRONG_MODEL_NAME}\n")
result = manager("Find our most delayed order and draft an apology email to that customer.")
print(f"Manager: {str(result).strip()}")
