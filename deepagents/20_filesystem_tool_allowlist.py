from typing import get_args

from dotenv import load_dotenv

from deepagents import FilesystemPermission, FsToolName, create_deep_agent
from deepagents.backends import StateBackend
from deepagents.backends.utils import create_file_data
from deepagents.middleware.filesystem import FilesystemMiddleware

from settings import settings

load_dotenv()

"""
-----------------------------------------------------------------------
In this example, we explore Deep Agents with the following features:
- FilesystemMiddleware(tools=[...]): an allowlist of filesystem tools
- FsToolName, the exported Literal of valid filesystem tool names
- Overriding a default middleware by .name match via middleware=[...]
- The replacement inheriting create_deep_agent(permissions=...) rules

By default the harness hands the model every filesystem tool it can,
including write_file, edit_file and a recursive delete. Passing tools=
narrows that set, and passing the configured instance through middleware=
replaces the default FilesystemMiddleware in place, because a custom
middleware whose .name matches a default one substitutes for it. Here we
build a research agent that can look at files but never change them.

An allowlist is an ergonomics feature, not a security boundary; permission
rules are, and the replacement middleware inherits the agent's permissions=.

For more details, visit:
https://docs.langchain.com/oss/python/deepagents/customization
-----------------------------------------------------------------------
"""


def tool_names(agent) -> list[str]:
    """Names of the tools installed on a compiled deep agent's tool node."""
    return sorted(agent.nodes["tools"].bound.tools_by_name)


# --- 1. The full set of filesystem tool names FsToolName accepts ---
print("=== Deep Agents Filesystem Tool Allowlist ===")
print(f"FsToolName options: {list(get_args(FsToolName))}")
print(f"Default agent tools: {tool_names(create_deep_agent(model=f'openai:{settings.OPENAI_MODEL_NAME}'))}")

# --- 2. Configure a read-only filesystem middleware (read_file is mandatory) ---
backend = StateBackend()
readonly_fs = FilesystemMiddleware(
    backend=backend,
    tools=["ls", "read_file", "glob", "grep"],
)
print(f"\nAllowlisted middleware '{readonly_fs.name}' exposes: {[t.name for t in readonly_fs.tools]}")

# --- 3. Passing it via middleware= replaces the default instance by .name ---
agent = create_deep_agent(
    model=f"openai:{settings.OPENAI_MODEL_NAME}",
    backend=backend,
    middleware=[readonly_fs],
    system_prompt=(
        "You are a read-only research assistant. Use the tools you have. "
        "If a request needs a tool you do not have, say so plainly in one sentence."
    ),
)
installed = tool_names(agent)
print(f"Agent tools after the override: {installed}")
for mutating in ("write_file", "edit_file", "delete"):
    assert mutating not in installed, f"{mutating} should have been excluded"
print("Mutating tools write_file / edit_file / delete are absent ✅")

# --- 4. Omitting read_file is rejected outright ---
try:
    FilesystemMiddleware(backend=backend, tools=["ls", "glob"])
except ValueError as exc:
    print(f"\nFilesystemMiddleware(tools=['ls', 'glob']) -> ValueError: {exc}")

# --- 5. The agent can read the seeded file but cannot remove it ---
seed_files = {"/report.md": create_file_data("# Q3 Report\nRevenue grew 12%.\n")}
task = "Read /report.md and tell me the revenue figure, then delete the file."
result = agent.invoke(
    {"messages": [{"role": "user", "content": task}], "files": seed_files}
)

print(f"\nUser: {task}")
print(f"Agent: {result['messages'][-1].text}")
print(f"\nFiles still present: {sorted(result.get('files', {}))}")

# --- 6. Agent permissions= rules carry over to the replacement middleware ---
deny_internal = [FilesystemPermission(operations=["read"], paths=["/internal/**"], mode="deny")]
guarded = create_deep_agent(
    model=f"openai:{settings.OPENAI_MODEL_NAME}",
    backend=backend,
    middleware=[readonly_fs],
    permissions=deny_internal,
    system_prompt=(
        "You are a read-only research assistant. Always call read_file on every path "
        "the user names, then report the result in one sentence."
    ),
)
internal_files = {"/internal/roadmap.md": create_file_data("# Roadmap\nLaunch Atlas in May.\n")}
task = "Read /internal/roadmap.md and summarize it."
result = guarded.invoke(
    {"messages": [{"role": "user", "content": task}], "files": internal_files}
)

print("\n=== Replacement middleware + permissions=[deny read /internal/**] ===")
print(f"User: {task}")
for msg in result["messages"]:
    if msg.type == "tool":
        print(f"[tool result] {msg.name}: {msg.text}")
reply = result["messages"][-1].text
print(f"Agent: {reply}")
print(f"Denied content leaked into the reply: {'Atlas' in reply}")

# --- 7. Conflicting rule lists on the agent and the middleware are rejected ---
try:
    create_deep_agent(
        model=f"openai:{settings.OPENAI_MODEL_NAME}",
        backend=backend,
        permissions=deny_internal,
        middleware=[
            FilesystemMiddleware(
                backend=backend,
                tools=["ls", "read_file"],
                permissions=[FilesystemPermission(operations=["read"], paths=["/**"], mode="allow")],
            )
        ],
    )
except ValueError as exc:
    print(f"\nConflicting middleware permissions= -> ValueError: {exc}")

# --- 8. tools= trims the tool list; permission rules are the boundary ---
print(
    "\nCaveat: tools= only hides tools; it does not restrict which paths the "
    "remaining tools can touch. Enforce that with permission rules, either on "
    "create_deep_agent(permissions=...), which a replacement FilesystemMiddleware "
    "inherits, or on FilesystemMiddleware(permissions=...) directly. Interrupt "
    "(human-approval) rules still have to go through create_deep_agent(permissions=...)."
)
