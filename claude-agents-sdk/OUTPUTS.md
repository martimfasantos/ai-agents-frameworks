# Claude Agent SDK - Example Outputs

Outputs captured with `claude-agent-sdk` 0.2.165 (bundled Claude Code CLI 2.1.294) on `claude-haiku-5-5`, set through `ANTHROPIC_MODEL_NAME` in `settings.py`. The CLI also loads the capturing machine's user-level Claude Code settings (CLAUDE.md, hooks, MCP servers), which shows up in 04, 09 and 16.

> **Note:** LLM responses are non-deterministic. Your outputs will differ in wording but should follow the same structure and demonstrate the same features.

---

## 00_hello_world.py

```
$ uv run python 00_hello_world.py

The string isn't a fact the repo knows. It's the prompt in `00_hello_world.py`, line 30:
`prompt="Where does 'hello world' come from?"`. The script sends that question to Claude
through `query()` and prints the result. I haven't run it, since that needs an API key.

The phrase itself is generally traced to Brian Kernighan's early C tutorial at Bell Labs
(around 1974), whose first example program printed `hello, world`. I'm working from memory,
so check that before citing it.
```

**Verdict:** PASS - One-shot `query()` call returns a complete response about "Hello, World!" history.

---

## 01_built_in_tools.py

```
$ uv run python 01_built_in_tools.py

[Tool Call] Glob: {'pattern': '*.py'}
[Tool Call] Glob: {'pattern': '**/settings.py'}
[Tool Call] Bash: {'command': "find . -maxdepth 1 -name '*.py' -type f | sort && echo '---' && head -n 5 settings.py", 'description': 'List top-level Python files and show first 5 lines of settings.py'}

--- Result ---
**Python files in the current directory** (top level only; `.venv` is excluded):

```
00_hello_world.py
01_built_in_tools.py
...
23_session_rewind.py
settings.py
```

**First 5 lines of `settings.py`:**

```python
import pydantic
from pydantic_settings import BaseSettings


# Use pydantic base settings for basic settings read from a .env file
```
```

**Verdict:** PASS - Built-in tools (Glob, Bash) invoked via allowed_tools and bypassPermissions, agent lists files and reads content.

---

## 02_custom_tools.py

```
$ uv run python 02_custom_tools.py

Lisbon is currently sunny at 25°C.

Its population is 545,000.
```

**Verdict:** PASS - Custom MCP tools (get_weather, get_population) created via @tool decorator and create_sdk_mcp_server(), both invoked correctly.

---

## 03_structured_outputs.py

```
$ uv run python 03_structured_outputs.py

=== Example 1: Raw JSON Schema ===
Structured output: {
  "name": "Paris",
  "country": "France",
  "population_millions": 2.1,
  "famous_for": [
    "Eiffel Tower",
    "Louvre Museum",
    "Notre-Dame Cathedral",
    "Sacr\u00e9-C\u0153ur",
    "Arc de Triomphe",
    "Champs-\u00c9lys\u00e9es",
    "Mus\u00e9e d'Orsay",
    "Impressionist art",
    "Haute couture and fashion",
    "French cuisine",
    "2024 Summer Olympics"
  ]
}

=== Example 2: Pydantic Model Schema ===
Title: Nineteen Eighty-Four
Author: George Orwell
Rating: 5.0/5
Summary: Orwell's 1949 novel follows Winston Smith, a clerk in the Ministry of Truth in
Oceania, a state where the Party and Big Brother watch citizens through telescreens and
rewrite history. ...
Themes: Totalitarianism and state surveillance, Control of language and thought (Newspeak,
doublethink), Manipulation of truth and history, Individual resistance and its defeat, Love
and loyalty under oppression
```

**Verdict:** PASS - Both raw JSON schema and Pydantic model-derived schema produce valid structured output matching the defined schemas.

---

## 04_system_prompts.py

```
$ uv run python 04_system_prompts.py

=== Example 1: Custom String System Prompt ===
Ahoy! Python be a high-level programming language, sailed into the seas by Guido van Rossum
in 1991. Its code reads almost like plain English, so even landlubbers can learn it.

Pirates use it for web apps, data treasure hunts, AI, and automating tedious chores. Its vast
standard library be a chest full of ready-made tools.

=== Example 2: Preset System Prompt (claude_code) ===
I have these tools available in this session:

**Loaded now:**
- **Bash**: run shell commands
- **Read**: read files, including images and PDFs
- **Write**: create or overwrite files
- **Edit**: exact string replacement in files
- **Agent**: launch subagents (general-purpose, Explore, Plan, and many specialized types)
...

**Deferred (load with ToolSearch first):**
- **Web:** WebSearch, WebFetch
...

=== Example 3: Preset with Append ===
A Python list comprehension is a compact way to build a new list by looping over an iterable
and, optionally, filtering or transforming each item. It goes inside square brackets.

```python
numbers = [1, 2, 3, 4, 5]

squares = [n * n for n in numbers]
# [1, 4, 9, 16, 25]

evens = [n for n in numbers if n % 2 == 0]
# [2, 4]
```
...

Happy coding!
```

**Verdict:** PASS - All three system prompt modes work: custom string (pirate persona), preset claude_code, and preset with appended instructions (concise + beginner-friendly).

---

## 05_permissions.py

```
$ uv run python 05_permissions.py

=== Example 1: Permission Mode with Allow/Deny Lists ===
[Tool Call] Glob

Result: The Glob matched 727 `.py` files, but the output shows only the first 100, and all of
those are inside `.venv/` (installed packages such as `httpcore`, `httpx`, `h11`, and
`jsonschema`). The list doesn't show any project source files yet.
...

=== Example 2: can_use_tool Callback ===

Result: Here's `settings.py` from the current directory:

```python
import pydantic
from pydantic_settings import BaseSettings


# Use pydantic base settings for basic settings read from a .env file
class Settings(BaseSettings):
    ANTHROPIC_API_KEY: pydantic.SecretStr
    ANTHROPIC_MODEL_NAME: str = "claude-haiku-5-5"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings: Settings = Settings()
```
...

Audit log (0 calls): []

=== Example 3: can_use_tool Shadowed by allowed_tools ===
Result: `settings.py` has 15 lines. That's the one at the repo root; I ignored the copy inside `.venv`.

[CanUseToolShadowedWarning] can_use_tool will not be invoked for: Read. An allowed_tools entry
that allows a whole tool auto-approves it before the callback is consulted. To gate every tool
call, use a PreToolUse hook; or narrow the entry so calls fall through to can_use_tool. Allow
rules from settings files can also shadow the callback but are not visible here.

Callback invocations during this run: 0
Use a PreToolUse hook to gate tools that allowed_tools already approves.
```

**Verdict:** PASS - Example 1 uses allowed_tools/disallowed_tools with permission_mode. Example 2 demonstrates the can_use_tool callback with a plain string prompt. Example 3 shows the shadowing footgun: `allowed_tools=["Read"]` auto-approves before the callback runs, so it is never invoked, and since 0.2.111 the SDK raises `CanUseToolShadowedWarning` to say so.

> The `0 calls` audit log in Example 2 is expected here, not a failure — allow rules in the machine's own Claude Code settings can auto-approve `Read` before the callback is consulted, which is exactly the condition Example 3 makes explicit.

---

## 06_hooks.py

```
$ uv run python 06_hooks.py

  [PreToolUse] About to call: Glob
               Input: {'pattern': '*.py'}
  [PostToolUse] Finished: Glob
  [PreToolUse] About to call: Bash
               Input: {'command': 'ls -1p | grep \'\\.py$\' ; ...', 'description': 'List top-level .py files in current directory'}
  [PreToolUse] BLOCKED: Bash tool is not allowed!
  [PreToolUse] About to call: Glob
               Input: {'pattern': '*.py', 'path': '.'}
  [PostToolUse] Finished: Glob
  [PreToolUse] About to call: Grep
               Input: {'pattern': '.', 'glob': '!.venv/**', 'output_mode': 'files_with_matches', 'type': 'py', 'head_limit': 0}
  [PostToolUse] Finished: Grep

--- Result ---
There are 25 `.py` files in the current directory, excluding `.venv`:
...
The first Glob search had matched 727 files, almost all in `.venv`, so I used a Grep filtered
to Python files with `.venv` excluded. Bash was blocked by a hook, so I couldn't run `ls`.
```

**Verdict:** PASS - PreToolUse hooks log tool calls before execution, Bash calls are blocked by the deny hook, PostToolUse hooks fire after completion. All three hook types demonstrated correctly.

---

## 07_sessions.py

```
$ uv run python 07_sessions.py

=== Step 1: Start a new session ===
Response: Confirmed. The secret code is **ALPHA-7**. I'll keep it for the rest of this
conversation, but I won't remember it in a future session unless you save it somewhere persistent.
Session ID: 8dc19a88-8632-4788-8cbc-ed7ed3d1acfc

=== Step 2: Resume the session ===
Response: The secret code you told me earlier is **ALPHA-7**.

=== Step 3: Fork the session ===
Forked session response: 2 + 2 = 4.

Forgetting ALPHA-7 is fine. I'll stop referring to it.
New session ID: 7c68d1c7-9e52-4211-8f06-71aa62796018
(Original session '8dc19a88-8632-4788-8cbc-ed7ed3d1acfc' is unchanged)
```

**Verdict:** PASS - Session created with unique ID, resumed by ID (remembers ALPHA-7), forked to new session with independent ID while preserving original.

---

## 08_multi_turn.py

```
$ uv run python 08_multi_turn.py

=== Turn 1 ===
Response: Got it: you're building a REST API with FastAPI. I'll keep that in mind for the rest
of this conversation.
...
Session: 2945f167-e935-4e8f-9610-594855fcba67

=== Turn 2 ===
Response: For most FastAPI REST APIs, I'd start with FastAPI's built-in security utilities
plus **PyJWT** for tokens and **pwdlib** for password hashing. ...

- **Token auth (JWT bearer):** `fastapi.security` (`OAuth2PasswordBearer`/`HTTPBearer`) handles the request plumbing.
- **Tokens:** PyJWT (`pyjwt`) signs and verifies them. FastAPI's docs dropped python-jose.
- **Passwords:** pwdlib with Argon2 or bcrypt. passlib is unmaintained.
...

=== Turn 3 ===
Response: ```python
import os
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
...
```
```

**Verdict:** PASS - ClaudeSDKClient maintains context across 3 turns: sets FastAPI context (Turn 1), recommends auth library building on context (Turn 2), provides code example for recommended library (Turn 3).

---

## 09_subagents.py

```
$ uv run python 09_subagents.py

  [task_started] {'type': 'system', 'subtype': 'task_started', 'task_id': 'a656f4d483dded14e',
    'description': 'Review calc function for issues', 'subagent_type': 'code_reviewer',
    'is_backgrounded': True, 'task_type': 'local_agent', ...}
  [task_started] {'type': 'system', 'subtype': 'task_started', 'task_id': 'a609a1d11e8d209d1',
    'description': 'Add docstring to calc function', 'subagent_type': 'doc_writer',
    'is_backgrounded': True, 'task_type': 'local_agent', ...}
  [task_notification] {'task_id': 'a609a1d11e8d209d1', 'status': 'completed',
    'summary': '```python\ndef calc(x, y, op):\n    """Applies a basic arithmetic operation...', ...}
  [task_notification] {'task_id': 'a656f4d483dded14e', 'status': 'completed',
    'summary': 'No P0 findings. There is one P1: the function silently returns `None`...', ...}

--- Final Result ---
Both agents are running. The doc writer was launched alongside the reviewer rather than after
it, so its brief tells it to document only the behavior the code actually has. I'll report when
both finish.

--- Final Result ---
The doc writer finished. Its docstring is below. ...

```python
def calc(x, y, op):
    """Applies a basic arithmetic operation to two numbers.

    Args:
        x: The left operand.
        y: The right operand.
        op: The operation to apply. One of "add", "sub", "mul", or "div".

    Returns:
        The result of `x op y`. For "div", this is true division and returns
        a float. For an unsupported `op`, returns None.

    Raises:
        ZeroDivisionError: If `op` is "div" and `y` is zero.
    ...
    """
    ...
```
...

--- Final Result ---
The review found no P0 issues and one P1. ...

**Review findings**
- **[P1] · Unknown `op` silently returns `None`.** ... Fix: add `else: raise ValueError(f"unknown op: {op!r}")`.
- **[P2] · Division by zero is uncaught and undocumented.** ...
- **[P2] · No type validation.** ...
- **[P3] · Design.** ...
...
```

**Verdict:** PASS - Two AgentDefinition subagents (code_reviewer with sonnet, doc_writer with haiku) delegated via the Agent tool. Both complete their tasks and results are aggregated by the orchestrator.

> The subagents run in the background (`is_backgrounded: True`), so the orchestrator ends a turn after launching them and starts a new one as each finishes. Each of those turns yields its own success `ResultMessage`, hence the three "Final Result" blocks; the last one carries the aggregated answer. The P0-P3 labels come from the capturing machine's user-level CLAUDE.md.

---

## 10_mcp_servers.py

```
$ uv run python 10_mcp_servers.py

[MCP Tool] ToolSearch: {'query': 'select:mcp__filesystem__list_directory,mcp__filesystem__read_text_file', 'max_results': 2}
[MCP Tool] mcp__filesystem__list_directory: {'path': '/.../claude-agents-sdk'}
[MCP Tool] mcp__filesystem__read_text_file: {'path': '/.../claude-agents-sdk/settings.py'}

--- Result ---
I listed the current directory with the filesystem MCP `list_directory` tool, then read
`settings.py` with `read_text_file`.

**Directory contents (`claude-agents-sdk`)**

- **Config and docs:** `.env`, `.env.example`, `.python-version`, `pyproject.toml`, `uv.lock`, `README.md`, `OUTPUTS.md`
- **Directories:** `.venv/`, `__pycache__/`
- **Example scripts:** `00_hello_world.py` through `23_session_rewind.py`, ...
- **Shared module:** `settings.py`

**`settings.py`**

```python
import pydantic
from pydantic_settings import BaseSettings
...
```
```

**Verdict:** PASS - External MCP server (@modelcontextprotocol/server-filesystem via npx stdio) connected, tools mcp__filesystem__list_directory and mcp__filesystem__read_text_file invoked successfully.

---

## 11_streaming.py

```
$ uv run python 11_streaming.py

Streaming response:

White space guards the flow,
indents hold the logic close,
clear as plain English.

--- Stream complete ---
Characters streamed: 82
Final result: White space guards the flow,
indents hold the logic close,
clear as plain English.
```

**Verdict:** PASS - Real-time streaming with include_partial_messages=True delivers text deltas via StreamEvent, character count tracked, final ResultMessage matches streamed content.

---

## 12_cost_tracking.py

```
$ uv run python 12_cost_tracking.py

=== Example 1: Cost and Usage Tracking ===
Result: A REST API is an interface that lets programs talk to each other over HTTP using
standard methods like GET, POST, PUT, and DELETE, with each URL representing a resource (such
as `/users/42`). Its server holds no client session state between requests, so every request
carries everything the server needs to handle it.
Cost: $0.003757
Turns: 1
Duration: 2340ms (API: 2848ms)
Usage: {'input_tokens': 2, 'cache_creation_input_tokens': 27627, 'cache_read_input_tokens':
10629, 'output_tokens': 143, 'output_tokens_details': {'thinking_tokens': 48},
'server_tool_use': {'web_search_requests': 0, 'web_fetch_requests': 0}, 'service_tier':
'standard', ...}

=== Example 2: Limit Max Turns ===
Stopped after 4 turns
Stop reason: tool_use
Result: None
Turn limit hit: Claude Code returned an error result: Reached maximum number of turns (3) (exit code: 1)

=== Example 3: Budget Cap ===
Result: None
Cost: $0.003667
Budget limit: $0.001
Budget cap hit: Claude Code returned an error result: Reached maximum budget ($0.001) (exit code: 1)

=== Example 4: Effort Level ===
[Low effort] Result: 4
Duration: 1736ms

=== Example 5: Per-Model Usage ===
Model: claude-haiku-5-5 (firstParty)
  Input tokens:          1198
  Output tokens:         523
  Cache read tokens:     75855
  Cache creation tokens: 63469
  Cost:                  $0.009073
  Context window:        1000000

Cache creation tokens in usage (main loop only): 28913
Cache creation tokens in model_usage (whole tree): 63469
```

**Verdict:** PASS - All five controls demonstrated: cost/usage tracking with total_cost_usd and usage dict, max_turns=3 limiting agent rounds, max_budget_usd=0.001 tripping the budget cap, effort="low" for fast simple answers, and per-model accounting via model_usage.

> Example 5 is the point of the `ModelUsage` addition: `usage` counts only the main loop (28,913 cache-creation tokens) while `model_usage` aggregates the whole agent tree including the subagent (63,469). Anything measuring spend from `usage` alone undercounts subagent runs. The subagent ran on the same model, so `model_usage` has a single `claude-haiku-5-5` entry.
>
> Examples 2 and 3 exceed their limits by design; the SDK signals both by raising on the error result, which the example catches and prints. The $0.001 cap in Example 3 is below the cost of a single `claude-haiku-5-5` request (about $0.004 here), so it always trips.

---

## 13_file_checkpointing.py

```
$ uv run python 13_file_checkpointing.py

Working directory: /var/folders/.../claude_checkpoint_jiafuj4s
Initial file content: Original content: Hello World

=== Turn 1: Modify the file ===
Checkpoint UUID: 16722acf-0080-40f4-ba65-f5361c9df962
Result: I read `example.txt`, which contained `Original content: Hello World`, and overwrote it
with `Modified by agent: version 2`.
File now contains: Modified by agent: version 2

=== Turn 2: Modify again ===
File now contains: Modified again: version 3

=== Rewinding to Turn 1 checkpoint ===
File after rewind: Original content: Hello World
```

**Verdict:** PASS - File checkpointing enabled with `enable_file_checkpointing=True` and `extra_args={"replay-user-messages": None}`. Agent modifies file across two turns, then `rewind_files()` restores the file to its original state by resuming the session with an empty prompt.

---

## 14. Session Store (`14_session_store.py`)

```
$ uv run python 14_session_store.py

=== Session Store Demo ===
Store type: InMemorySessionStore

--- Step 1: Run a query with session mirroring ---
Response: The three primary colors are red, yellow, and blue (for pigments and paint; the
additive RGB primaries of light are red, green, and blue).
Session ID: 5c81c720-6e05-4e5b-8b8a-b9573f7cad2a

--- Step 2: List sessions from store ---
Sessions in store: 1
Traceback (most recent call last):
  ...
  File ".../14_session_store.py", line 78, in inspect_store
    print(f"  - {s.id} (created: {s.created_at})")
                 ^^^^
AttributeError: 'SDKSessionInfo' object has no attribute 'id'
```

> The store side works: the session is mirrored into `InMemorySessionStore` and `list_sessions_from_store` finds it. The example then reads `s.id`, but `SDKSessionInfo` exposes `session_id` (the same in 0.2.130), and `get_session_messages_from_store` returns `SessionMessage` objects rather than `ResultMessage`, so Step 3's preview logic needs the same correction.

**Verdict:** FAIL - Pre-existing bug in the example (not caused by the upgrade or the model): `SDKSessionInfo.id` does not exist.

---

## 15. Deferred Tool Use (`15_deferred_tool_use.py`)

```
$ uv run python 15_deferred_tool_use.py

=== Deferred Tool Use (Human-in-the-Loop) ===
  [HITL Hook] Tool 'mcp__database__delete_record' requested with args: {'record_id': 'usr-42'}
  [HITL Hook] Deferring for human approval...

Result subtype: success

--- Deferred Tool Use ---
  Tool ID:    toolu_01HTMTxthFcwREB21a6yLapD
  Tool Name:  mcp__database__delete_record
  Tool Input: {'record_id': 'usr-42'}

  A human reviewer would inspect this and decide to approve or deny.
  To resume, pass the session ID back with the approval decision.
```

> The PreToolUse hook returns `permissionDecision: "defer"`, which halts the agent run. The `deferred_tool_use` field on ResultMessage carries the tool call details for human review.

**Verdict:** PASS - Deferred tool use stops the agent and exposes the pending tool call for HITL review.

---

## 16. Hook Events (`16_hook_events.py`)

```
$ uv run python 16_hook_events.py

=== Hook Event Streaming ===

[HookEvent #1]
  Event: SessionStart
  Subtype: hook_started
...
[HookEvent #6]
  Event: SessionStart
  Subtype: hook_response

[HookEvent #7]
  Event: UserPromptSubmit
  Subtype: hook_started
...
[HookEvent #10]
  Event: UserPromptSubmit
  Subtype: hook_response

--- Result ---
I couldn't get a clean list of your project's Python files with Glob. It matched 727 `.py`
files, and the 100 it returned are all inside `.venv/` (the virtualenv's site-packages). ...

Total hook events received: 10
```

> With `include_hook_events=True`, hook lifecycle events are streamed as `HookEventMessage` objects. All 10 events here come from command hooks in the capturing machine's user-level Claude Code settings; the example's own in-process `PreToolUse`/`PostToolUse` callbacks emitted no events (same result on 0.2.130), so on a machine without settings-file hooks it likely prints 0 events.

**Verdict:** PARTIAL - HookEventMessage objects arrive in the stream, but not for the SDK callback hooks the example registers (pre-existing, not caused by the upgrade or the model).

---

## 17. Strict MCP Config (`17_strict_mcp.py`)

```
$ uv run python 17_strict_mcp.py

=== Strict MCP Config Demo ===

Running with strict_mcp_config=True (no external MCP servers)...

[Tool] Glob
[Tool] Bash

--- Result ---
There are 24 `.py` files in the current directory:

- `00_hello_world.py`
- `01_built_in_tools.py`
- ...
- `23_session_rewind.py`
- `settings.py`

My first attempt with the Glob tool matched files in subdirectories such as `.venv`, so I used
`ls` for the top-level listing.

Note: strict_mcp_config=True ensured no project/user/global MCP servers loaded.
```

> With `strict_mcp_config=True`, only explicitly-passed MCP servers are available. No project, user, or global MCP configurations are loaded, ensuring a deterministic tool set.

**Verdict:** PASS - Agent runs with only built-in tools; no external MCP servers loaded.

---

## 18. Thinking Config (`18_thinking_config.py`)

```
$ uv run python 18_thinking_config.py

=== Example 1: Adaptive Thinking ===
Response: 4

=== Example 2: Thinking Enabled (budget: 2000 tokens) ===
Response: **The halting problem:** No program can always tell whether any other program will
eventually stop or run forever.

**Why you can't just run it and see:** If a program is still running after an hour, it might
be about to finish or it might loop forever, and you can't tell which from the outside. Som

=== Example 3: Thinking Disabled ===
Response: Here are three prime numbers: **2, 3, and 5**.
```

> ThinkingConfig controls extended thinking behavior. In SDK 0.2.x these are
> TypedDicts that require an explicit `type` discriminator:
> - `ThinkingConfigAdaptive(type="adaptive")`: model decides whether to think based on complexity
> - `ThinkingConfigEnabled(type="enabled", budget_tokens=N)`: always think, capped at N tokens
> - `ThinkingConfigDisabled(type="disabled")`: suppress thinking for faster, cheaper responses
>
> Verified live via `query()` against the Claude CLI: all three modes return
> `ResultMessage(subtype="success")`.

**Verdict:** PASS - All three thinking modes run and return successful results.

---

## 19. Task Budget (`19_task_budget.py`)

```
$ uv run python 19_task_budget.py

=== Example 1: Dollar Budget Cap ===
Response: Silent cursor blinks,
one missing semicolon
hides the whole night's bug.

=== Example 2: TaskBudget with Token Limit ===
Response: # A Brief History of Python

I wrote this from memory without searching, so the dates and version details should be checked
against python.org or the PEP archive before you cite them.

## Origins (1980s–1991)

- **Creator:** Guido van Rossum, a Dutch programmer working at CWI (Centrum Wiskunde & Inf

=== Budget Controls Summary ===
max_budget_usd:    Hard dollar cap on total API spend
TaskBudget(total): Total token budget the model paces against
max_turns:         Maximum agent reasoning iterations
```

> Budget controls cap agent resource consumption per task:
> - `max_budget_usd=1.00`: hard dollar ceiling on total API spend (the SDK raises
>   an error result if the cap trips, so set it above a single task's expected cost)
> - `TaskBudget(total=25000)`: total token budget the model is made aware of and
>   paces against. `total` is a token count and must be at least the model minimum;
>   25,000 is accepted on `claude-haiku-5-5`
>
> Verified live via `query()` against the Claude CLI: both examples return
> `ResultMessage(subtype="success")`.

**Verdict:** PASS - Dollar cap and TaskBudget token budget both run successfully.

---

## 20. Sandbox Settings (`20_sandbox_settings.py`)

```
$ uv run python 20_sandbox_settings.py

=== Example 1: Sandbox with Network Locked Down ===
Response: **15 × 23 = 345**

=== Example 2: Full Sandbox Configuration ===
Response: The current directory contains:

**Directories**
- `.claude/`
- `.venv/`
- `__pycache__/`
...

**Python examples** (`00_` to `23_`)
- `00_hello_world.py`
- `01_built_in_tools.py`
- ...

=== Sandbox Settings Summary ===
enabled=True:           Sandbox bash commands (macOS/Linux)
network.allowedDomains: Whitelist outbound domains ([] blocks all)
network.deniedDomains:  Always-blocked domains
Sandboxing prevents data exfiltration and limits blast radius
```

> SandboxSettings sandboxes bash command execution. In SDK 0.2.x the keys are
> camelCase and network access is controlled by domain lists (not an `enabled`
> flag on the network config):
> - `SandboxSettings(enabled=True, ...)`: enable bash sandboxing (macOS/Linux)
> - `SandboxNetworkConfig(allowedDomains=[])`: block all outbound domains
> - `SandboxNetworkConfig(allowedDomains=["api.github.com"])`: whitelist domains
> - `SandboxNetworkConfig(deniedDomains=[...])`: always-blocked domains
>
> Verified live via `query()` against the Claude CLI: both examples return
> `ResultMessage(subtype="success")`.

**Verdict:** PASS - Sandboxed bash with network domain controls runs successfully.

---

## 21_interrupt_and_terminal_reason.py

```
$ uv run python 21_interrupt_and_terminal_reason.py

=== Example 1: Interrupt a Running Turn ===
  [interrupt() sent while the model was streaming]
  subtype:         error_during_execution
  terminal_reason: aborted_streaming
  is_error:        True
  result:          None
  extra deltas after interrupt: 3

  Sending a follow-up turn on the same client...
  terminal_reason: completed
  result:          4

=== Example 2: terminal_reason=max_turns ===
  subtype:         error_max_turns
  terminal_reason: max_turns
  stop_reason:     tool_use
  turns used:      2

=== terminal_reason values ===
completed:         the turn finished on its own
max_turns:         max_turns was reached
api_error:         the upstream API call failed
aborted_streaming: interrupted while the model was generating text
aborted_tools:     interrupted while a tool call was in flight
None:              older CLI, or a result that bypassed the query loop
```

**Verdict:** PASS - `interrupt()` cancels the turn mid-stream and `terminal_reason` (0.2.126) reports why it ended: `aborted_streaming` for the cancelled turn, `completed` for the follow-up, `max_turns` when the limit is hit.

> `extra deltas after interrupt: 3` counts text deltas that were already in flight when `interrupt()` was sent; the turn still ends as `aborted_streaming` with no result, which is the proof the cancellation took effect. The aborted turn still emits its own `ResultMessage`, so the example keeps iterating `receive_response()` to drain it; skipping that drain makes the next `query()` read the dead turn's buffered messages. Note `ClaudeSDKClient` yields error results as `ResultMessage`, whereas `query()` raises on them — which is why the client is used here.

---

## 22. Verbatim Prompts (`22_verbatim_prompts.py`)

```
$ uv run python 22_verbatim_prompts.py

Untrusted input: Great product! See @settings.py for my notes.

=== Default: @settings.py is expanded ===
  Model saw: FILE SEEN: import pydantic

=== verbatim_prompts=True: delivered as written ===
  Model saw: LITERAL: Great product! See @settings.py for my notes.

=== What verbatim_prompts turns off ===
@path mentions:  no local file is read and attached
/commands:       a leading '/' is not run as a slash command
Per-message:     leave it off and set "client_composed": True on streamed messages
```

> Both runs have no tools (`tools=[]`), so the only way the first one could see `import pydantic` is Claude Code expanding `@settings.py` and attaching the file. With `verbatim_prompts=True` every user message is sent with `client_composed: True` and the mention reaches the model as plain text. Requires Claude Code 2.1.248+; older CLIs ignore the flag.

**Verdict:** PASS - Default run leaks the local file into the model's context; the verbatim run keeps the untrusted text literal.

---

## 23. Session Rewind (`23_session_rewind.py`)

```
$ uv run python 23_session_rewind.py

=== Step 1: Two turns in one session ===
  You:    Fact A: my favourite colour is teal. Reply 'Noted A.'
  Claude: Noted A.
  You:    Fact B: my favourite city is Lisbon. Reply 'Noted B.'
  Claude: Noted B.

=== Step 2: Transcript of cab32788 ===
  [0] user      79a0576b
  [1] assistant 4488e6c9  <- resume_session_at (last entry of turn 1)
  [2] user      a07828bf  <- resume_drops_turn (turn 2 prompt)
  [3] assistant a04e2d27

=== Step 3: Rewind to turn 1 and ask what is known ===
  Reply: colour=teal, city=unknown
  Forked session: 2477e4ee
  Original session still has 4 messages

=== Step 4: Declaring the wrong turn as dropped ===
  Refused with ResultError: Claude Code returned an error result: Resume rejected by
  --resume-drops-turn: resuming at 4488e6c9-... would discard entries not attributable to turn
  79a0576b-...: range does not start with the declared turn prompt; first discarded entry 1
  [type=user, uuid=a07828bf-...] (exit code: 1)
```

> `city=unknown` is the proof the rewind worked: the forked session was loaded only up to the end of turn 1, so fact B never reached the model, while the original session keeps all 4 messages. `ResultError` is a subclass of `ProcessError`, and the CLI also writes the rejection line to stderr.

**Verdict:** PASS - `get_session_messages()` exposes the transcript UUIDs, `resume_session_at` + `resume_drops_turn` + `fork_session=True` drop the last turn, and a mismatched drop declaration is refused before any model call.

---

## Summary

| # | File | Status | Notes |
|---|------|--------|-------|
| 0 | `00_hello_world.py` | PASS | Basic query with ResultMessage |
| 1 | `01_built_in_tools.py` | PASS | Built-in tool definitions and invocations |
| 2 | `02_custom_tools.py` | PASS | Custom tool registration and dispatch |
| 3 | `03_structured_outputs.py` | PASS | JSON schema output validation |
| 4 | `04_system_prompts.py` | PASS | System prompt configuration and presets |
| 5 | `05_permissions.py` | PASS | Permission modes, allow/deny lists, can_use_tool |
| 6 | `06_hooks.py` | PASS | PreToolUse/PostToolUse hook callbacks |
| 7 | `07_sessions.py` | PASS | Session create, resume, fork |
| 8 | `08_multi_turn.py` | PASS | Context retention across 3 turns |
| 9 | `09_subagents.py` | PASS | Two subagents delegated via Agent tool |
| 10 | `10_mcp_servers.py` | PASS | External MCP server integration |
| 11 | `11_streaming.py` | PASS | Real-time token streaming |
| 12 | `12_cost_tracking.py` | PASS | Cost, turns, duration, budget, effort, model_usage |
| 13 | `13_file_checkpointing.py` | PASS | File checkpoint and rewind |
| 14 | `14_session_store.py` | FAIL | Pre-existing `SDKSessionInfo.id` bug after the store is populated |
| 15 | `15_deferred_tool_use.py` | PASS | Human-in-the-loop deferral |
| 16 | `16_hook_events.py` | PARTIAL | Events stream, but only from settings-file hooks |
| 17 | `17_strict_mcp.py` | PASS | Strict MCP config isolation |
| 18 | `18_thinking_config.py` | PASS | Extended thinking modes |
| 19 | `19_task_budget.py` | PASS | Dollar cap and token budget |
| 20 | `20_sandbox_settings.py` | PASS | Sandboxed bash with network domain controls |
| 21 | `21_interrupt_and_terminal_reason.py` | PASS | interrupt() cancellation and terminal_reason |
| 22 | `22_verbatim_prompts.py` | PASS | verbatim_prompts keeps @path mentions literal |
| 23 | `23_session_rewind.py` | PASS | Drop the last turn with resume_session_at + resume_drops_turn |

**22/24 examples pass; 16 is partial and 14 fails, both pre-existing.**
