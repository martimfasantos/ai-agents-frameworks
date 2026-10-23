# Claude Agent SDK - Example Outputs

Outputs captured with `claude-agent-sdk` 0.2.165 (bundled Claude Code CLI 2.1.294) for examples 00-23 and 0.2.173 (bundled Claude Code CLI 2.1.309) for examples 24-25, all re-verified on 0.2.173, on `claude-haiku-5-5`, set through `ANTHROPIC_MODEL_NAME` in `settings.py`. Every run used an empty `CLAUDE_CONFIG_DIR`, so user-level Claude Code settings don't affect them.

> **Note:** LLM responses are non-deterministic. Your outputs will differ in wording but should follow the same structure and demonstrate the same features.

---

## 00_hello_world.py

```
$ uv run python 00_hello_world.py

The string comes from this repo's own example script, not from anywhere in the code's behavior.

In `00_hello_world.py` (line 30), the prompt is literally:

```python
prompt="Where does 'hello world' come from?"
```
...

As for the phrase itself: it's the traditional first program in programming, printing
`hello, world`. It's usually traced to Brian Kernighan's tutorial for the B language at Bell
Labs in 1972. The 1978 *The C Programming Language* book by Kernighan and Ritchie popularized
it. ...
```

**Verdict:** PASS - One-shot `query()` call returns a complete response about "Hello, World!" history.

---

## 01_built_in_tools.py

```
$ uv run python 01_built_in_tools.py

[Tool Call] Glob: {'pattern': '*.py'}
[Tool Call] Bash: {'command': 'head -n 5 settings.py', 'description': 'Show first 5 lines of settings.py'}
[Tool Call] Bash: {'command': 'ls -1 *.py 2>/dev/null', 'description': 'List top-level Python files in current directory'}

--- Result ---
Python files in the current directory (top level only):

```
00_hello_world.py
01_built_in_tools.py
...
23_session_rewind.py
settings.py
```

First 5 lines of `settings.py`:

```python
import pydantic
from pydantic_settings import BaseSettings


# Use pydantic base settings for basic settings read from a .env file
```

My first search was recursive and picked up many files under `.venv/`. The list above comes
from a top-level-only `ls`.
```

**Verdict:** PASS - Built-in tools (Glob, Bash) invoked via allowed_tools and bypassPermissions, agent lists files and reads content.

---

## 02_custom_tools.py

```
$ uv run python 02_custom_tools.py

Lisbon is currently sunny at 25°C. Its population is 545,000.
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
    "Louvre Museum and the Mona Lisa",
    "Notre-Dame Cathedral",
    "Arc de Triomphe and the Champs-\u00c9lys\u00e9es",
    "Sacr\u00e9-C\u0153ur in Montmartre",
    "Mus\u00e9e d'Orsay",
    "Caf\u00e9s, bakeries, and fine dining",
    "Fashion and haute couture",
    "Impressionist and Left Bank art and literary history",
    "Hosting the Summer Olympics in 1900, 1924, and 2024"
  ]
}

=== Example 2: Pydantic Model Schema ===
Title: Nineteen Eighty-Four
Author: George Orwell
Rating: 5.0/5
Summary: Orwell's final novel follows Winston Smith, who secretly rebels against a totalitarian
state of perpetual war, doublethink, and constant surveillance. ...
Themes: Totalitarianism and state control, Surveillance, Propaganda and manipulation of truth,
Language and thought (Newspeak, doublethink), Individual resistance and its defeat, Memory and
history, Torture and political coercion
```

**Verdict:** PASS - Both raw JSON schema and Pydantic model-derived schema produce valid structured output matching the defined schemas.

---

## 04_system_prompts.py

```
$ uv run python 04_system_prompts.py

=== Example 1: Custom String System Prompt ===
Ahoy! **Python** be a high-level programming language, created by Guido van Rossum and first
released in 1991. Its clean, readable syntax makes it a favorite for web development, data
science, AI, automation, and scripting. Arr, it's one of the most popular languages on the
seven seas of code!

=== Example 2: Preset System Prompt (claude_code) ===
I have these tools available now:

**Loaded:**
- **Bash**: run shell commands
- **Read**: read files, images, PDFs, and notebooks
- **Edit**: exact string replacement in files
- **Write**: create or overwrite files
- **Agent**: launch subagents (Explore, Plan, general-purpose, and others)
...

**Deferred (load with ToolSearch first):**
- **WebSearch** and **WebFetch**: search the web and fetch pages
...

=== Example 3: Preset with Append ===
A Python list comprehension is a concise way to build a new list by applying an expression to
each item in an iterable, optionally filtering items with a condition. It's a compact
replacement for a `for` loop that appends to a list.

**Basic form:**

```python
[expression for item in iterable]
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

Result: The Glob search matched 727 `.py` files, but the tool only returned the first 100, so
this list is incomplete. Almost all of the results are inside `.venv/` (installed packages). ...

=== Example 2: can_use_tool Callback ===

Result: Here are the contents of `settings.py`:

```python
import pydantic
from pydantic_settings import BaseSettings
...
settings: Settings = Settings()
```
...

Audit log (0 calls): []

=== Example 3: can_use_tool Shadowed by allowed_tools ===
  [Permission Check] Bash requested
Result: `settings.py` has **15 lines**. I read it with the file reader because the Bash tool is
blocked by policy, so this count comes from the reader's line numbering. ...

[CanUseToolShadowedWarning] can_use_tool will not be invoked for: Read. An allowed_tools entry
that allows a whole tool auto-approves it before the callback is consulted. To gate every tool
call, use a PreToolUse hook; or narrow the entry so calls fall through to can_use_tool. Allow
rules from settings files can also shadow the callback but are not visible here.

Callback invocations during this run: 1
Use a PreToolUse hook to gate tools that allowed_tools already approves.
```

**Verdict:** PASS - Example 1 uses allowed_tools/disallowed_tools with permission_mode. Example 2 demonstrates the can_use_tool callback with a plain string prompt. Example 3 shows the shadowing footgun: `allowed_tools=["Read"]` auto-approves Read before the callback runs, so the callback only sees the Bash call (and denies it), and since 0.2.111 the SDK raises `CanUseToolShadowedWarning` to say so.

> The `0 calls` audit log in Example 2 is expected, not a failure: even with no allow rules configured, Claude Code does not ask permission to read files in the working directory, so `can_use_tool` is only consulted for tools that need approval, like the Bash call in Example 3.

---

## 06_hooks.py

```
$ uv run python 06_hooks.py

  [PreToolUse] About to call: Glob
               Input: {'pattern': '*.py'}
  [PostToolUse] Finished: Glob
  [PreToolUse] About to call: Glob
               Input: {'pattern': '*.py', 'path': '/.../claude-agents-sdk'}
  [PostToolUse] Finished: Glob
  [PreToolUse] About to call: Bash
               Input: {'command': 'ls -1 *.py 2>/dev/null', 'description': 'List top-level .py files in current directory'}
  [PreToolUse] BLOCKED: Bash tool is not allowed!
  [PreToolUse] About to call: Grep
               Input: {'pattern': '.', 'path': '/.../claude-agents-sdk', 'glob': '/*.py', 'output_mode': 'files_with_matches'}
  [PostToolUse] Finished: Grep

--- Result ---
There are 25 `.py` files in the top level of the current directory:

- `00_hello_world.py`
- `01_built_in_tools.py`
- ...
- `23_session_rewind.py`
- `settings.py`

My first two searches returned files inside `.venv` as well, so I used a root-anchored search
for this list. ...
```

**Verdict:** PASS - PreToolUse hooks log tool calls before execution, Bash calls are blocked by the deny hook, PostToolUse hooks fire after completion. All three hook types demonstrated correctly.

---

## 07_sessions.py

```
$ uv run python 07_sessions.py

=== Step 1: Start a new session ===
Response: I can't store it in any persistent way. I don't have a memory tool or a file I've been
asked to write to, so the code won't survive past this conversation. Within this conversation,
I'll remember that the secret code is ALPHA-7 and can repeat it when you ask.
Session ID: b73ddfdb-cae9-42cf-9cb9-af18ebd48843

=== Step 2: Resume the session ===
Response: The secret code you told me earlier was ALPHA-7.

=== Step 3: Fork the session ===
Forked session response: 4
New session ID: 4c4d341c-2211-41b5-aef1-5f2ac6958913
(Original session 'b73ddfdb-cae9-42cf-9cb9-af18ebd48843' is unchanged)
```

**Verdict:** PASS - Session created with unique ID, resumed by ID (remembers ALPHA-7), forked to new session with independent ID while preserving original.

---

## 08_multi_turn.py

```
$ uv run python 08_multi_turn.py

=== Turn 1 ===
Response: Got it: you're building a REST API with FastAPI. I'll keep that in mind for the rest
of this conversation. ...
Session: 83c9d7b8-e51a-423e-ac3f-f1055b9a02c5

=== Turn 2 ===
Response: For most FastAPI REST APIs, I'd start with **FastAPI's built-in security utilities
plus PyJWT**. It's the approach the FastAPI docs describe, it adds few dependencies, and you
keep control of your user model.

- **Password hashing:** `pwdlib` with Argon2 or bcrypt. The FastAPI docs now use it in place of the older `passlib`.
- **Tokens:** `PyJWT` for signing and verifying JWT access tokens. Avoid `python-jose`, which is poorly maintained.
- **Wiring:** `OAuth2PasswordBearer` as the dependency that extracts the token, plus a `get_current_user` dependency you can reuse on protected routes.
...

=== Turn 3 ===
Response: ```python
import time
import jwt
from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
...
```
...
```

**Verdict:** PASS - ClaudeSDKClient maintains context across 3 turns: sets FastAPI context (Turn 1), recommends auth library building on context (Turn 2), provides code example for recommended library (Turn 3).

---

## 09_subagents.py

```
$ uv run python 09_subagents.py

  [task_started] {'type': 'system', 'subtype': 'task_started', 'task_id': 'a333c65537e71f7e7',
    'description': 'Review calc function for issues', 'subagent_type': 'code_reviewer',
    'is_backgrounded': True, 'task_type': 'local_agent', ...}
  [task_notification] {'task_id': 'a333c65537e71f7e7', 'status': 'completed',
    'summary': 'Findings, ranked by severity:\n\n**1. High: unknown `op` silently returns `None`.**...', ...}
  [task_started] {'type': 'system', 'subtype': 'task_started', 'task_id': 'af01c2a30b19b805d',
    'description': 'Add docstring to calc function', 'subagent_type': 'doc_writer',
    'is_backgrounded': True, 'task_type': 'local_agent', ...}
  [task_notification] {'task_id': 'af01c2a30b19b805d', 'status': 'completed',
    'summary': '```python\ndef calc(x, y, op):\n    """Applies a binary arithmetic operation...', ...}

--- Final Result ---
The code reviewer is running on the function now. I'll have the doc writer add the docstring
once its findings are back, so the docstring can describe the edge cases it identifies (unknown
ops, division by zero).

--- Final Result ---
The code reviewer has finished, and the doc writer is now adding the docstring. ...

**Review findings so far (ranked by severity):**

1. **High: unknown `op` silently returns `None`.** ...
2. **High: division by zero is unhandled and undocumented.** ...
3. **Medium: no type handling.** ...
...

--- Final Result ---
Both subagents are done. Here is the updated function with the docstring:

```python
def calc(x, y, op):
    """Applies a binary arithmetic operation to two values.
    ...
    Raises:
        ZeroDivisionError: If ``op`` is ``"div"`` and ``y`` is 0.
    ...
    """
    ...
```
...
```

**Verdict:** PASS - Two AgentDefinition subagents (code_reviewer with sonnet, doc_writer with haiku) delegated via the Agent tool. Both complete their tasks and results are aggregated by the orchestrator.

> The subagents run in the background (`is_backgrounded: True`), so the orchestrator ends a turn after launching them and starts a new one as each finishes. Each of those turns yields its own success `ResultMessage`, hence the three "Final Result" blocks; the last one carries the aggregated answer.

---

## 10_mcp_servers.py

```
$ uv run python 10_mcp_servers.py

[MCP Tool] ToolSearch: {'query': 'select:mcp__filesystem__list_directory,mcp__filesystem__read_text_file', 'max_results': 2}
[MCP Tool] mcp__filesystem__list_directory: {'path': '/.../claude-agents-sdk'}
[MCP Tool] mcp__filesystem__read_text_file: {'path': '/.../claude-agents-sdk/settings.py'}

--- Result ---
Here are the files in the current directory, listed with the filesystem MCP tools:

- **Config and env:** `.env`, `.env.example`, `.python-version`, `pyproject.toml`, `uv.lock`
- **Directories:** `.venv/`, `__pycache__/`
- **Numbered examples:** `00_hello_world.py` through `23_session_rewind.py`, ...
- **Docs and settings:** `OUTPUTS.md`, `README.md`, `settings.py`

`settings.py` contents:

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

Indented whitespace, bright,
Snakes curl through clean, readable lines—
Import the Zen.

--- Stream complete ---
Characters streamed: 87
Final result: Indented whitespace, bright,
Snakes curl through clean, readable lines—
Import the Zen.
```

**Verdict:** PASS - Real-time streaming with include_partial_messages=True delivers text deltas via StreamEvent, character count tracked, final ResultMessage matches streamed content.

---

## 12_cost_tracking.py

```
$ uv run python 12_cost_tracking.py

=== Example 1: Cost and Usage Tracking ===
Result: A REST API is an interface that lets software systems communicate over HTTP using
standard methods like GET, POST, PUT, and DELETE, with each URL representing a resource. The
server stays stateless, so every request carries the information it needs to be processed.
Cost: $0.000184
Turns: 1
Duration: 1450ms (API: 1142ms)
Usage: {'input_tokens': 2, 'cache_creation_input_tokens': 0, 'cache_read_input_tokens': 14383,
'output_tokens': 79, 'output_tokens_details': {'thinking_tokens': 0}, 'server_tool_use':
{'web_search_requests': 0, 'web_fetch_requests': 0}, 'service_tier': 'standard', ...}

=== Example 2: Limit Max Turns ===
Stopped after 4 turns
Stop reason: tool_use
Result: None
Turn limit hit: Claude Code returned an error result: Reached maximum number of turns (3) (exit code: 1)

=== Example 3: Budget Cap ===
Result: None
Cost: $0.000551
Budget limit: $0.0001
Budget cap hit: Claude Code returned an error result: Reached maximum budget ($0.0001) (exit code: 1)

=== Example 4: Effort Level ===
[Low effort] Result: 2 + 2 = 4
Duration: 1087ms

=== Example 5: Per-Model Usage ===
Model: claude-haiku-5-5 (firstParty)
  Input tokens:          1198
  Output tokens:         511
  Cache read tokens:     47505
  Cache creation tokens: 4945
  Cost:                  $0.001468
  Context window:        1000000

Cache creation tokens in usage (main loop only): 576
Cache creation tokens in model_usage (whole tree): 4945
```

**Verdict:** PASS - All five controls demonstrated: cost/usage tracking with total_cost_usd and usage dict, max_turns=3 limiting agent rounds, max_budget_usd=0.0001 tripping the budget cap, effort="low" for fast simple answers, and per-model accounting via model_usage.

> Example 5 is the point of the `ModelUsage` addition: `usage` counts only the main loop (576 cache-creation tokens) while `model_usage` aggregates the whole agent tree including the subagent (4,945). Anything measuring spend from `usage` alone undercounts subagent runs. The subagent ran on the same model, so `model_usage` has a single `claude-haiku-5-5` entry.
>
> Examples 2 and 3 exceed their limits by design; the SDK signals both by raising on the error result, which the example catches and prints. The $0.0001 cap in Example 3 is below the cost of a single `claude-haiku-5-5` request (about $0.0002-$0.0006 here), so it always trips.

---

## 13_file_checkpointing.py

```
$ uv run python 13_file_checkpointing.py

Working directory: /var/folders/.../claude_checkpoint_majlvwyr
Initial file content: Original content: Hello World

=== Turn 1: Modify the file ===
Checkpoint UUID: cfdd9e5a-4653-4a4e-8aa8-61680df9b4b1
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
Response: The three primary colors are red, yellow, and blue in traditional (paint or pigment)
color theory, while in light (additive RGB) they are red, green, and blue.
Session ID: 46d2a331-e36f-4977-903e-2ecbe19dddc1

--- Step 2: List sessions from store ---
Sessions in store: 1
  - 46d2a331-e36f-4977-903e-2ecbe19dddc1 (created: 1791758341109)

--- Step 3: Read messages from session 46d2a331-e36... ---
Total messages in transcript: 2
  [user] What are the three primary colors? Answer in one sentence.
  [assistant] The three primary colors are red, yellow, and blue in tradit
```

> SessionStore decouples transcript storage from local disk. The InMemorySessionStore mirrors the session in real-time via `session_store_flush="eager"`. `list_sessions_from_store` returns `SDKSessionInfo` entries (`session_id`, `created_at` in milliseconds since epoch), and `get_session_messages_from_store` returns `SessionMessage` objects whose `message` is the raw API message — string content for the prompt, a list of blocks for the reply.

**Verdict:** PASS - InMemorySessionStore captures session transcript, list and read functions work.

---

## 15. Deferred Tool Use (`15_deferred_tool_use.py`)

```
$ uv run python 15_deferred_tool_use.py

=== Deferred Tool Use (Human-in-the-Loop) ===
  [HITL Hook] Tool 'mcp__database__delete_record' requested with args: {'record_id': 'usr-42'}
  [HITL Hook] Deferring for human approval...

Result subtype: success

--- Deferred Tool Use ---
  Tool ID:    toolu_01GwmyPucfzDoNybk6iQb3WY
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
  Event:   PreToolUse
  Subtype: hook_started
  Hook:    PreToolUse:Glob

[HookEvent #2]
  Event:   PreToolUse
  Subtype: hook_response
  Hook:    PreToolUse:Glob
  Outcome: success (stdout: pre-glob-check)

[HookEvent #3]
  Event:   PostToolUse
  Subtype: hook_started
  Hook:    PostToolUse:Glob

[HookEvent #4]
  Event:   PostToolUse
  Subtype: hook_response
  Hook:    PostToolUse:Glob
  Outcome: success (stdout: post-glob-audit)

--- Result ---
24

Total hook events received: 4
```

> With `include_hook_events=True`, hook lifecycle events are streamed as `HookEventMessage` objects. All four events come from the two `echo` command hooks the example registers through `settings=`: `hook_name` is `<event>:<matcher>`, and each `hook_response` carries the command's outcome and stdout. The CLI only emits these events around hooks it runs itself (command, http, mcp_tool); in-process callback hooks passed through `hooks=...` don't produce them.

**Verdict:** PASS - HookEventMessage objects show each hook starting and finishing, with the outcome and output of the hook that produced them.

---

## 17. Strict MCP Config (`17_strict_mcp.py`)

```
$ uv run python 17_strict_mcp.py

=== Strict MCP Config Demo ===

Running with strict_mcp_config=True (no external MCP servers)...

[Tool] Glob
[Tool] Glob
[Tool] Grep

--- Result ---
There are 25 `.py` files in the current directory, excluding `.venv`:

- 00_hello_world.py
- 01_built_in_tools.py
- ...
- 23_session_rewind.py
- settings.py

The `.venv` directory holds about 700 more `.py` files from installed packages, which I left out.

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
Response: The **halting problem** asks whether it's possible to write a program that can look at
any other program and its input, and always correctly tell whether that program will eventually
finish (halt) or run forever.

**The intuition:** Some programs finish quickly. Others get stuck in an infinite loop.

=== Example 3: Thinking Disabled ===
Response: 2, 3, and 5.
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
Response: Code compiles at dawn,
one missing semicolon waits—
the bug hides in plain sight.

=== Example 2: TaskBudget with Token Limit ===
Response: # A Brief History of Python

I wrote this from established knowledge rather than running a web search, since the core
history is well documented. Details on recent releases may have changed, so check python.org
for current version information.

## Origins (1980s to early 1990s)

- **Creator:** Guido

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
Response: 15 × 23 = **345**

(15 × 20 = 300, plus 15 × 3 = 45, gives 345.)

=== Example 2: Full Sandbox Configuration ===
Response: The current directory contains 23 numbered Python example scripts (`00_hello_world.py`
through `23_session_rewind.py`), plus these other files and directories:

- **Directories:** `__pycache__/`, `.claude/`, `.venv/`
- **Config and env:** `.env`, `.env.example`, `.python-version`, `pyproject.toml`,

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
  extra deltas after interrupt: 0

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

> `extra deltas after interrupt` counts text deltas that arrive after `interrupt()` is sent: 0 here, though a few already in flight can still arrive on a fast model. Either way the turn ends as `aborted_streaming` with no result. The aborted turn still emits its own `ResultMessage`, so the example keeps iterating `receive_response()` to drain it; skipping that drain makes the next `query()` read the dead turn's buffered messages. Note `ClaudeSDKClient` yields error results as `ResultMessage`, whereas `query()` raises on them — which is why the client is used here.

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

=== Step 2: Transcript of cfafcffa ===
  [0] user      87c87b2e
  [1] assistant a5c52754  <- resume_session_at (last entry of turn 1)
  [2] user      dbb04196  <- resume_drops_turn (turn 2 prompt)
  [3] assistant 04412bb7

=== Step 3: Rewind to turn 1 and ask what is known ===
  Reply: colour=teal, city=unknown
  Forked session: 9f617ff9
  Original session still has 4 messages

=== Step 4: Declaring the wrong turn as dropped ===
  Refused with ResultError: Claude Code returned an error result: Resume rejected by
  --resume-drops-turn: resuming at a5c52754-... would discard entries not attributable to turn
  87c87b2e-...: range does not start with the declared turn prompt; first discarded entry 1
  [type=user, uuid=dbb04196-...] (exit code: 1)
```

> `city=unknown` is the proof the rewind worked: the forked session was loaded only up to the end of turn 1, so fact B never reached the model, while the original session keeps all 4 messages. `ResultError` is a subclass of `ProcessError`, and the CLI also writes the rejection line to stderr.

**Verdict:** PASS - `get_session_messages()` exposes the transcript UUIDs, `resume_session_at` + `resume_drops_turn` + `fork_session=True` drop the last turn, and a mismatched drop declaration is refused before any model call.

---

## 24. Stop Hook Feedback (`24_stop_hook_feedback.py`)

```
$ uv run python 24_stop_hook_feedback.py

Prompt: Use Glob to count the files matching '[0-9][0-9]_*.py', then Read settings.py and tell me how many examples there are and which default model they use.

  [PostToolUse] Glob  took 18 ms
  [PostToolUse] Read  took 8 ms

  [Stop] stop_hook_active=False, has TL;DR line=False
  [Stop] last_assistant_message: 'There are 26 examples (`00_hello_world.py` through `25_session_init_info.py`), and `settings.py` sets the default model to `claude-haiku-5-5` via `ANTHROPIC_MODEL_NAME`.'
  [Stop] -> missing TL;DR, returning additionalContext so the agent continues

  [Stop] stop_hook_active=True, has TL;DR line=True
  [Stop] last_assistant_message: 'There are 26 examples (`00_hello_world.py` through `25_session_init_info.py`), and `settings.py` sets the default model to `claude-haiku-5-5` via `ANTHROPIC_MODEL_NAME`.\nTL;DR: 26 examples; default model is claude-haiku-5-5.'
  [Stop] -> allow the agent to stop

--- Final answer ---
There are 26 examples (`00_hello_world.py` through `25_session_init_info.py`), and `settings.py` sets the default model to `claude-haiku-5-5` via `ANTHROPIC_MODEL_NAME`.
TL;DR: 26 examples; default model is claude-haiku-5-5.

Ends with a TL;DR line: True
```

> The first Stop sees an answer with no `TL;DR:` line, so the hook returns `hookSpecificOutput.additionalContext` and the CLI keeps the turn going instead of ending it. The second Stop arrives with `stop_hook_active=True` (and the line now present), so the hook lets the agent stop; checking that flag is what prevents an endless loop. `duration_ms` on `PostToolUseHookInput` is the tool's own execution time, excluding permission prompts and hook time.

**Verdict:** PASS - The Stop hook reads `last_assistant_message`, its `additionalContext` makes the agent continue and add the TL;DR line, and the PostToolUse hook prints per-tool `duration_ms`.

---

## 25. Session Init Info (`25_session_init_info.py`)

```
$ uv run python 25_session_init_info.py

=== Session init ===
Model:             claude-haiku-5-5
Claude Code:       2.1.309
Permission mode:   default
Tools:             25 (first five: Task, Bash, CronCreate, CronDelete, CronList)
MCP tools:         mcp__clock__get_time
MCP server:        clock -> connected
Capabilities:      ['interrupt_receipt_v1', 'interrupt_cancel_queued_v1', 'interrupt_send_now_v1', 'interrupt_requeue_taken_v1', 'msg_lifecycle_v1', 'request_marker_lists_v1', 'lifecycle_wakeup_source_v1', 'sdk_mcp_tools_list_changed', 'sdk_mcp_manifests', 'mcp_read_resource_v1', 'mcp_tool_ui_meta_v1', 'ui_surface_v1', 'mcp_call_result_not_for_model_v1']
Fast mode state:   off

interrupt_receipt_v1 supported: True

=== Result ===
ready
```

> The init `SystemMessage` arrives before any reply. `cast(SystemInitData, message.data)` gives typed access to its keys; the cast does not validate anything, so optional keys such as `capabilities` and `fast_mode_state` are read with `.get()` (`capabilities` is absent on CLIs before 2.1.205). The in-process `clock` server shows up as `connected` and its tool is in the tool list, which is how an app can confirm an MCP server is usable before relying on it.

**Verdict:** PASS - `SystemInitData` exposes model, CLI version, permission mode, tool count, MCP server status, and the `capabilities` list used to detect `interrupt_receipt_v1`.

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
| 14 | `14_session_store.py` | PASS | InMemorySessionStore transcript |
| 15 | `15_deferred_tool_use.py` | PASS | Human-in-the-loop deferral |
| 16 | `16_hook_events.py` | PASS | Hook event streaming from command hooks |
| 17 | `17_strict_mcp.py` | PASS | Strict MCP config isolation |
| 18 | `18_thinking_config.py` | PASS | Extended thinking modes |
| 19 | `19_task_budget.py` | PASS | Dollar cap and token budget |
| 20 | `20_sandbox_settings.py` | PASS | Sandboxed bash with network domain controls |
| 21 | `21_interrupt_and_terminal_reason.py` | PASS | interrupt() cancellation and terminal_reason |
| 22 | `22_verbatim_prompts.py` | PASS | verbatim_prompts keeps @path mentions literal |
| 23 | `23_session_rewind.py` | PASS | Drop the last turn with resume_session_at + resume_drops_turn |
| 24 | `24_stop_hook_feedback.py` | PASS | Stop hook additionalContext continues the turn, tool duration_ms |
| 25 | `25_session_init_info.py` | PASS | Typed SystemInitData, MCP status, capabilities |

**26/26 examples pass.**
