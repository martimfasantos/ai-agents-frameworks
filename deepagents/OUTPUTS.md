# Deep Agents Examples — Outputs

Captured output from running all examples with `deepagents==0.7.31`, `langchain==1.4.7`, `langchain-openai==1.6.2`, `langchain-quickjs==0.3.9`, and `gpt-4o-mini`.

LLM output is non-deterministic, so exact wording will vary between runs. The structure and demonstrated behavior stay the same.

> Every transcript below was recaptured for 0.7.4. That release ships a much leaner default system prompt (the authored base prompt is now empty), which changes what the model volunteers to do, so older captures no longer match.

---

## 00_hello_world.py

> **Feature:** The minimal Deep Agent created with `create_deep_agent`. The output shows a single agent answering a question — the simplest possible Deep Agents application.

```
=== Deep Agents Hello World ===

User: Where does the phrase 'hello world' come from?
Agent: The phrase "Hello, World!" originated from a simple programming example in the early days of computing, notably popularized by the 1978 book "The C Programming Language" by Brian Kernighan and Dennis Ritchie.
```

---

## 01_tools.py

> **Feature:** Giving an agent custom tools. The agent calls a `get_weather` tool twice (once per city) and synthesizes the results into a single answer.

```
=== Deep Agents Tools ===
[tool] get_weather called with city='Lisbon'
[tool] get_weather called with city='Tokyo'

User: What's the weather like in Lisbon and Tokyo?
Agent: The weather in Lisbon is sunny at 25°C, while Tokyo is cloudy at 19°C.
```

---

## 02_task_planning.py

> **Feature:** Task planning via `TodoListMiddleware`, which supplies the `write_todos` tool and the `todos` state channel. Since 0.7.0 this middleware is no longer in the default stack, so it is passed explicitly — without it there is no `write_todos` tool and `result["todos"]` is always empty.

```
=== Deep Agents Task Planning ===

Task: Plan a 3-step process for launching a simple blog: pick a platform, write a first post, and publish it.

Agent created 3 todo(s):
  1. [completed] Selected WordPress as the blogging platform for launching the blog.
  2. [completed] Draft the first blog post including title and content.
  3. [completed] Published the blog post on the chosen platform.

Summary: I have successfully completed all steps to launch the blog: 

1. Selected WordPress as the blogging platform.
2. Drafted the first blog post.
3. Published the blog post on the platform.

The blog is now live and ready for readers!
```

---

## 03_virtual_filesystem.py

> **Feature:** The built-in virtual filesystem (default `StateBackend`). The agent writes a file with `write_file`; we read it back out of the returned state under the `files` key.

```
=== Deep Agents Virtual Filesystem ===

Files in the virtual filesystem: ['/haiku.txt']

--- /haiku.txt ---
Waves whisper secrets,
Dancing on the golden shore,
Endless blue beckons.

Agent: I saved the haiku in a file named haiku.txt.
```

---

## 04_filesystem_permissions.py

> **Feature:** `FilesystemPermission` rules that guard filesystem paths. A deny rule blocks writes to `secret.txt` while allowing `notes.txt`. The second half shows that 0.7.0's recursive `delete` tool is classified as a `"write"` operation, so the same rules gate deletion: `/draft.txt` is removed but `/protected/config.txt` survives.

```
=== Deep Agents Filesystem Permissions ===

Files that were actually written: ['/notes.txt']
(secret.txt should be missing — the deny rule blocked it)

Agent: The file `notes.txt` was successfully created with the content 'hello', but writing to `secret.txt` was blocked due to permission restrictions.

=== Deletes are gated by the same write rules ===
Seeded files: ['/draft.txt', '/protected/config.txt']
Files remaining: ['/protected/config.txt']
(/draft.txt was deleted; /protected/config.txt survived the write-deny rule)

Agent: The deletion of `/draft.txt` succeeded, while the deletion of `/protected/config.txt` was blocked due to permissions.
```

---

## 05_structured_output.py

> **Feature:** Typed structured output via `response_format=<PydanticModel>`. The agent extracts fields from free text and returns a validated `ContactInfo` object read from `result["structured_response"]`.

```
=== Deep Agents Structured Output ===

Input text: Hey, I'm Ada Lovelace from Analytical Engines Inc. You can reach me at ada@analyticalengines.io.

Parsed ContactInfo object:
  name    = 'Ada Lovelace'
  email   = 'ada@analyticalengines.io'
  company = 'Analytical Engines Inc.'

Type: ContactInfo
```

---

## 06_streaming.py

> **Feature:** Streaming intermediate steps with `agent.stream(..., stream_mode="updates")`. Tool calls and tool results are printed as they arrive, before the final answer.

```
=== Deep Agents Streaming ===
User: Which is bigger by population, Tokyo or Cairo?

Streaming updates as they arrive:
  [model] tool call -> get_population({'city': 'Tokyo'})
  [model] tool call -> get_population({'city': 'Cairo'})
  [tools] tool result -> 22 million
  [tools] tool result -> 37 million

Agent: Tokyo is bigger by population, with about 37 million people compared to Cairo's 22 million.
```

---

## 07_subagents.py

> **Feature:** Delegation to a `SubAgent`. The main agent calls the built-in `task` tool to hand a subproblem to a specialized subagent, which uses its own `multiply` tool.

```
=== Deep Agents Subagents ===
[subagent tool] multiply(23.0, 17.0)

Tools the main agent called: ['task']
(the 'task' tool means work was delegated to a subagent)

Agent: The total number of boxes in the warehouse is 391.
```

---

## 08_composite_backend.py

> **Feature:** `CompositeBackend` routing paths to different backends. `/memories/` is routed to a durable `StoreBackend` while everything else stays in the ephemeral `StateBackend` — files land in the correct store based on their path prefix.

```
=== Deep Agents Composite Backend ===

StateBackend (ephemeral, thread state): ['/scratch.txt']
StoreBackend (durable, /memories/ route):
  namespace=('memories',) key=/profile.txt content='the user prefers tea'

Agent: The files have been created successfully.
```

---

## 09_human_in_the_loop.py

> **Feature:** Human-in-the-loop approvals via `interrupt_on` + a checkpointer. The agent pauses before a guarded `delete_file` tool call; a rejected request is not executed, an approved one runs.

```
=== Deep Agents Human-in-the-Loop ===

[Case 1] Reject the deletion:
  PAUSED — agent wants to call delete_file({'name': 'important.txt'})
  Human decision: REJECT
  Agent: The file important.txt could not be deleted.

[Case 2] Approve the deletion:
  PAUSED — agent wants to call delete_file({'name': 'temp.log'})
  Human decision: APPROVE
  [tool] delete_file('temp.log') EXECUTED
  Agent: The file temp.log has been deleted.
```

---

## 10_memory.py

> **Feature:** Long-term memory via the `memory=` parameter. An `AGENTS.md` file is loaded into the system prompt at startup, so the agent recalls persistent facts about the user.

```
=== Deep Agents Memory ===

Loaded memory from /memory/AGENTS.md
User: Remind me — who am I and what do I do for work?
Agent: You are Marie, a marine biologist based in Lisbon.
```

---

## 11_skills.py

> **Feature:** Skills — on-demand capabilities described in `SKILL.md` files. The agent discovers the `haiku-writer` skill, reads it with `read_file` (progressive disclosure), and follows its instructions — proven by the required `(fin)` marker. With 0.7.0's leaner default prompt the example's own `system_prompt` has to spell out the read-then-follow step; without it `gpt-4o-mini` answers from general knowledge and never opens the skill.

```
=== Deep Agents Skills ===

[agent read the skill] /skills/haiku-writer/SKILL.md

User: Please write a haiku about the ocean.
Agent:
Waves dance with the tide,  
Whispers of the deep call forth,  
Secrets held in blue. (fin)

Skill discovered & read: True
Skill instructions followed (ends with '(fin)'): True
```

---

## 12_local_filesystem_backend.py

> **Feature:** `FilesystemBackend` writing to a real directory on disk. The agent creates a file that we then read back directly from the filesystem. `virtual_mode=True` (agent path `/notes.txt` -> `<root_dir>/notes.txt`) became the default in 0.7.0 and is passed explicitly only to make the mapping visible.

```
=== Deep Agents Local Filesystem Backend ===
Workspace on disk: /var/folders/qm/1yt5968d2fq18bjj_zff6jtc0000gn/T/deepagents_fs_s144ign_

Agent: The file **notes.txt** has been created containing the text: **Hello from disk**.

Real files written to disk:
  notes.txt -> 'Hello from disk'
```

---

## 13_store_backend.py

> **Feature:** `StoreBackend` for durable, cross-conversation storage. Conversation 1 saves a file; a brand-new agent in conversation 2 — with no shared message history — reads it back from the store.

```
=== Deep Agents Store Backend ===

[Conversation 1 - write] Agent: The file **preferences.txt** has been saved with the content: "favorite color is teal."

Files now living in the durable store:
  key=/preferences.txt content='favorite color is teal'

[Conversation 2 - read] Agent: Your favorite color is teal.

(Conversation 2 had no shared message history — only the store persisted it.)
```

---

## 14_summarization.py

> **Feature:** Conversation summarization via the `compact_conversation` tool (`SummarizationToolMiddleware`) with a low token trigger. The agent folds a bloated conversation into a compact summary on demand. `system_prompt` defaults to `None` since 0.7.0, so the compaction nudge is passed explicitly.

```
=== Deep Agents Summarization ===

Messages after a detailed turn: 2

[agent called tool] compact_conversation
[tool result] Conversation compacted. Summarized 2 messages into a concise summary.

Agent: The conversation has been compacted, and the older messages have been summarized to free up context space. If you have another topic or question, feel free to ask!
```

---

## 15_custom_middleware.py

> **Feature:** A custom `AgentMiddleware` using `wrap_model_call`. The middleware counts model invocations and deterministically stamps a sign-off line onto the final answer by rewriting the response.

```
=== Deep Agents Custom Middleware ===
  [middleware] intercepted model call #1

User: In one sentence, what is a vector database?
Agent:
A vector database is a type of database designed to store and query data represented as high-dimensional vectors, often used in applications involving machine learning and similarity searches.
-- handled by SignOffMiddleware

Model calls counted by middleware: 1
Response reshaped by middleware: True
```

---

## 16_runtime_context.py

> **Feature:** Runtime context via a `context_schema` and `ToolRuntime`. Request-scoped data (user, tier) is passed at invoke time and read inside a tool — the same agent answers differently per user with no code changes.

```
=== Deep Agents Runtime Context ===

context=UserContext(user_name='Marie', tier='gold')
Agent: You are Marie, and you are on the gold tier.

context=UserContext(user_name='Tom', tier='free')
Agent: You are Tom, and you are on the free tier.
```

---

## 17_rubric_middleware.py

> **Feature:** `RubricMiddleware` for self-evaluated iteration. A grader sub-agent checks the answer against a rubric; the first attempt needs revision, and the agent iterates until the grader is satisfied.

```
=== Deep Agents Rubric Middleware ===
Rubric: exactly 3 numbered benefits + a final 'STATUS: COMPLETE' line

  [grader] iteration 0: result=needs_revision
  [grader] iteration 1: result=satisfied

Grading iterations: 2

Final answer (passed the rubric):
Here are three key benefits of unit testing:

1. **Improved Code Quality**: Unit testing helps identify bugs early in the development process, leading to cleaner and more reliable code.

2. **Facilitates Changes**: A solid suite of unit tests allows developers to refactor or modify code confidently, reducing the risk of introducing new bugs.

3. **Faster Debugging**: When a unit test fails, it pinpoints the location of the issue, making it quicker to diagnose and fix the problem.

STATUS: COMPLETE
```

---

## 18_harness_profiles.py

> **Feature:** `HarnessProfile` registered per model with `register_harness_profile`. Both halves of the profile are proven: the prompt suffix forces the `AHOY:` prefix, and `excluded_tools` strips `write_todos` before the model sees it, so no todos reach state even though `TodoListMiddleware` really installed the tool. Note `excluded_tools` is a plain name filter — excluding a name nothing installed is a silent no-op.

```
=== Deep Agents Harness Profiles ===
Registered profile for: openai:gpt-4o-mini
  system_prompt_suffix: enforces the 'AHOY:' house style
  excluded_tools: {'write_todos'}

Tools installed by TodoListMiddleware: ['write_todos']

User: Plan a 2-step process for brewing tea, then answer briefly.
Agent: AHOY: 

1. Boil water to the desired temperature depending on the type of tea.
2. Steep the tea leaves or tea bags in the hot water for the recommended time.

House-style suffix applied (answer starts with 'AHOY:'): True
write_todos installed in the harness: True
Todos recorded in state after the run: 0 -> the profile stripped write_todos before the model ever saw it
```

---

## 19_interpreters.py

> **Feature:** A sandboxed code interpreter via the QuickJS `CodeInterpreterMiddleware` (`eval` tool). The agent writes JavaScript, executes it in an isolated engine, and grounds its answer in the real computed result.

```
=== Deep Agents Interpreters ===

[agent ran JS]
let sumOfSquares = 0; for (let i = 1; i <= 20; i++) { sumOfSquares += i * i; } sumOfSquares;

[sandbox output] <result>2870</result>

User: Using the eval tool, compute the sum of squares from 1 to 20. Show the number.
Agent: The sum of squares from 1 to 20 is **2870**.
```

---

## 20_filesystem_tool_allowlist.py

> **Feature:** `FilesystemMiddleware(tools=[...])` narrows the filesystem tool set (typed by the exported `FsToolName`), and passing the configured instance through `middleware=[...]` replaces the default `FilesystemMiddleware` by `.name` match. The resulting agent can read but not mutate: `write_file`, `edit_file` and `delete` are gone from its tool node, and the delete request fails. Omitting `read_file` from the list is rejected with a `ValueError`.

```
=== Deep Agents Filesystem Tool Allowlist ===
FsToolName options: ['ls', 'read_file', 'write_file', 'edit_file', 'delete', 'glob', 'grep', 'execute']
Default agent tools: ['delete', 'edit_file', 'execute', 'glob', 'grep', 'ls', 'read_file', 'task', 'write_file']

Allowlisted middleware 'FilesystemMiddleware' exposes: ['ls', 'read_file', 'glob', 'grep']
Agent tools after the override: ['glob', 'grep', 'ls', 'read_file', 'task']
Mutating tools write_file / edit_file / delete are absent ✅

FilesystemMiddleware(tools=['ls', 'glob']) -> ValueError: read_file must be included in tools; it is required by FilesystemMiddleware

User: Read /report.md and tell me the revenue figure, then delete the file.
Agent: The revenue figure mentioned in the report is a growth of 12%. However, I'm unable to delete the file as that functionality is not supported.

Files still present: ['/report.md']

=== Replacement middleware + permissions=[deny read /internal/**] ===
User: Read /internal/roadmap.md and summarize it.
[tool result] read_file: Error: permission denied for read on /internal/roadmap.md
Agent: I can't access the file `/internal/roadmap.md` due to permission restrictions.
Denied content leaked into the reply: False

Conflicting middleware permissions= -> ValueError: FilesystemMiddleware permissions conflict with the agent permissions; use the same rules or omit middleware permissions.

Caveat: tools= only hides tools; it does not restrict which paths the remaining tools can touch. Enforce that with permission rules, either on create_deep_agent(permissions=...), which a replacement FilesystemMiddleware inherits, or on FilesystemMiddleware(permissions=...) directly. Interrupt (human-approval) rules still have to go through create_deep_agent(permissions=...).
```

> An allowlist keeps the model's options tidy; it is not a security boundary, permission rules are. Since 0.7.31 a replacement `FilesystemMiddleware` inherits the agent's `create_deep_agent(permissions=...)` rules, so the deny rule on `/internal/**` still fires through the read-only instance (the `read_file` tool result is a permission error). Passing a different non-empty `FilesystemMiddleware(permissions=...)` list alongside agent rules raises `ValueError` instead of silently picking one.

---

## 21_subagent_fork.py

> **Feature:** `mode="fork"` on a subagent spec (new in 0.7.12). Both runs delegate the same context-free task string, but the isolated subagent replies `NO CONTEXT.` while the fork answers from the parent's earlier turn. The output reads the `task` ToolMessage rather than the parent's final message — the parent can see the history either way, so its own answer would hide the difference. Forked subagents emit a `LangChainBetaWarning` on stderr.

```
=== Isolated vs forked subagents ===

mode='isolated'
  delegated task: 'Which day does the backfill run?'
  subagent returned: NO CONTEXT.

mode='fork'
  delegated task: 'which day does the backfill run?'
  subagent returned: The backfill runs on Monday.

Neither task string carried the plan — only the fork inherited it.
```

---

## 22_grader_hooks.py

> **Feature:** the rubric grader integration hooks added in 0.7.11. `prepare_messages_for_grader` cuts the transcript to the final answer and redacts the `INTERNAL:` line; `build_grader_state` + `grader_state_schema` hand the nested grader a house style guide that never appears in the conversation; `grader_middleware` pulls that field into the grader's system prompt. The `[hook]` lines show each one firing before the grader returns `satisfied`.

```
=== Rubric grader integration hooks ===

  [hook] grader sees 1 of 2 messages
  [hook] redacted an internal note
  [hook] injecting style guide for grader iteration 0
  [grader] iteration 0: result=satisfied

Final answer:
  A deploy freeze is a period during which no new code is allowed to be deployed to production, often to stabilize the system before major releases or updates.  
INTERNAL: drafted by the assistant  
EOM

The grader never saw the INTERNAL note, and graded against a style
guide that was never part of the conversation.
```

---

## 23_skill_tools.py

> **Feature:** Skill-scoped tools (new in 0.7.22). `lookup_order` is passed to a `SkillsMiddleware(tools=[...])` that replaces the default skills middleware, and the `order-support` skill lists it under `metadata.include_tools`. The tool is never registered on the agent's tool node, and a model callback shows it was not offered on the first model call — only after `read_file` on the `SKILL.md` does it appear, and the agent then calls it. Without the read, a call to `lookup_order` fails as an unknown tool.

```
=== Deep Agents Skill Tools ===
Tools registered on the agent: ['delete', 'edit_file', 'execute', 'glob', 'grep', 'ls', 'read_file', 'task', 'write_file']
lookup_order registered: False

User: Where is my order A-1042?
  [model call 1] lookup_order offered: False
  [tool] read_file({'file_path': '/skills/order-support/SKILL.md', 'offset': 0, 'limit': 1000})
  [model call 2] lookup_order offered: True
  [tool] lookup_order({'order_id': 'A-1042'})
  [model call 3] lookup_order offered: True

Tool calls in order: read_file -> lookup_order
Agent: Your order A-1042 was shipped on October 9 via DHL and is expected to arrive on October 13.
```

---

## 24_pinned_skills.py

> **Feature:** Pinned skills (new in 0.7.23). The app finds `/haiku-writer` in the message and passes `pinned_skills=["haiku-writer"]` in the run input; `SkillsMiddleware` appends the `SKILL.md` body as a `HumanMessage` tagged `additional_kwargs["lc_source"] == "pinned_skill"` before the model call, so the reply carries the skill's `(fin)` marker with no `read_file`. The unpinned run depends on the model choosing the skill: here it ignored it, but on some runs it reads `SKILL.md` itself and follows it. `pinned_skills` is cleared after use and is not returned in the output state.

```
=== Deep Agents Pinned Skills ===
User: /haiku-writer the first frost of autumn

[Run with pinned_skills=[]]
  read_file calls: none
  Agent:
    Here’s a haiku about the first frost of autumn:

    Crisp leaves fall gently,  
    Whispers of winter's approach,  
    First frost paints the dawn.
  Followed the skill ('(fin)' marker present): False

[Run with pinned_skills=['haiku-writer']]
  [pinned skill injected] name=haiku-writer path=/skills/haiku-writer/SKILL.md
  read_file calls: none
  Agent:
    Here’s a haiku inspired by the first frost of autumn:

    Silent icy breath,  
    Drapes the world in crystal frost,  
    Whispers of the cold.  
    (fin)
  Followed the skill ('(fin)' marker present): True
```

---

## 25_custom_summarizer.py

> **Feature:** `SummarizationMiddleware(summarizer=...)` (new in 0.7.26) replaces who writes the conversation summary while the middleware still decides when to summarize and what to keep. Passed via `middleware=[...]`, it replaces the default `SummarizationMiddleware` by `.name`. With `trigger=("messages", 6)` and `keep=("messages", 2)`, summarization fires on the fourth turn and folds the first 5 messages. Mode A uses a plain function (one call, no LLM, deterministic output); Mode B uses a model string prompted with a custom `summary_prompt`. Either way the agent still answers the question from the summary.

```
=== Deep Agents Custom Summarizer ===

=== Mode A: summarizer=user_facts_summarizer ===
Middleware name: SummarizationMiddleware
User: Our project codename is Bluefin.
  Agent: Noted.
User: The launch date is March 14.
  Agent: Noted.
User: The budget is 2 million euros.
  Agent: Noted.
User: What are the codename, launch date and budget? Answer in one sentence.
  Agent: The project codename is Bluefin, the launch date is March 14, and the budget is 2 million euros.

Summarization fired: True
Messages folded into the summary: 5
Summary message the model saw instead of those turns:
    You are in the middle of a conversation that has been summarized.

    The full conversation history has been saved to /conversation_history/session_a616b6c553064763af07e492c55949b6.md should you need to refer back to it for details.

    A condensed summary follows:

    <summary>
    User-stated facts:
    - Our project codename is Bluefin.
    - The launch date is March 14.
    - The budget is 2 million euros.
    </summary>

Function summarizer calls (messages received): [5]

=== Mode B: summarizer='openai:gpt-4o-mini' ===
Middleware name: SummarizationMiddleware
User: Our project codename is Bluefin.
  Agent: Noted.
User: The launch date is March 14.
  Agent: Noted.
User: The budget is 2 million euros.
  Agent: Noted.
User: What are the codename, launch date and budget? Answer in one sentence.
  Agent: The project codename is Bluefin, the launch date is March 14, and the budget is 2 million euros.

Summarization fired: True
Messages folded into the summary: 5
Summary message the model saw instead of those turns:
    You are in the middle of a conversation that has been summarized.

    The full conversation history has been saved to /conversation_history/session_2d4dc25dc4954a78b74d312a857984c9.md should you need to refer back to it for details.

    A condensed summary follows:

    <summary>
    - Project codename is Bluefin.
    - Launch date is March 14.
    - Budget is 2 million euros.
    </summary>
```

> The summary is read from the private `_summarization_event` state key (`SUMMARIZATION_EVENT_KEY`) through the checkpointer; it is not part of `result["messages"]`. The offload file under `/conversation_history/` gets a new session id each run. Mode B reuses `settings.OPENAI_MODEL_NAME` to avoid hardcoding a model name; in practice point `summarizer=` at a cheaper model than the agent's.
