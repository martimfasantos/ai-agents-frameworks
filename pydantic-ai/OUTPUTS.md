# Pydantic AI — Example Outputs

All examples run with `pydantic-ai>=1.70.0`, model `gpt-4o-mini`.

> **Note:** LLM responses are non-deterministic. Your outputs will differ in wording
> but should follow the same structure and demonstrate the same features.

---

## 00. Hello World (`00_hello_world.py`)

```
"Hello, World!" is a simple program often used as a beginner's introduction to programming,
popularized by the 1978 book "The C Programming Language" by Brian Kernighan and Dennis Ritchie.
```

> The agent responds concisely in one sentence, as instructed.

---

## 01. Tools and Metrics (`01_tools_and_metrics.py`)

```
=== Example 1: Basic Tool Usage ===
🎲 Die rolled: 6
🎮 Game Result: The die rolled a 6, but your guess was 4. Better luck next time, Alice!
    Want to play again?

📈 Final Metrics:
   - Total requests: 2
   - Tool calls executed: 2
   - Input tokens used: 241
   - Output tokens generated: 68

============================================================

=== Example 2: Advanced Tool Registration Patterns and Usage Limits ===
🌤️  Weather Agent:
   Response: The current weather in London is rainy, with a temperature of 15°C
   and humidity at 82%.
   Tool Calls: 1 tool calls

============================================================

=== Example 3: Message History and Tool Inspection ===
🔍 Research Result: AI, or artificial intelligence, refers to the simulation of human
   intelligence processes by machines, particularly computer systems...

📋 Message History Analysis:
   Message 1: ModelRequest
      Part 1: SystemPromptPart
      Part 2: UserPromptPart
   Message 2: ModelResponse
      Part 1: ToolCallPart - search_database
   Message 3: ModelRequest
      Part 1: ToolReturnPart - AI is cool...
   Message 4: ModelResponse
      Part 1: TextPart

📈 Final Metrics:
   - Total requests: 2
   - Tool calls executed: 1
   - Input tokens used: 157
   - Output tokens generated: 126

============================================================
```

> Demonstrates custom tools (`@agent.tool`), usage metrics tracking, message history inspection, and usage limits.

---

## 02. Dependencies (`02_dependencies.py`)

```
=== Example 1: Free-Tier Customer ===
Response: You are eligible for a Pro upgrade at $9.99 per month.
    Would you like more information about the benefits of the Pro plan?

=== Example 2: Pro Customer with Purchases ===
Response: Your Premium Gadget order was delivered on March 15, 2025,
    and the order number is #ORD-42178. If you need any further assistance,
    feel free to ask!

=== Example 3: Enterprise Customer ===
Response: You are currently on the 'enterprise' plan, which is the highest tier
    available. Therefore, you are not eligible for an upgrade.
```

> Each example injects different customer data via `deps_type` and `RunContext`, showing how the agent adapts behavior based on runtime dependencies.

---

## 03. Built-in Tools (`03_built_in_tools.py`)

```
=== Example 1: Web Search Tool ===
Response: As of March 24, 2026, Pydantic AI is at version 1.0.1,
    released on September 5, 2025.

=== Example 2: Code Execution Tool ===
Response: The first 10 Fibonacci numbers are:
    [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]

=== Example 3: Combined Web Search + Code Execution ===
Response: The value of 2^100 is 1,267,650,600,228,229,401,496,703,205,376.
```

> Uses `OpenAIResponsesModel` with `WebSearchTool` and `CodeExecutionTool` from `pydantic_ai.builtin_tools`.

---

## 04. Structured Outputs (`04_structured_outputs.py`)

```
=== Basic Structured Output ===
Output: city='London' country='United Kingdom'
Type: <class '__main__.CityLocation'>

=== Union Types ===
Person Output: name='John' age=25
Animal Output: species='Lion' habitat='African savanna'

=== Tool Output Mode ===
Tool Output: name='Sarah' age=30

=== Native Output Mode ===
Native Output: name='Mike' age=28

=== Prompted Output Mode ===
Prompted Output: name='Lisa' age=35
```

> Shows Pydantic models as output types, union types, and the three output modes: `ToolOutput`, `NativeOutput`, and `PromptedOutput`.

---

## 05. Output Validators (`05_output_validators.py`)

```
=== Test 1: Valid User ===
Output: username='newuser' email='new@example.com' age=25
Username: newuser
Email: new@example.com

--------------------------------------------------

=== Test 2: Existing Username (will retry) ===
Output: username='admin123' email='fresh@example.com' age=30
New username suggested: admin123

--------------------------------------------------

📈 Final Metrics:
   - Total requests: 2
   - Tool calls executed: 0
   - Input tokens used: 203
   - Output tokens generated: 65

============================================================

Once upon a time, in a quaint little village nestled between lush green hills
and a shimmering river, there lived a cat named Whiskers...
...
One sunny afternoon, as Whiskers lounged in the meadow chasing after
butterflies, he overheard a group of children laughing and whispering about a
hidden treasure...
```

> Test 1 passes validation immediately. Test 2 triggers `ModelRetry` because "admin" is taken — the agent retries and suggests "admin123". The streaming section shows partial output validation accepting incremental chunks.

---

## 06. Output Functions (`06_output_functions.py`)

```
=== Example 1: TextOutput Post-Processing ===
Output: {'text': 'Python is a high-level, interpreted programming language known
    for its readability and simplicity...', 'word_count': 74, 'char_count': 574}
Type: <class 'dict'>

=== Example 2: TextOutput for Format Conversion ===
Output lines: ['IMPROVES CARDIOVASCULAR HEALTH.',
    'BOOSTS MENTAL WELL-BEING AND REDUCES ANXIETY.',
    'ENHANCES STRENGTH AND FLEXIBILITY.']
Type: <class 'list'>

=== Example 3: Mixed Output Types ===
Factual output: answer='The capital of France is Paris.' confidence=1.0
Type: StructuredAnswer
```

> `TextOutput` enables post-processing of plain text (word counting, format conversion). Mixed output types combine structured and text outputs.

---

## 07. Streaming (`07_streaming.py`)

```
=== Streaming with Custom Handler ===
[EVENT] Tool called: get_weather with args: {"location":"Paris","date":"2023-10-05"}
[EVENT] Final result started (tool: None)
Tomorrow in Paris, the weather will be sunny with a temperature of 24°C.

=== Stream All Events and filtering ===
[STREAM EVENT] Tool call: get_weather
[STREAM EVENT] Final result event: None
```

> Demonstrates `run_stream` with custom event handlers that intercept tool calls and final result events in real time.

---

## 08. Message History (`08_message_history.py`)

```
=== Basic Conversation ===
Response: The joke is a play on words that combines a well-known brand (Colgate)
    with the idea of a "scandal." The humor comes from the pun...

=== Message Inspection ===
Total messages in conversation: 3
   Message 1: ModelResponse
      Part 1: TextPart
   Message 2: ModelRequest
      Part 1: UserPromptPart
   Message 3: ModelResponse
      Part 1: TextPart

=== Storing and Loading Messages ===
Serialized 2226 bytes to JSON
Loaded 3 messages from JSON
Response of History Agent using loaded history: We discussed a joke that involved
    a play on words related to the Colgate brand...

=== New History after Context-Aware Processing ===
   Message 1: ModelRequest
      Part 1: UserPromptPart
   Message 2: ModelRequest
      Part 1: UserPromptPart
   Message 3: ModelResponse
      Part 1: TextPart
```

> Shows multi-turn conversations, message serialization to JSON, loading history into new agents, and message structure inspection.

---

## 09. Agent Delegation (`09_agent_delegation.py`)

```
=== Simple Agent Delegation Example ===

  Generating 5 jokes...
Selected joke:
Here's a cat joke that I think is the best:

**Why was the cat sitting on the computer? Because it wanted to keep an eye
on the mouse!**

📈 Final Metrics:
   - Total requests: 3
   - Tool calls executed: 1
   - Input tokens used: 294
   - Output tokens generated: 163

============================================================
```

> A "selector" agent delegates joke generation to a "joke generator" agent via a tool call, then picks the best one.

---

## 10. Programmatic Handoff (`10_programmatic_handoff.py`)

```
=== Programmatic Hand-Off Example ===

Step 1: Flight Search
============================================================
Flight found: AK456 (Lisbon -> London)
Messages from flight agent: 5

Step 2: Seat Selection
============================================================
Seat selected: Row 1, Seat A

Step 3: Booking Summary
============================================================
  Flight: AK456
  Route:  Lisbon -> London
  Seat:   Row 1, Seat A

📈 Final Metrics:
   - Total requests: 3
   - Tool calls executed: 1
   - Input tokens used: 577
   - Output tokens generated: 68

============================================================
```

> Three specialized agents (flight search, seat selection, summary) are chained programmatically, passing structured outputs between steps.

---

## 11. Toolsets (`11_toolsets.py`)

```
=== Example 1: FunctionToolset ===
Response: The weather in Paris is sunny with a temperature of 22°C.
    The population of Paris is approximately 2.1 million.

=== Example 2: PrefixedToolset ===
Response: The weather in London is cloudy with a temperature of 14°C.
    Additionally, 100 USD is equivalent to 92.00 EUR.

=== Example 3: FilteredToolset ===
Response: The weather in Tokyo is currently rainy with a temperature of 18°C.

Filtered toolset only exposes weather tools, population tool is hidden.
```

> Demonstrates `FunctionToolset`, `PrefixedToolset`, `FilteredToolset`, and `CombinedToolset` for composable tool management.

---

## 12. MCP Client (`12_mcp_client.py`)

```
=== MCP Client Example ===

MCP connection error: unhandled errors in a TaskGroup (1 sub-exception)

This is expected if the MCP server binary is not installed.
The example demonstrates the MCP client configuration pattern.
```

> Exits gracefully when no MCP server is available. The code demonstrates the `MCPServerStdio` configuration pattern for connecting agents to MCP-compatible tool servers.

---

## 13. Agent Iteration (`13_agent_iteration.py`)

```
=== Example 1: Step-by-Step Iteration ===

  Step 1: UserPromptNode
  Step 2: ModelRequestNode
  Step 3: CallToolsNode
  Step 4: ModelRequestNode
  Step 5: CallToolsNode
  Step 6: End

Final output: The temperature in Lisbon is 26°C.
Total steps: 6

=== Example 2: Inspect Messages During Iteration ===

  Node: UserPromptNode | Messages so far: 0
  Node: ModelRequestNode | Messages so far: 0
  Node: CallToolsNode | Messages so far: 2
  Node: ModelRequestNode | Messages so far: 2
  Node: CallToolsNode | Messages so far: 4
  Node: End | Messages so far: 4

Final output: The temperature in Paris is 22°C, and in Tokyo, it is 18°C.
Total messages: 4

=== Example 3: Usage Tracking During Iteration ===

  UserPromptNode: requests=0, tool_calls=0, tokens=0
  ModelRequestNode: requests=0, tool_calls=0, tokens=0
  CallToolsNode: requests=1, tool_calls=0, tokens=115
  ModelRequestNode: requests=1, tool_calls=2, tokens=115
  CallToolsNode: requests=2, tool_calls=2, tokens=281
  End: requests=2, tool_calls=2, tokens=281

Final output: The current temperatures are as follows:
- London: 14°C
- Lisbon: 26°C

Lisbon is significantly warmer than London at the moment.
Final usage: 2 requests, 2 tool calls
```

> Uses `agent.iter()` to step through agent execution node-by-node, inspecting messages and usage at each step.

---

## 14. Stateful Graphs (`14_stateful_graphs.py`)

```
=== Stateful Graph Example ===

Vending Machine Workflow
============================================================
  Inserted $1.00
   Balance: $1.00
Available products:
   - water: $1.25
   - soda: $1.50
   - crisps: $1.75
   - chocolate: $2.00
  Selected: soda
  Insufficient funds for soda
   Need $0.50 more
  Inserted $0.50
   Balance: $1.50
  Purchased soda!
  Change returned: $0.00

============================================================
Result: Enjoy your purchase!
Final state:
   Balance: $0.00
   Product: soda

Mermaid Diagram of Graph:
---
title: vending_machine_graph
---
stateDiagram-v2
  [*] --> InsertCoin
  InsertCoin --> CoinsInserted
  CoinsInserted --> SelectProduct
  CoinsInserted --> Purchase
  SelectProduct --> Purchase
  Purchase --> InsertCoin
  Purchase --> SelectProduct
  Purchase --> [*]
```

> A deterministic vending machine modeled with `pydantic-graph` — `BaseNode`, `End`, `Graph`, and mermaid diagram generation.

---

## 15. Graphs with GenAI (`15_graphs_with_genai.py`)

```
=== Graphs with GenAI: Content Review Pipeline ===

============================================================
  [Writer] Draft v1: Open source software offers numerous benefits,
      including enhanced collaboration...
  [Reviewer] APPROVED

============================================================

Final content: Open source software offers numerous benefits, including
    enhanced collaboration and innovation, as developers worldwide can
    contribute and improve the code. It promotes transparency, allowing
    users to inspect, modify, and secure the software according to their
    needs...
Revisions: 1
Approved: True

Revision history (1 versions):
  - Draft v1: Open source software offers numerous benefits, including enhanced coll...

Mermaid Diagram:
---
title: review_pipeline
---
stateDiagram-v2
  [*] --> WriteDraft
  WriteDraft --> ReviewDraft
  ReviewDraft --> WriteDraft
  ReviewDraft --> [*]
```

> Combines `pydantic-graph` with Pydantic AI agents: a writer agent drafts content, a reviewer agent approves or requests revisions, looping until approved.

---

## 16. Human-in-the-Loop (`16_human_in_the_loop.py`)

```
=== Human-in-the-Loop Tool Approval ===

Step 1: Initial agent run
============================================================
Agent needs approval for 2 tool calls:

  1. read_file({"filename": ".env"})
  2. delete_file({"filename": "temp.log"})

Step 2: Human decision-making
============================================================
  Denied: read_file('.env')
   Reason: Cannot read sensitive configuration
  Deleting files is not allowed

Step 3: Continue execution with approved Reads but denied Deletes
============================================================

Final result:
I was able to read the contents of both the README.md and .env files
successfully. However, I couldn't delete the temp.log file as file
deletion is not permitted.
```

> Uses `DeferredToolRequests` to intercept tool calls before execution, simulating human approval/denial of sensitive operations.

---

## 17. Evals (`17_evals.py`)

```
=== Pydantic Evals: Capital Cities Quiz ===

Evaluating ask_capital ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00
  Evaluation Summary:
      ask_capital
┏━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Case ID   ┃ Duration ┃
┡━━━━━━━━━━━╇━━━━━━━━━━┩
│ france    │  607.2ms │
├───────────┼──────────┤
│ japan     │  607.5ms │
├───────────┼──────────┤
│ portugal  │  606.8ms │
├───────────┼──────────┤
│ australia │  607.5ms │
├───────────┼──────────┤
│ Averages  │  607.2ms │
└───────────┴──────────┘
```

> Uses `pydantic-evals` with `Dataset`, `Case`, and `evaluate_sync` to run a structured evaluation of agent responses against expected answers.

---

## 18. A2A Protocol (`18_a2a.py`)

```
(Server file — not auto-executed)

Verified: imports succeed and FastA2A app is created.
Run with: uvicorn 18_a2a:app
```

> Exposes a Pydantic AI agent as an A2A-compatible ASGI server using `agent.to_a2a()` from the `fasta2a` package.
