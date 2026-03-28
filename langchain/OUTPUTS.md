# LangChain Examples — Output Log

All 17 examples executed successfully with `gpt-4o-mini` on 2026-03-21.

> Outputs are non-deterministic — your results will vary slightly on each run.

---

## 00_hello_world.py

```
The phrase "Hello, World!" originated from the 1972 programming language tutorial for the B
implemented by Brian Kernighan and has since become a common example for teaching programming.
```

---

## 01_chat_models.py

```
=== Example 1: init_chat_model ===
Response: The capital of Portugal is Lisbon.

=== Example 2: ChatOpenAI ===
Response: The capital of Japan is Tokyo.

=== Example 3: Multi-turn Conversation ===
Turn 1: 2 + 2 equals 4.
Turn 2: 4 multiplied by 10 equals 40.
```

---

## 02_tools.py

```
Weather query: The weather in Lisbon is sunny with a temperature of 25°C.

Math query: The result of 42 × 17 + 3 is 717.
```

---

## 03_streaming.py

```
=== Example 1: Stream Updates ===
[model] {'messages': [AIMessage(content='', ..., tool_calls=[{'name': 'get_population',
    'args': {'country': 'Portugal'}, ...}], ...)]}

[tools] {'messages': [ToolMessage(content='10.3 million', name='get_population', ...)]}

[model] {'messages': [AIMessage(content='The population of Portugal is approximately
    10.3 million.', ...)]}

=== Example 2: Stream Messages (token-by-token) ===
A fun fact about Japan's population is that it is approximately 125 million people, making
it one of the most populous countries in the world, yet it has one of the highest life
expectancies, contributing to a significant percentage of elderly citizens.
```

---

## 04_structured_output.py

```
=== Example 1: Structured Movie Review ===
Title: Inception
Year: 2010
Genre: sci-fi
Rating: 8.8/10
Summary: A skilled thief, Dom Cobb, is offered a chance to have his criminal history erased
if he can successfully plant an idea into a target's subconscious.

=== Example 2: Structured City Info ===
City: Lisbon, Portugal
Population: 0.55M
Known for: pastéis de nata, fado music, tram 28
```

---

## 05_short_term_memory.py

```
=== Thread 'abc': Multi-turn conversation ===
Turn 1: The current time in UTC is 14:30.

Turn 2: The current time in EST is 09:30.

Turn 3: Yes, your name is Alice.

=== Thread 'xyz': Fresh conversation ===
Turn 1: No, I don't know your name. You haven't provided it.
```

---

## 06_runtime_context.py

```
=== User 42: Alice ===
Here is your profile information:

- **User ID:** user_42
- **Name:** Alice
- **Preferred Language:** English

And here are your recent orders:

1. **Order ID:** ORD-001
   - **Item:** Wireless Mouse
   - **Status:** Delivered

2. **Order ID:** ORD-002
   - **Item:** USB-C Hub
   - **Status:** Shipped

=== User 99: Bob ===
You have one recent order:

- **Order ID:** ORD-100
  **Item:** Keyboard
  **Status:** Processing

If you need further assistance with this order or anything else, just let me know!
```

---

## 07_middleware.py

```
=== Agent with Middleware ===
  [before_model] Processing for Alice, 1 messages
  [after_model] Response preview: (no content)
  [before_model] Processing for Alice, 3 messages
  [after_model] Response preview: Did you know that Finland consumes the most coffee per
    capita in the world?

Final: Did you know that Finland consumes the most coffee per capita in the world?
```

---

## 08_guardrails.py

```
=== Safe Request ===
  [guardrail] Input passed content filter
  [guardrail] Output has 106 words
Response: It seems I couldn't find specific documentation on resetting your password.
However, typically, you can reset your password by following these general steps:

1. **Go to the login page** of the service you are trying to access.
2. Look for a link or button that says **"Forgot Password?"** or **"Reset Password."**
3. Click on this link, and you will usually be prompted to enter your email address or
   username.
4. Check your email for a password reset link or instructions.
5. Follow the link to create a new password.

If you need more specific guidance, please let me know what service or platform you're
referring to!

=== Blocked Request ===
  [guardrail] BLOCKED: found banned keyword 'hack'
  [guardrail] Output has 11 words
Response: I cannot process requests containing inappropriate content. Please rephrase your
request.
```

---

## 09_human_in_the_loop.py

```
=== Requesting email send (will interrupt) ===
Interrupt received!
  Action requests: [{'name': 'send_email', 'args': {'to': 'alice@example.com',
    'subject': 'Refund Policy Information',
    'body': 'Dear Alice,\n\nI hope this message finds you well. I wanted to share some
    important information regarding our refund policy.\n\nPlease let me know if you have
    any questions or need further clarification.\n\nBest regards,\n\n[Your Name]'},
    'description': "Tool execution requires approval\n\nTool: send_email\nArgs: {...}"}]

=== Approving the action ===
Final response: The email has been successfully sent to alice@example.com regarding the
refund policy. If you need anything else, feel free to ask!
```

---

## 10_long_term_memory.py

```
=== Reading Alice's Preferences ===
Response: Your preferences are as follows:
- **Tone**: Casual
- **Topics of Interest**: AI, cooking, travel

=== Saving a New Preference ===
Response: Your favorite color has been saved as blue.

=== Verifying Persistence ===
Response: Your current preferences are as follows:
- **Tone**: Casual
- **Topics of Interest**: AI, cooking, travel
- **Favorite Color**: Blue

=== Different User (Bob) ===
Response: It looks like there are no preferences saved for you at this time. If you have
specific preferences you'd like to set, feel free to let me know!
```

---

## 11_retrieval_rag.py

```
=== Question 1: What are LangChain agents? ===
In LangChain, agents are built using the `create_agent()` function, which facilitates the
creation of a graph-based agent runtime. Here's a brief overview of how agents work:

1. **Agent Structure**: Agents can utilize various tools and maintain memory via
   checkpointers. This enables them to remember previous interactions and context.

2. **Agent Loop**: The operation of an agent consists of alternating between making calls
   to the language model and executing tools.

3. **Middleware**: LangChain supports middleware that can hook into the agent lifecycle at
   strategic points.

4. **Tool Integration**: Tools available to agents are created using the `@tool` decorator.

=== Question 2: What is middleware? ===
LangChain middleware is a feature that allows developers to hook into the agent lifecycle at
strategic points to customize or extend functionality.

- **Hooks**: `@before_model`, `@after_model`, `@dynamic_prompt`, and `@wrap_model_call`
- **Built-in Middleware**: `SummarizationMiddleware`, `HumanInTheLoopMiddleware`

=== Question 3: Simple greeting (no retrieval needed) ===
I can assist you with information and documentation related to LangChain features,
functionalities, and best practices. If you have specific questions or need help with
something related to LangChain, feel free to ask!
```

---

## 12_mcp.py

```
Loaded 3 tools from MCP server: ['add', 'multiply', 'factorial']

=== Math Query 1: Addition and Multiplication ===
The result of (3 + 5) * 12 is 96.

=== Math Query 2: Factorial ===
The factorial of 7 is 5040.
```

---

## 13_multi_agent_subagents.py

```
=== Multi-Agent: Research + Write ===
Final response:
Python's rising popularity in programming is driven by its ease of learning, versatile
applications across domains like web development and AI, and a strong community offering
extensive libraries and frameworks. Key libraries such as NumPy and TensorFlow, along with
frameworks like Django, enhance its functionality, allowing efficient application
development. Additionally, Python's cross-platform compatibility enables seamless operation
across different operating systems. The increasing demand for Python skills, especially in
data science and machine learning, further solidifies its status as a preferred programming
language.
```

---

## 14_multi_agent_handoffs.py

```
=== Step 1: Collect Information ===
Agent: I apologize for the inconvenience regarding your billing issue. To address the
double charge, could you please confirm if both charges were for the same service or
product? If they were, I can assist you in processing a refund for the duplicate charge.
Let me know!

=== Step 2: Resolution ===
Agent: I can help you initiate a refund for the duplicate charge. Please provide me with
the details of the charges, including the amounts and any relevant transaction IDs or dates
so I can proceed with the refund process.
```

---

## 15_context_engineering.py

```
=== Admin User ===
Response: The current system statistics are as follows:

- **CPU Usage**: 45%
- **Memory Usage**: 62%
- **Disk Usage**: 78%

Additionally, the alert threshold has been successfully updated to **90%**.

If you require further analysis or actions related to these stats, please let me know!

=== Viewer User ===
Response: The monthly revenue is $45,678, with a growth of +12%.
```

---

## 16_observability.py

```
LangSmith tracing enabled: false

=== Traced Invocation with Metadata ===
Response: The status of order ORD-002 is: Shipped, arriving on January 20.

=== Selective Tracing ===
Traced response: Order ORD-001: Wireless Mouse — Delivered on Jan 15.

Untraced response: Order ORD-003 for the Mechanical Keyboard is currently processing.
```

---

## Summary

| # | Example | Status | Notes |
|---|---------|--------|-------|
| 00 | hello_world | PASS | Simple agent responds correctly |
| 01 | chat_models | PASS | Both init_chat_model and ChatOpenAI work, multi-turn OK |
| 02 | tools | PASS | Weather and calculator tools called correctly |
| 03 | streaming | PASS | Updates mode shows step-by-step, messages mode streams tokens |
| 04 | structured_output | PASS | Pydantic models populated with correct types |
| 05 | short_term_memory | PASS | Thread isolation and multi-turn memory verified |
| 06 | runtime_context | PASS | Per-user context injected into tools correctly |
| 07 | middleware | PASS | before/after_model hooks fire, dynamic_prompt personalizes |
| 08 | guardrails | PASS | Banned keyword blocked, safe request passed through |
| 09 | human_in_the_loop | PASS | Interrupt on send_email, resumed with approval |
| 10 | long_term_memory | PASS | Store read/write works, persistence across calls verified |
| 11 | retrieval_rag | PASS | Knowledge base searched, answers cite retrieved content |
| 12 | mcp | PASS | FastMCP server spawned via stdio, 3 tools loaded and used |
| 13 | multi_agent_subagents | PASS | Research + writer subagents coordinated by supervisor |
| 14 | multi_agent_handoffs | PASS | Middleware switches prompt/tools based on state step |
| 15 | context_engineering | PASS | Admin gets all tools, viewer gets read-only; dynamic prompts |
| 16 | observability | PASS | Tracing context works (LangSmith auth skipped — no API key) |
