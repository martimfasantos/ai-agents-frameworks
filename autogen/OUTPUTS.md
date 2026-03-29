# AutoGen - Example Outputs

All examples run with `autogen-agentchat==0.7.5`, `autogen-core==0.7.5`, `autogen-ext==0.7.5`, and `OpenAIChatCompletionClient` (`gpt-4o-mini`) as the model provider.

> **Note:** LLM responses are non-deterministic. Your outputs will differ in wording but should follow the same structure and demonstrate the same features.

---

## 00_hello_world.py

```
$ uv run python 00_hello_world.py

---------- TextMessage (user) ----------
Say 'Hello World!'
---------- TextMessage (assistant) ----------
Hello World!
```

**Verdict:** PASS - Agent creates, runs, and streams a simple response via Console.

---

## 01_tools.py

```
$ uv run python 01_tools.py

---------- TextMessage (user) ----------
Find information on AutoGen
---------- ToolCallRequestEvent (assistant) ----------
[FunctionCall(id='call_FhuJgVuRqYMwGmUsYHspTvvK', arguments='{"query":"AutoGen"}', name='web_search_func')]
[Prompt tokens: 62, Completion tokens: 16]
---------- ToolCallExecutionEvent (assistant) ----------
[FunctionExecutionResult(content='AutoGen is a programming framework for building multi-agent applications.', name='web_search_func', call_id='call_FhuJgVuRqYMwGmUsYHspTvvK', is_error=False)]
---------- ToolCallSummaryMessage (assistant) ----------
AutoGen is a programming framework for building multi-agent applications.
---------- Summary ----------
Number of messages: 4
Finish reason: None
Total prompt tokens: 62
Total completion tokens: 16
Duration: 1.05 seconds
--------------------------------------------------
Final Answer: AutoGen is a programming framework for building multi-agent applications.
--------------------------------------------------
Tool Schema: {'name': 'web_search_func', 'description': 'Find information on the web', 'parameters': {'type': 'object', 'properties': {'query': {'description': 'query', 'title': 'Query', 'type': 'string'}}, 'required': ['query'], 'additionalProperties': False}, 'strict': False}
```

**Verdict:** PASS - Agent calls the web_search_func tool, returns the result, and prints the auto-generated tool schema.

---

## 02_streaming_and_metrics.py

```
$ uv run python 02_streaming_and_metrics.py

---------- TextMessage (user) ----------
What is the result of 2 + 4?
---------- ToolCallRequestEvent (assistant) ----------
[FunctionCall(id='call_ZJmtNjlUG7ZWlQbQJkg0ESD3', arguments='{"values":[2,4]}', name='add_numbers')]
[Prompt tokens: 70, Completion tokens: 16]
---------- ToolCallExecutionEvent (assistant) ----------
[FunctionExecutionResult(content='6', name='add_numbers', call_id='call_ZJmtNjlUG7ZWlQbQJkg0ESD3', is_error=False)]
---------- ToolCallSummaryMessage (assistant) ----------
6
---------- Summary ----------
Number of messages: 4
Finish reason: None
Total prompt tokens: 70
Total completion tokens: 16
Duration: 1.03 seconds
```

**Verdict:** PASS - Agent calls add_numbers tool (previously named `sum`, renamed to avoid shadowing builtin), streams result with token usage metrics.

---

## 03_structured_outputs.py

```
$ uv run python 03_structured_outputs.py

---------- TextMessage (user) ----------
I am happy.
---------- StructuredMessage[AgentResponse] (assistant) ----------
{"thoughts":"The user expresses a clear sense of happiness, indicating positive feelings or contentment.","response":"happy"}
Thought: The user expresses a clear sense of happiness, indicating positive feelings or contentment.
Response: happy
```

**Verdict:** PASS - Agent returns a StructuredMessage with chain-of-thought `thoughts` field and validated `response` literal ("happy").

---

## 04_human_in_the_loop.py

```
$ uv run python 04_human_in_the_loop.py

==================================================
FIRST RUN: Agent will hand off to user
==================================================
---------- TextMessage (user) ----------
What is the weather in New York?
---------- ToolCallRequestEvent (lazy_assistant) ----------
[FunctionCall(id='call_ZsZG9VcRJOyepKky3kV8LwZY', arguments='{}', name='transfer_to_user')]
---------- ToolCallExecutionEvent (lazy_assistant) ----------
[FunctionExecutionResult(content='Transfer to user.', name='transfer_to_user', call_id='call_ZsZG9VcRJOyepKky3kV8LwZY', is_error=False)]
---------- HandoffMessage (lazy_assistant) ----------
Transfer to user.

==================================================
SECOND RUN: User provides the answer
==================================================
---------- TextMessage (user) ----------
The weather in New York is sunny and 72 degrees.
---------- TextMessage (lazy_assistant) ----------
Thank you for the update on the weather in New York! If you have any other questions or need further assistance, feel free to ask.
---------- TextMessage (lazy_assistant) ----------
TERMINATE
```

**Verdict:** PASS - Agent hands off to user via HandoffTermination, team pauses, then resumes with user-provided information. No `input()` calls.

---

## 05_multi_agent_teams.py

```
$ uv run python 05_multi_agent_teams.py

-----------------------------------------------------------------
               1. RoundRobinGroupChat
-----------------------------------------------------------------

---------- TextMessage (user) ----------
Write a short poem about the fall season.
---------- TextMessage (primary) ----------
Golden leaves in crisp, cool air,
Whispers of autumn everywhere.
...
---------- TextMessage (critic) ----------
This poem beautifully captures the essence of fall...
When you've made adjustments based on this feedback, please respond with 'APPROVE.'

-----------------------------------------------------------------
               2. SelectorGroupChat
-----------------------------------------------------------------

---------- TextMessage (user) ----------
Book a 3-day trip from Lisbon to NY with flight and hotel.
---------- ToolCallRequestEvent (flight_agent) ----------
[FunctionCall(... name='lookup_flight')]
---------- ToolCallSummaryMessage (flight_agent) ----------
Here are some flights from Lisbon to New York: flight1, flight2, flight3.
Here are some flights from New York to Lisbon: flight1, flight2, flight3.
---------- ToolCallRequestEvent (hotel_agent) ----------
[FunctionCall(... name='lookup_hotel')]
---------- ToolCallSummaryMessage (hotel_agent) ----------
Here are some hotels in New York: hotel1, hotel2, hotel3.
---------- ToolCallRequestEvent (travel_advisor) ----------
[FunctionCall(... name='book_trip')]
---------- ToolCallSummaryMessage (travel_advisor) ----------
Your trip is booked!
---------- TextMessage (flight_agent) ----------
Your 3-day trip from Lisbon to New York has been successfully booked...
TERMINATE

-----------------------------------------------------------------
               3. MagenticOneGroupChat
-----------------------------------------------------------------

---------- TextMessage (user) ----------
What is the capital of France?
---------- TextMessage (MagenticOneOrchestrator) ----------
We are working to address the following user request...
Here is the plan to follow as best as possible...
---------- TextMessage (MagenticOneOrchestrator) ----------
The capital of France is Paris.

-----------------------------------------------------------------
               4. Swarm
-----------------------------------------------------------------

---------- TextMessage (user) ----------
What is Bob's birthday?
---------- ToolCallRequestEvent (Alice) ----------
[FunctionCall(... name='transfer_to_bob')]
---------- HandoffMessage (Alice) ----------
Transferred to Bob, adopting the role of Bob immediately.
---------- TextMessage (Bob) ----------
My birthday is on January 1st.
```

**Verdict:** PASS - All 4 team presets demonstrated: RoundRobin (writer/critic), Selector (travel agents with tool routing), MagenticOne (orchestrator planning), Swarm (Alice hands off to Bob).

---

## 06_agents_as_tool.py

```
$ uv run python 06_agents_as_tool.py

-----------------------------------------------------------------
               1. Agent as Tool
-----------------------------------------------------------------

---------- TextMessage (user) ----------
Write a poem about the sea.
---------- ToolCallRequestEvent (assistant) ----------
[FunctionCall(... name='writer')]
---------- TextMessage (writer) ----------
In the hush of dawn, where horizons meet,
The sea awakens, with secrets discreet...
---------- ToolCallSummaryMessage (assistant) ----------
writer: In the hush of dawn, where horizons meet...

-----------------------------------------------------------------
               2. Team of Agents as Tool
-----------------------------------------------------------------

---------- TextMessage (user) ----------
Write a poem about the sea.
---------- TextMessage (assistant) ----------
The sea is a vast and endless blue...
```

**Verdict:** PASS - Both AgentTool (single agent as tool) and TeamTool (writer+summarizer team as tool) demonstrated with parallel_tool_calls disabled.

---

## 07_memory.py

```
$ uv run python 07_memory.py

==================================================
Weather Query (expects metric units from memory):
==================================================
---------- TextMessage (user) ----------
What's the weather like in New York?
---------- MemoryQueryEvent (assistant_agent) ----------
[MemoryContent(content='The weather should be in metric units', ...), MemoryContent(content='Meal recipes must be vegan', ...)]
---------- ToolCallRequestEvent (assistant_agent) ----------
[FunctionCall(... arguments='{"city":"New York","units":"metric"}', name='get_weather')]
---------- ToolCallSummaryMessage (assistant_agent) ----------
The weather in New York is 23 °C and Sunny.

==================================================
Recipe Query (expects vegan from memory):
==================================================
---------- TextMessage (user) ----------
Write a brief meal recipe with broth.
---------- MemoryQueryEvent (assistant_agent) ----------
[MemoryContent(content='The weather should be in metric units', ...), MemoryContent(content='Meal recipes must be vegan', ...)]
---------- TextMessage (assistant_agent) ----------
Here's a simple vegan broth recipe:
### Vegan Vegetable Broth
...
```

**Verdict:** PASS - ListMemory stores user preferences. Weather query uses metric units (memory preference), recipe query produces vegan output (memory preference). MemoryQueryEvent shows memory being consulted.

---

## 08_graph_flow.py

```
$ uv run python 08_graph_flow.py

-----------------------------------------------------------------
               1. Sequential Workflow
-----------------------------------------------------------------

---------- TextMessage (user) ----------
Write a short story about a robot.
---------- TextMessage (writer) ----------
In a forgotten corner of a post-apocalyptic world, a small robot named Tink tirelessly sorted through rubble...
---------- TextMessage (reviewer) ----------
This short story presents a compelling scenario... APPROVE

-----------------------------------------------------------------
               2. Parallel Fan-Out Workflow
-----------------------------------------------------------------

---------- TextMessage (user) ----------
Plan a 1-day trip to Tokyo.
---------- TextMessage (planner) ----------
Sure! Here's a suggested itinerary...
---------- TextMessage (food_expert) ----------
You must try ramen, specifically a bowl of Tonkotsu ramen...
---------- TextMessage (culture_expert) ----------
For a cultural highlight, don't miss the stunning Senso-ji Temple...
---------- TextMessage (summarizer) ----------
### 1-Day Trip to Tokyo Summary
Start your day by visiting the Senso-ji Temple... DONE

-----------------------------------------------------------------
               3. Conditional Routing (Positive)
-----------------------------------------------------------------

---------- TextMessage (user) ----------
I just got promoted at work!
---------- TextMessage (classifier) ----------
POSITIVE
---------- TextMessage (positive_responder) ----------
Congratulations! That's such an exciting achievement... DONE

-----------------------------------------------------------------
               3. Conditional Routing (Negative)
-----------------------------------------------------------------

---------- TextMessage (user) ----------
I lost my wallet today.
---------- TextMessage (classifier) ----------
NEGATIVE
---------- TextMessage (negative_responder) ----------
I'm really sorry to hear that you lost your wallet... DONE.
```

**Verdict:** PASS - All three GraphFlow patterns demonstrated: sequential (writer→reviewer), parallel fan-out (planner→food_expert+culture_expert→summarizer), and conditional routing (classifier→positive/negative based on message content).

---

## 09_termination_conditions.py

```
$ uv run python 09_termination_conditions.py

-----------------------------------------------------------------
               1. MaxMessageTermination (3 msgs)
-----------------------------------------------------------------

---------- TextMessage (user) ----------
Count from 1 to 10.
---------- TextMessage (assistant) ----------
Sure! Here you go: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10.
---------- TextMessage (assistant) ----------
Sure! Here you go: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10.
Stop reason: Maximum number of messages 3 reached, current message count: 3

-----------------------------------------------------------------
               2. TextMentionTermination ('DONE')
-----------------------------------------------------------------

---------- TextMessage (user) ----------
Say hello and then say DONE.
Stop reason: Text 'DONE' mentioned

-----------------------------------------------------------------
               3. TimeoutTermination (5 seconds)
-----------------------------------------------------------------

---------- TextMessage (user) ----------
Tell me a very short joke.
---------- TextMessage (assistant) ----------
Why did the scarecrow win an award? Because he was outstanding in his field!
...
Stop reason: Timeout of 5 seconds reached

-----------------------------------------------------------------
               4. Combined OR: TextMention | MaxMessage
-----------------------------------------------------------------

---------- TextMessage (user) ----------
Write one sentence about the moon, then say FINISHED.
Stop reason: Text 'FINISHED' mentioned

-----------------------------------------------------------------
               5. Combined AND: MaxMessage & TextMention
-----------------------------------------------------------------

---------- TextMessage (user) ----------
Say one word per message. After two messages say COMPLETE.
---------- TextMessage (assistant) ----------
Hello.
---------- TextMessage (assistant) ----------
World.
Stop reason: Text 'COMPLETE' mentioned, Maximum number of messages 3 reached, current message count: 3
```

**Verdict:** PASS - All termination conditions demonstrated: MaxMessage (stops at 3), TextMention (stops on keyword), Timeout (stops after 5 seconds), OR combinator (first condition wins), AND combinator (both conditions must be met).

---

## 10_state_management.py

```
$ uv run python 10_state_management.py

-----------------------------------------------------------------
               1. Agent State Management
-----------------------------------------------------------------

---------- TextMessage (user) ----------
What is the capital of France?
---------- TextMessage (assistant) ----------
The capital of France is Paris.

--- Agent state saved ---
State keys: ['type', 'version', 'llm_context']
--- Agent state loaded into new agent ---
---------- TextMessage (user) ----------
What was my previous question?
---------- TextMessage (assistant) ----------
Your previous question was, "What is the capital of France?"

-----------------------------------------------------------------
               2. Team State Management
-----------------------------------------------------------------

--- First team run ---
---------- TextMessage (user) ----------
Start a story about a dragon.
---------- TextMessage (writer) ----------
Once, high above the silver-tipped mountains, there lived a solitary dragon named Zephyra...
---------- TextMessage (critic) ----------
The opening line beautifully sets the scene... APPROVE.
---------- TextMessage (writer) ----------
...Zephyra felt an unfamiliar stir of longing...

--- Team state saved ---
State JSON preview: {"type": "TeamState", "version": "1.0.0", "agent_states": ...
--- Team reset ---
--- Team state restored ---

--- Resumed team run ---
---------- TextMessage (critic) ----------
This revision beautifully enhances the imagery... APPROVE.
---------- TextMessage (writer) ----------
...a peculiar sound echoed through the valley...
```

**Verdict:** PASS - Agent state saved/loaded (new agent remembers previous question). Team state saved, reset, restored, and resumed (story continues from checkpoint).

---

## 11_custom_agents.py

```
$ uv run python 11_custom_agents.py

-----------------------------------------------------------------
               1. CountDownAgent
-----------------------------------------------------------------

4...
3...
2...
1...
Liftoff! 🚀

-----------------------------------------------------------------
               2. EchoAgent
-----------------------------------------------------------------

Input:  Hello, World!
Output: Echo (reversed): !dlroW ,olleH

Input:  Autogen is great
Output: Echo (reversed): taerg si negotuA
```

**Verdict:** PASS - Both custom agents work correctly: CountDownAgent counts from 4 to liftoff, EchoAgent reverses input text. No LLM calls needed — pure custom logic via BaseChatAgent subclass.
