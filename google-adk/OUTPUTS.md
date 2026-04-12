# Google ADK Example Outputs

Captured outputs from running examples against Google ADK with `gemini-2.0-flash`.

> These outputs may vary between runs due to LLM non-determinism. The structure and tool invocations should remain consistent.

---

## 15_token_usage.py

> **Feature:** Token usage tracking via `Event.usage_metadata`. Per-event prompt, candidate, and total token counts with aggregated totals across the full agent conversation, including thoughts token tracking.

```
=== Google ADK Token Usage ===

--- Running agent ---

--- Per-Event Token Usage ---
  Event 1:
    Prompt tokens:     507
    Candidate tokens:  23
    Total tokens:      530
    Thoughts tokens:   0

--- Aggregated Token Usage ---
  Total prompt tokens:    507
  Total candidate tokens: 23
  Total tokens:           530
  Events with usage:      1

--- Agent Response ---
  The weather in London is cloudy at 14°C, while Tokyo is sunny at 28°C.

=== Token Usage Demo Complete ===
```

> **Note:** The Google ADK uses Gemini models which report `prompt_token_count`, `candidates_token_count`, and `thoughts_token_count` in `usage_metadata`. Actual token counts will vary per run.
