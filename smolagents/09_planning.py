from smolagents import CodeAgent, OpenAIModel, DuckDuckGoSearchTool

from settings import settings

"""
-------------------------------------------------------
In this example, we explore smolagents planning capabilities:

- Enabling planning with planning_interval parameter
- Agent creating explicit plans before acting
- Periodic re-planning during multi-step execution
- Comparing planned vs unplanned agent behavior

The planning_interval parameter makes the agent periodically
stop to create or revise an explicit plan. This improves
performance on complex tasks by encouraging structured
reasoning before action.

For more details, visit:
https://huggingface.co/docs/smolagents/tutorials/building_good_agents
-------------------------------------------------------
"""

# --- 1. Create the model ---
model = OpenAIModel(
    model_id=settings.OPENAI_MODEL_NAME,
    api_key=settings.OPENAI_API_KEY.get_secret_value(),
)

# --- 2. Create agent WITH planning ---
print("=== Planning Demo ===\n")

agent_with_planning = CodeAgent(
    tools=[DuckDuckGoSearchTool()],
    model=model,
    max_steps=5,
    planning_interval=2,  # Re-plan every 2 steps
)

print("--- Agent with planning (planning_interval=2) ---")
result = agent_with_planning.run(
    "Search for the population of Tokyo and the population of Delhi. "
    "Which city has more people? Reply in 2-3 sentences."
)
print(f"\nResult: {result}\n")

# --- 3. Show planning steps from memory ---
print("--- Steps taken by the planning agent ---")
for i, step in enumerate(agent_with_planning.memory.steps):
    step_type = type(step).__name__
    if step_type == "PlanningStep":
        plan_preview = str(getattr(step, "plan", ""))[:200]
        print(f"  Step {i}: PLANNING - {plan_preview}")
    elif step_type == "ActionStep":
        if hasattr(step, "tool_calls") and step.tool_calls:
            tools_used = [tc.name for tc in step.tool_calls]
            print(f"  Step {i}: ACTION - tools: {tools_used}")
        else:
            print(f"  Step {i}: ACTION (code execution)")
    else:
        print(f"  Step {i}: {step_type}")
