from strands import Agent, AgentSkills, Skill
from strands.models.openai import OpenAIModel

from settings import settings

"""
-------------------------------------------------------
In this example, we explore Strands Agents SDK with the following features:
- AgentSkills plugin for modular, on-demand instructions
- Programmatic skill creation with the Skill dataclass
- Skill discovery and activation at runtime
- Managing skills dynamically (add, replace, inspect)

Skills give your agent on-demand access to specialized instructions without
bloating the system prompt. Lightweight metadata (name + description) is
injected into the system prompt, and full instructions load only when the
agent activates a skill through a tool call.

For more details, visit:
https://strandsagents.com/docs/user-guide/concepts/plugins/skills/
-------------------------------------------------------
"""

# --- 1. Create skills programmatically ---
code_review_skill = Skill(
    name="code-review",
    description="Review code for best practices, bugs, and performance issues",
    instructions=(
        "You are a senior code reviewer. When reviewing code:\n"
        "1. Check for potential bugs and edge cases\n"
        "2. Evaluate code readability and naming conventions\n"
        "3. Look for performance bottlenecks\n"
        "4. Suggest specific improvements with code examples\n"
        "5. Rate the code quality on a scale of 1-10"
    ),
)

data_analysis_skill = Skill(
    name="data-analysis",
    description="Analyze datasets, identify trends, and create summaries",
    instructions=(
        "You are a data analysis expert. When analyzing data:\n"
        "1. Identify key patterns and trends\n"
        "2. Calculate relevant statistics\n"
        "3. Note any outliers or anomalies\n"
        "4. Provide actionable insights\n"
        "5. Summarize findings in a clear, structured format"
    ),
)

writing_skill = Skill(
    name="technical-writing",
    description="Write clear technical documentation and reports",
    instructions=(
        "You are a technical writer. When creating documentation:\n"
        "1. Use clear, concise language\n"
        "2. Structure content with headings and sections\n"
        "3. Include code examples where relevant\n"
        "4. Define technical terms on first use\n"
        "5. Follow a logical flow from overview to details"
    ),
)

# --- 2. Create the AgentSkills plugin ---
skills_plugin = AgentSkills(
    skills=[code_review_skill, data_analysis_skill, writing_skill]
)

# --- 3. Configure model and create agent with skills ---
openai_model = OpenAIModel(
    client_args={
        "api_key": settings.OPENAI_API_KEY.get_secret_value()
        if settings.OPENAI_API_KEY
        else ""
    },
    model_id=settings.OPENAI_MODEL_NAME,
)
# Default: Agent() uses Amazon Bedrock (requires AWS credentials)
agent = Agent(
    model=openai_model,
    system_prompt="You are a versatile assistant. Use your available skills when appropriate.",
    plugins=[skills_plugin],
    callback_handler=None,
)

# --- 4. List available skills ---
print("=== Skills Plugin ===\n")
print("Available skills:")
for skill in skills_plugin.get_available_skills():
    print(f"  - {skill.name}: {skill.description}")

# --- 5. Run the agent (it will activate skills as needed) ---
print("\n--- Agent with Skills ---\n")
result = agent(
    "Review this Python function for issues:\n\n"
    "def process(data):\n"
    "    result = []\n"
    "    for i in range(len(data)):\n"
    "        if data[i] != None:\n"
    "            result.append(data[i] * 2)\n"
    "    return result"
)
print(f"Agent response: {result.message}")

# --- 6. Manage skills at runtime ---
print("\n--- Runtime Skill Management ---\n")
new_skill = Skill(
    name="security-audit",
    description="Audit code for security vulnerabilities",
    instructions="You are a security expert. Check for injection, XSS, CSRF, and auth issues.",
)
skills_plugin.set_available_skills(skills_plugin.get_available_skills() + [new_skill])

print("Updated skills:")
for skill in skills_plugin.get_available_skills():
    print(f"  - {skill.name}: {skill.description}")

# --- 7. Check activated skills ---
activated = skills_plugin.get_activated_skills(agent)
print(f"\nActivated skills: {activated}")
