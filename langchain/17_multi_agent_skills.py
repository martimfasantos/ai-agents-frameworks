import os
import json
from pathlib import Path

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI

from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore LangChain with the following features:
- Multi-agent skills pattern with progressive disclosure
- On-demand skill loading via a tool
- Prompt-driven specialization without full sub-agents

In the skills architecture, specialized capabilities are packaged
as invocable "skills" that augment an agent's behavior. Skills are
primarily prompt-driven specializations that an agent can invoke
on-demand. This follows the progressive disclosure pattern — the
agent only loads detailed instructions when they become relevant,
keeping the context window lean.

For more details, visit:
https://docs.langchain.com/oss/python/langchain/multi-agent/skills
-------------------------------------------------------
"""

# --- 1. Create the model ---
model = ChatOpenAI(model=settings.OPENAI_MODEL_NAME)

# --- 2. Define skills as specialized prompts ---
SKILLS = {
    "write_sql": {
        "name": "SQL Query Writer",
        "prompt": (
            "You are a SQL expert. Write correct, efficient SQL queries. "
            "Rules:\n"
            "- Always use parameterized queries to prevent SQL injection\n"
            "- Use JOINs instead of subqueries when possible for performance\n"
            "- Add comments explaining complex logic\n"
            "- Use consistent formatting (uppercase keywords)\n\n"
            "Available tables:\n"
            "- users (id, name, email, created_at, is_active)\n"
            "- orders (id, user_id, total, status, created_at)\n"
            "- products (id, name, price, category, stock)\n"
            "- order_items (id, order_id, product_id, quantity, unit_price)"
        ),
        "description": "Expert at writing SQL queries with best practices",
    },
    "review_code": {
        "name": "Code Reviewer",
        "prompt": (
            "You are a senior code reviewer. Review code for:\n"
            "- Bugs and logic errors\n"
            "- Security vulnerabilities\n"
            "- Performance issues\n"
            "- Code style and readability\n"
            "- Missing error handling\n\n"
            "Format your review as:\n"
            "1. Summary (1-2 sentences)\n"
            "2. Issues found (numbered list)\n"
            "3. Suggestions (numbered list)"
        ),
        "description": "Reviews code for bugs, security, and best practices",
    },
    "write_tests": {
        "name": "Test Writer",
        "prompt": (
            "You are a testing expert. Write thorough test cases that cover:\n"
            "- Happy path scenarios\n"
            "- Edge cases and boundary values\n"
            "- Error handling paths\n"
            "- Input validation\n\n"
            "Use pytest-style test functions. Each test should have a clear "
            "docstring explaining what it tests. Use descriptive test names "
            "like test_<function>_<scenario>_<expected_result>."
        ),
        "description": "Writes comprehensive test cases with pytest",
    },
}


# --- 3. Create the skill loading tool ---
@tool
def load_skill(skill_name: str) -> str:
    """Load a specialized skill prompt to augment your capabilities.

    Available skills:
    - write_sql: SQL query writing expert (knows the database schema)
    - review_code: Senior code reviewer (checks bugs, security, style)
    - write_tests: Test writing expert (pytest, edge cases, coverage)

    Returns the skill's specialized prompt and context.
    """
    skill = SKILLS.get(skill_name)
    if not skill:
        available = ", ".join(SKILLS.keys())
        return f"Unknown skill '{skill_name}'. Available skills: {available}"

    return (
        f"=== Skill Loaded: {skill['name']} ===\n\n"
        f"{skill['prompt']}\n\n"
        f"You now have the {skill['name']} skill active. "
        f"Apply this expertise to the user's request."
    )


# --- 4. Create the agent with skill loading ---
agent = create_agent(
    model=model,
    tools=[load_skill],
    system_prompt=(
        "You are a helpful coding assistant with access to specialized skills. "
        "When a user asks for help with SQL, code review, or writing tests, "
        "FIRST load the appropriate skill using the load_skill tool, then "
        "apply that skill's expertise to answer the question. "
        "Be concise and practical in your responses."
    ),
)

# --- 5. Test SQL skill ---
print("=== SQL Skill: Write a query ===")
result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Write a SQL query to find the top 5 customers by total order value.",
            }
        ]
    }
)
print(f"Response:\n{result['messages'][-1].content}\n")

# --- 6. Test code review skill ---
print("=== Code Review Skill: Review a function ===")
result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Review this Python function:\n\n"
                    "def get_user(id):\n"
                    "    data = eval(open('users.json').read())\n"
                    "    for u in data:\n"
                    "        if u['id'] == id:\n"
                    "            return u\n"
                    "    return None"
                ),
            }
        ]
    }
)
print(f"Response:\n{result['messages'][-1].content}\n")

# --- 7. Test writing tests skill ---
print("=== Test Writing Skill: Write tests ===")
result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Write pytest tests for a function called `calculate_discount(price, percentage)` that returns the discounted price.",
            }
        ]
    }
)
print(f"Response:\n{result['messages'][-1].content}")
