from smolagents import CodeAgent, ToolCallingAgent, OpenAIModel, tool

from settings import settings

"""
-------------------------------------------------------
In this example, we explore smolagents multi-agent orchestration:

- Creating specialized child agents with name and description
- A manager agent that delegates to child agents via managed_agents
- CodeAgent managing ToolCallingAgent children (mixed agent types)
- Automatic agent selection based on task requirements

Multi-agent systems let you decompose complex tasks into subtasks
handled by specialized agents. The manager agent decides which
child agent to invoke based on the task description.

For more details, visit:
https://huggingface.co/docs/smolagents/examples/multiagents
-------------------------------------------------------
"""

# --- 1. Create the model ---
model = OpenAIModel(
    model_id=settings.OPENAI_MODEL_NAME,
    api_key=settings.OPENAI_API_KEY.get_secret_value(),
)


# --- 2. Define specialized tools ---
@tool
def search_recipes(query: str) -> str:
    """Search for recipes matching a query.

    Args:
        query: The recipe search query.

    Returns:
        A string with matching recipes.
    """
    recipes = {
        "pasta": "Spaghetti Carbonara: eggs, pecorino, guanciale, black pepper",
        "salad": "Caesar Salad: romaine, parmesan, croutons, caesar dressing",
        "soup": "Tomato Basil Soup: tomatoes, basil, garlic, cream",
    }
    for key, recipe in recipes.items():
        if key in query.lower():
            return recipe
    return "Recipe not found. Try: pasta, salad, or soup."


@tool
def calculate_calories(ingredients: str) -> str:
    """Estimate total calories for a list of ingredients.

    Args:
        ingredients: Comma-separated list of ingredients.

    Returns:
        A string with the calorie estimate.
    """
    # Simplified calorie lookup
    cal_per_ingredient = {
        "eggs": 150,
        "pecorino": 110,
        "guanciale": 200,
        "black pepper": 5,
        "romaine": 15,
        "parmesan": 110,
        "croutons": 120,
        "caesar dressing": 160,
        "tomatoes": 30,
        "basil": 5,
        "garlic": 10,
        "cream": 200,
    }
    total = 0
    found = []
    for ingredient in ingredients.lower().split(","):
        ingredient = ingredient.strip()
        for key, cal in cal_per_ingredient.items():
            if key in ingredient:
                total += cal
                found.append(f"{key}: {cal} cal")
                break
    if found:
        return f"Total: ~{total} cal ({', '.join(found)})"
    return f"Could not estimate calories for: {ingredients}"


# --- 3. Create specialized child agents ---
recipe_agent = ToolCallingAgent(
    tools=[search_recipes],
    model=model,
    max_steps=3,
    name="recipe_finder",
    description="Finds recipes based on a food query. Use this agent when you need to look up a recipe.",
)

nutrition_agent = ToolCallingAgent(
    tools=[calculate_calories],
    model=model,
    max_steps=3,
    name="nutrition_calculator",
    description="Calculates calorie estimates for ingredients. Use this agent for nutrition questions.",
)

# --- 4. Create a manager agent that delegates to children ---
manager = CodeAgent(
    tools=[],
    model=model,
    managed_agents=[recipe_agent, nutrition_agent],
    max_steps=5,
)

# --- 5. Run a complex query that requires multiple agents ---
print("=== Multi-Agent Demo ===\n")

result = manager.run(
    "Find a pasta recipe and then estimate its total calories. "
    "Give a brief summary in 2-3 sentences."
)
print(f"Manager result: {result}")
