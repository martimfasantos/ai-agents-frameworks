from strands import Agent, tool
from strands.models.openai import OpenAIModel

from settings import settings

"""
-------------------------------------------------------
In this example, we explore Strands Agents SDK with the following features:
- Class-based tools with the @tool decorator on methods
- Shared state between tool invocations
- Object-oriented tool patterns (TaskManager)

Class-based tools let you maintain state and shared resources across
multiple tool calls. This is useful for tools that need to track data,
manage connections, or accumulate results over the course of a conversation.

For more details, visit:
https://strandsagents.com/docs/user-guide/concepts/tools/custom-tools/#class-based-tools
-------------------------------------------------------
"""

# --- 1. Define a class with tool methods ---


class TaskManager:
    """A simple in-memory task manager demonstrating class-based tools."""

    def __init__(self):
        self.tasks: list[dict] = []
        self.next_id: int = 1

    @tool
    def add_task(self, title: str, priority: str = "medium") -> str:
        """Add a new task to the task list.

        Args:
            title: The title/description of the task
            priority: Priority level - low, medium, or high
        """
        task = {
            "id": self.next_id,
            "title": title,
            "priority": priority,
            "completed": False,
        }
        self.tasks.append(task)
        self.next_id += 1
        return f"Task #{task['id']} added: '{title}' (priority: {priority})"

    @tool
    def list_tasks(self) -> str:
        """List all tasks with their status."""
        if not self.tasks:
            return "No tasks found."
        lines = []
        for t in self.tasks:
            status = "done" if t["completed"] else "pending"
            lines.append(f"  #{t['id']} [{status}] ({t['priority']}) {t['title']}")
        return "Current tasks:\n" + "\n".join(lines)

    @tool
    def complete_task(self, task_id: int) -> str:
        """Mark a task as completed.

        Args:
            task_id: The ID of the task to complete
        """
        for t in self.tasks:
            if t["id"] == task_id:
                t["completed"] = True
                return f"Task #{task_id} '{t['title']}' marked as completed."
        return f"Task #{task_id} not found."


# --- 2. Create task manager instance ---
task_mgr = TaskManager()

# --- 3. Configure model and create agent with class-based tools ---
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
    system_prompt="You are a task management assistant. Use the available tools to manage tasks.",
    tools=[task_mgr.add_task, task_mgr.list_tasks, task_mgr.complete_task],
    callback_handler=None,
)

# --- 4. Run the agent ---
print("=== Class-Based Tools: Task Manager ===\n")
result = agent(
    "Please do the following:\n"
    "1. Add a high priority task 'Deploy to production'\n"
    "2. Add a medium priority task 'Write unit tests'\n"
    "3. Add a low priority task 'Update documentation'\n"
    "4. List all tasks\n"
    "5. Complete the 'Write unit tests' task\n"
    "6. List all tasks again to show the updated status"
)

# --- 5. Print results ---
print(f"Agent: {result.message}")

# --- 6. Verify internal state ---
print("\n--- Internal State ---")
for t in task_mgr.tasks:
    print(f"  Task #{t['id']}: {t['title']} - completed={t['completed']}")
