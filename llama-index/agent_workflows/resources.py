import asyncio
from typing import Annotated
from workflows import Workflow, step
from workflows.events import Event, StartEvent, StopEvent
from workflows.resource import Resource


"""
-------------------------------------------------------
In this example, we explore LlamaIndex Workflows with the following features:
- Injecting external dependencies as resources
- Sharing resources across multiple workflow steps
- Using Resource() wrapper with factory functions
- Controlling resource caching behavior

Resources allow you to inject dependencies like databases, LLMs, or
other services into workflow steps, with automatic management and sharing.

For more details, visit:
https://developers.llamaindex.ai/python/llamaagents/workflows/resources/
-------------------------------------------------------
"""


# Simple resource class
class DatabaseConnection:
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.query_count = 0
    
    async def query(self, sql: str) -> str:
        """Simulate a database query"""
        self.query_count += 1
        await asyncio.sleep(0.5)
        return f"Result for: {sql}"


def get_database():
    """Factory function that creates the database connection"""
    return DatabaseConnection("postgres://localhost:5432/mydb")


class QueryEvent(Event):
    query: str


class ResourceWorkflow(Workflow):
    @step
    async def first_step(
        self,
        ev: StartEvent,
        db: Annotated[DatabaseConnection, Resource(get_database)],
    ) -> QueryEvent:
        """First step that uses the database resource"""
        await db.query("SELECT * FROM users")
        return QueryEvent(query="SELECT * FROM orders")

    @step
    async def second_step(
        self,
        ev: QueryEvent,
        db: Annotated[DatabaseConnection, Resource(get_database)],
    ) -> StopEvent:
        """Second step that reuses the same database resource"""
        await db.query(ev.query)
        return StopEvent(
            result=f"Completed {db.query_count} queries (same DB instance)"
        )


async def main():
    workflow = ResourceWorkflow(timeout=30, verbose=False)
    result = await workflow.run()
    print(f"Result: {result}")


if __name__ == "__main__":
    asyncio.run(main())
