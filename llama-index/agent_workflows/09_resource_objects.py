import asyncio
from typing import Annotated
from llama_index.llms.openai import OpenAI
from workflows import Workflow, step
from workflows.events import Event, StartEvent, StopEvent
from workflows.resource import Resource, ResourceConfig


"""
-------------------------------------------------------
In this example, we explore LlamaIndex Workflows with the following features:
- Injecting external dependencies as resources with Resource()
- Sharing resources across multiple workflow steps
- Controlling resource caching (cache=True/False)
- Config-backed resources with ResourceConfig for JSON files
- Chaining resources with dependency resolution

Resources are external dependencies injected into workflow steps using
Annotated[Type, Resource(factory_fn)]. The factory function is called
once per workflow run (cached), and the same instance is shared across
steps. Pass cache=False to create a new instance per step. For config
data stored in JSON, use ResourceConfig to load and validate into a
Pydantic model. Resources can be chained: a Resource factory can depend
on a ResourceConfig, and the dependency chain is resolved automatically.

For more details, visit:
https://developers.llamaindex.ai/python/llamaagents/workflows/resources/
-------------------------------------------------------
"""


# --- 1. Define a resource class ---
class DatabaseConnection:
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.query_count = 0

    async def query(self, sql: str) -> str:
        """Simulate a database query"""
        self.query_count += 1
        await asyncio.sleep(0.1)
        return f"Result for: {sql}"


def get_database():
    """Factory function that creates the database resource"""
    return DatabaseConnection("postgres://localhost:5432/mydb")


class QueryEvent(Event):
    query: str


# --- 2. Inject and share resources across steps ---
class ResourceWorkflow(Workflow):
    @step
    async def first_step(
        self,
        ev: StartEvent,
        db: Annotated[DatabaseConnection, Resource(get_database)],
    ) -> QueryEvent:
        """First step uses the database resource"""
        result = await db.query("SELECT * FROM users")
        print(f"Step 1: {result} (queries so far: {db.query_count})")
        return QueryEvent(query="SELECT * FROM orders")

    @step
    async def second_step(
        self,
        ev: QueryEvent,
        db: Annotated[DatabaseConnection, Resource(get_database)],
    ) -> StopEvent:
        """Second step reuses the SAME database instance (cached by default)"""
        result = await db.query(ev.query)
        print(f"Step 2: {result} (queries so far: {db.query_count})")
        return StopEvent(
            result=f"Completed {db.query_count} queries on same DB instance"
        )


# --- 3. Disable caching to get a fresh resource per step ---
class UncachedResourceWorkflow(Workflow):
    @step
    async def step_one(
        self,
        ev: StartEvent,
        db: Annotated[DatabaseConnection, Resource(get_database, cache=False)],
    ) -> QueryEvent:
        """Each step gets its own fresh instance when cache=False"""
        await db.query("SELECT 1")
        print(f"Step 1 query count: {db.query_count}")
        return QueryEvent(query="SELECT 2")

    @step
    async def step_two(
        self,
        ev: QueryEvent,
        db: Annotated[DatabaseConnection, Resource(get_database, cache=False)],
    ) -> StopEvent:
        """This is a DIFFERENT instance — query_count starts at 0"""
        await db.query(ev.query)
        print(f"Step 2 query count: {db.query_count}")
        return StopEvent(result="Each step had its own DB instance")


# --- 4. Config-backed resources (ResourceConfig) ---

from pydantic import BaseModel
from workflows.resource import ResourceConfig

class ClassifierConfig(BaseModel):
    categories: list[str]
    threshold: float

class ConfigWorkflow(Workflow):
    @step
    async def classify(
        self,
        ev: StartEvent,
        config: Annotated[
            ClassifierConfig,
            ResourceConfig(config_file="classifier.json"),
        ],
    ) -> StopEvent:
        return StopEvent(result=f"Threshold: {config.threshold}")


# --- 5. Chaining resources ---
class LLMConfig(BaseModel):
    model: str
    temperature: float
    max_tokens: int

def get_llm(
    config: Annotated[LLMConfig, ResourceConfig(config_file="llm.json")],
) -> OpenAI:
    return OpenAI(model=config.model, temperature=config.temperature)

class MyWorkflow(Workflow):
    @step
    async def generate(
        self,
        ev: StartEvent,
        llm: Annotated[OpenAI, Resource(get_llm)],
    ) -> StopEvent:
        response = await llm.acomplete(ev.input)
        return StopEvent(result=response.text)


# --- 6. Run the examples ---
async def main():
    print("=== Shared Resource (cached) ===")
    w1 = ResourceWorkflow(timeout=30, verbose=False)
    result = await w1.run()
    print(f"Result: {result}\n")

    print("=== Fresh Resource per Step (cache=False) ===")
    w2 = UncachedResourceWorkflow(timeout=30, verbose=False)
    result = await w2.run()
    print(f"Result: {result}")


if __name__ == "__main__":
    asyncio.run(main())
