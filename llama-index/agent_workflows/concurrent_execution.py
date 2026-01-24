import asyncio
import random
from workflows import Workflow, Context, step
from workflows.events import Event, StartEvent, StopEvent


"""
-------------------------------------------------------
In this example, we explore LlamaIndex Workflows with the following features:
- Concurrent execution of multiple workflow steps
- Using ctx.send_event() to emit multiple events
- Using ctx.collect_events() to wait for parallel operations
- Configuring num_workers for concurrent step execution

Workflows can run steps concurrently, enabling parallel execution of
time-consuming operations that await independently.

For more details, visit:
https://developers.llamaindex.ai/python/llamaagents/workflows/concurrent_execution/
-------------------------------------------------------
"""


# Define events for concurrent execution
class QueryEvent(Event):
    query: str


class ResultEvent(Event):
    result: str


class ConcurrentWorkflow(Workflow):
    @step
    async def start(self, ctx: Context, ev: StartEvent) -> QueryEvent | None:
        """Emit multiple events to trigger parallel processing"""
        ctx.send_event(QueryEvent(query="Query 1"))
        ctx.send_event(QueryEvent(query="Query 2"))
        ctx.send_event(QueryEvent(query="Query 3"))
        ctx.send_event(QueryEvent(query="Query 4"))
        return None

    @step(num_workers=4)
    async def process_query(self, ctx: Context, ev: QueryEvent) -> ResultEvent:
        """Process queries concurrently (up to 4 at a time)"""
        delay = random.randint(1, 3)
        await asyncio.sleep(delay)
        return ResultEvent(result=f"{ev.query} completed in {delay}s")

    @step
    async def collect_results(
        self, ctx: Context, ev: ResultEvent
    ) -> StopEvent | None:
        """Collect all results before proceeding"""
        results = ctx.collect_events(ev, [ResultEvent] * 4)
        
        if results is None:
            return None
        
        # All 4 events collected
        return StopEvent(result=f"Completed {len(results)} concurrent operations")


async def main():
    workflow = ConcurrentWorkflow(timeout=30, verbose=False)
    result = await workflow.run()
    print(f"Result: {result}")


if __name__ == "__main__":
    asyncio.run(main())
