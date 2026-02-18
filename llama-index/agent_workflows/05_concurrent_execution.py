import asyncio
import random
from workflows import Workflow, Context, step
from workflows.events import Event, StartEvent, StopEvent


"""
-------------------------------------------------------
In this example, we explore LlamaIndex Workflows with the following features:
- Emitting multiple events with ctx.send_event()
- Running steps concurrently with num_workers
- Collecting events with ctx.collect_events()
- Waiting for multiple different event types

Workflows can run steps concurrently when you have multiple independent
operations. Use ctx.send_event() to emit multiple events from a single
step, triggering parallel workers. The num_workers parameter on @step
controls how many instances run concurrently (default: 4). Use
ctx.collect_events() to wait until all parallel results arrive before
proceeding. You can also collect different event types together.

For more details, visit:
https://developers.llamaindex.ai/python/llamaagents/workflows/concurrent_execution/
-------------------------------------------------------
"""


# --- 1. Emitting multiple events with ctx.send_event() ---
class QueryEvent(Event):
    query: str


class QueryResultEvent(Event):
    result: str


class SimpleParallelWorkflow(Workflow):
    @step
    async def start(self, ctx: Context, ev: StartEvent) -> QueryEvent | None:
        """Emit multiple events to trigger parallel processing"""
        ctx.send_event(QueryEvent(query="Query 1"))
        ctx.send_event(QueryEvent(query="Query 2"))
        ctx.send_event(QueryEvent(query="Query 3"))
        return None

    @step(num_workers=4)  # Run up to 4 instances of this step concurrently
    async def process_query(self, ctx: Context, ev: QueryEvent) -> StopEvent:
        """Process queries concurrently (up to 4 at a time)"""
        delay = random.uniform(0.5, 2.0)
        await asyncio.sleep(delay)
        # Note: without collect_events, the first StopEvent ends the workflow
        return StopEvent(result=f"{ev.query} completed in {delay:.1f}s")


# --- 2. Collecting events before proceeding using ctx.collect_events() ---
class CollectedResultEvent(Event):
    result: str


class CollectingWorkflow(Workflow):
    @step
    async def start(self, ctx: Context, ev: StartEvent) -> QueryEvent | None:
        """Emit 3 events for parallel processing"""
        ctx.send_event(QueryEvent(query="Query A"))
        ctx.send_event(QueryEvent(query="Query B"))
        ctx.send_event(QueryEvent(query="Query C"))
        return None

    @step(num_workers=4)
    async def process(self, ctx: Context, ev: QueryEvent) -> CollectedResultEvent:
        """Process each query concurrently"""
        delay = random.uniform(0.5, 2.0)
        await asyncio.sleep(delay)
        return CollectedResultEvent(result=f"{ev.query} done in {delay:.1f}s")

    @step
    async def collect_all(
        self, ctx: Context, ev: CollectedResultEvent
    ) -> StopEvent | None:
        """Wait for all 3 results before continuing"""
        results = ctx.collect_events(ev, [CollectedResultEvent] * 3)
        if results is None:
            return None  # Not all events received yet

        # All 3 events collected — process together
        all_results = [r.result for r in results]
        return StopEvent(result=f"All done: {all_results}")


# --- 3. Collecting multiple different event types ---
class TaskAEvent(Event):
    query: str


class TaskBEvent(Event):
    query: str


class TaskADoneEvent(Event):
    result: str


class TaskBDoneEvent(Event):
    result: str


class MultiTypeWorkflow(Workflow):
    @step
    async def start(
        self, ctx: Context, ev: StartEvent
    ) -> TaskAEvent | TaskBEvent | None:
        """Emit different event types for different processing paths"""
        ctx.send_event(TaskAEvent(query="Task A"))
        ctx.send_event(TaskBEvent(query="Task B"))
        return None

    @step
    async def handle_a(self, ctx: Context, ev: TaskAEvent) -> TaskADoneEvent:
        await asyncio.sleep(0.5)
        return TaskADoneEvent(result=f"{ev.query} completed")

    @step
    async def handle_b(self, ctx: Context, ev: TaskBEvent) -> TaskBDoneEvent:
        await asyncio.sleep(0.8)
        return TaskBDoneEvent(result=f"{ev.query} completed")

    @step
    async def collect_all(
        self, ctx: Context, ev: TaskADoneEvent | TaskBDoneEvent
    ) -> StopEvent | None:
        """Wait for both different event types"""
        results = ctx.collect_events(ev, [TaskADoneEvent, TaskBDoneEvent])
        if results is None:
            return None
        return StopEvent(result=f"Both done: {[r.result for r in results]}")


# --- 4. Run all workflows ---
async def main():
    print("=== Simple Parallel (first-to-finish wins) ===")
    w1 = SimpleParallelWorkflow(timeout=30, verbose=False)
    result = await w1.run()
    print(f"Result: {result}\n")

    print("=== Collecting All Events ===")
    w2 = CollectingWorkflow(timeout=30, verbose=False)
    result = await w2.run()
    print(f"Result: {result}\n")

    print("=== Multiple Event Types ===")
    w3 = MultiTypeWorkflow(timeout=30, verbose=False)
    result = await w3.run()
    print(f"Result: {result}")


if __name__ == "__main__":
    asyncio.run(main())
