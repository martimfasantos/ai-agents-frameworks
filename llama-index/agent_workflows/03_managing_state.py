import asyncio
from pydantic import BaseModel, Field
from workflows import Workflow, Context, step
from workflows.events import StartEvent, StopEvent


"""
-------------------------------------------------------
In this example, we explore LlamaIndex Workflows with the following features:
- Sharing state between steps using the Context store (get/set)
- Locking state with edit_state() for atomic updates
- Adding typed state with Pydantic models for validation
- Maintaining context across multiple workflow runs

Workflows provide a Context object for sharing information between steps.
By default, an untyped state store is initialized and can be accessed via
ctx.store.get() and ctx.store.set(). For typed state, annotate your
Context with a Pydantic model (Context[MyState]) to get type hints,
automatic validation, and control over serialization. Use edit_state()
to lock the state and prevent race conditions during concurrent steps.

For more details, visit:
https://developers.llamaindex.ai/python/llamaagents/workflows/managing_state/
-------------------------------------------------------
"""


# --- 1. Untyped state: get/set on the Context store ---
class UntypedStateWorkflow(Workflow):
    @step
    async def increment(self, ctx: Context, ev: StartEvent) -> StopEvent:
        """Access and update state using ctx.store.get() and ctx.store.set()"""
        current_count = await ctx.store.get("count", default=0)
        current_count += 1
        await ctx.store.set("count", current_count)
        return StopEvent(result=f"Count is now {current_count}")


# --- 2. Locking state with edit_state() for atomic updates ---
class LockedStateWorkflow(Workflow):
    @step
    async def safe_increment(self, ctx: Context, ev: StartEvent) -> StopEvent:
        """Lock the state to prevent race conditions during concurrent steps"""
        # No other steps can access the state while the `with` block is running
        async with ctx.store.edit_state() as ctx_state:
            if "count" not in ctx_state:
                ctx_state["count"] = 0
            ctx_state["count"] += 1
        return StopEvent(result="Locked count updated")


# --- 3. Typed state with Pydantic models ---
class CounterState(BaseModel):
    count: int = Field(default=0)
    total_runs: int = Field(default=0)
    messages: list[str] = Field(default_factory=list)


class TypedStateWorkflow(Workflow):
    @step
    async def increment_counter(
        self, ctx: Context[CounterState], ev: StartEvent
    ) -> StopEvent:
        # NOTE: Context is typed with CounterState
        # Allows for atomic state updates with typed fields
        async with ctx.store.edit_state() as state:
            state.count += 1
            state.total_runs += 1
            state.messages.append(f"Run #{state.total_runs}")

        current_state = await ctx.store.get_state()
        return StopEvent(
            result=f"Counter: {current_state.count}, Total runs: {current_state.total_runs}"
        )


# --- 4. Maintaining context across multiple runs ---
async def main():
    print("=== Untyped State ===")
    untyped_wf = UntypedStateWorkflow(timeout=30, verbose=False)
    result = await untyped_wf.run()
    print(f"Result: {result}\n")

    print("=== Locked State ===")
    locked_wf = LockedStateWorkflow(timeout=30, verbose=False)
    result = await locked_wf.run()
    print(f"Result: {result}\n")

    print("=== Typed State with Context Persistence ===")
    typed_wf = TypedStateWorkflow(timeout=30, verbose=False)
    # Create a context to persist across runs
    ctx = Context(typed_wf)

    # Run the workflow 3 times reusing the same context
    for i in range(3):
        result = await typed_wf.run(ctx=ctx)
        print(f"Run {i + 1}: {result}")

    # Optional: serialize and restore context
    # ctx_dict = ctx.to_dict()
    # restored_ctx = Context.from_dict(typed_wf, ctx_dict)

    # Get final state
    final_state: CounterState = await ctx.store.get_state()
    print(f"\nFinal state: count={final_state.count}, messages={final_state.messages}")


if __name__ == "__main__":
    asyncio.run(main())
