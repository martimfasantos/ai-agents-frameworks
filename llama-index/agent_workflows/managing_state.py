import asyncio
from pydantic import BaseModel, Field
from workflows import Workflow, Context, step
from workflows.events import StartEvent, StopEvent


"""
-------------------------------------------------------
In this example, we explore LlamaIndex Workflows with the following features:
- Managing state across workflow steps using Context
- Using typed state with Pydantic models for validation
- Locking state for atomic updates to prevent race conditions
- Maintaining context across multiple workflow runs

Workflows provide a Context object for sharing information between steps
with support for typed state and thread-safe operations.

For more details, visit:
https://developers.llamaindex.ai/python/llamaagents/workflows/managing_state/
-------------------------------------------------------
"""


# Define a Pydantic model for typed state
class CounterState(BaseModel):
    count: int = Field(default=0)
    total_runs: int = Field(default=0)
    messages: list[str] = Field(default_factory=list)


class StateManagementWorkflow(Workflow):
    @step
    async def increment_counter(
        self, ctx: Context[CounterState], ev: StartEvent
    ) -> StopEvent:
        """Increment the counter using atomic state updates"""
        async with ctx.store.edit_state() as ctx_state:
            ctx_state.count += 1
            ctx_state.total_runs += 1
            ctx_state.messages.append(f"Run #{ctx_state.total_runs}")
        
        current_state = await ctx.store.get_state()
        return StopEvent(
            result=f"Counter: {current_state.count}, Total runs: {current_state.total_runs}"
        )


async def main():
    workflow = StateManagementWorkflow(timeout=30, verbose=False)
    ctx = Context(workflow)
    
    # Run the workflow 3 times with the same context
    for i in range(3):
        result = await workflow.run(ctx=ctx)
        print(f"Run {i+1}: {result}")
    
    # Get final state
    final_state = await ctx.store.get_state()
    print(f"\nFinal state: {final_state.count} runs, {len(final_state.messages)} messages")


if __name__ == "__main__":
    asyncio.run(main())
