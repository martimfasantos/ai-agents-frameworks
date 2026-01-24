import asyncio
import random
from workflows import Workflow, step
from workflows.events import Event, StartEvent, StopEvent


"""
-------------------------------------------------------
In this example, we explore LlamaIndex Workflows with the following features:
- Conditional branching to different execution paths
- Looping through events with dynamic iteration counts
- Random path selection in workflow execution

Workflows enable flexible control flow patterns through branching and looping
logic, providing more flexibility than traditional graph-based approaches.

For more details, visit:
https://developers.llamaindex.ai/python/llamaagents/workflows/branches_and_loops/
-------------------------------------------------------
"""


# Define events for looping
class LoopEvent(Event):
    num_loops: int


# Define events for branching
class BranchAEvent(Event):
    payload: str


class BranchBEvent(Event):
    payload: str


class BranchingAndLoopingWorkflow(Workflow):
    @step
    async def start(self, ev: StartEvent) -> BranchAEvent | BranchBEvent:
        """Randomly select between two branches"""
        if random.randint(0, 1) == 0:
            return BranchAEvent(payload="Branch A selected")
        else:
            return BranchBEvent(payload="Branch B selected")

    @step
    async def branch_a_step(self, ev: BranchAEvent) -> LoopEvent:
        """Process Branch A with looping"""
        num_loops = random.randint(2, 5)
        return LoopEvent(num_loops=num_loops)

    @step
    async def branch_b_step(self, ev: BranchBEvent) -> StopEvent:
        """Process Branch B (no looping)"""
        return StopEvent(result="Branch B completed")

    @step
    async def loop_step(self, ev: LoopEvent) -> LoopEvent | StopEvent:
        """Loop until counter reaches zero"""
        if ev.num_loops <= 0:
            return StopEvent(result="Looping completed")
        
        return LoopEvent(num_loops=ev.num_loops - 1)


async def main():
    workflow = BranchingAndLoopingWorkflow(timeout=30, verbose=False)
    result = await workflow.run()
    print(f"Result: {result}")


if __name__ == "__main__":
    asyncio.run(main())
