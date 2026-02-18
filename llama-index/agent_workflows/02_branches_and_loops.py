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

# --- 1. Define the events for branching and looping ---
class LoopEvent(Event):
    num_loops: int

class BranchAEvent(Event):
    payload: str

class BranchBEvent(Event):
    payload: str


# --- 2. Define the workflow with branching and looping logic ---
class BranchingAndLoopingWorkflow(Workflow):
    @step
    async def start(self, ev: StartEvent) -> BranchAEvent | BranchBEvent:
        """Randomly select between two branches"""
        if random.randint(0, 1) == 0:
            print("Branch A selected")
            return BranchAEvent(payload="Branch A selected")
        else:
            print("Branch B selected")
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


# --- 3. Define the main function to run the workflow ---
async def main():
    workflow = BranchingAndLoopingWorkflow(timeout=30, verbose=False)
    result = await workflow.run()
    print(f"Result: {result}")
    print("--- Run the workflow multiple times to see different branches and loop counts. ---")


if __name__ == "__main__":
    asyncio.run(main())
