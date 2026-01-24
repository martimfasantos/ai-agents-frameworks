import asyncio
from workflows import Workflow, step
from workflows.events import Event, StartEvent, StopEvent


"""
-------------------------------------------------------
In this example, we explore LlamaIndex Workflows with the following features:
- Drawing workflow diagrams using draw_all_possible_flows
- Visualizing workflow execution paths
- Exporting workflow structure to HTML

Note: This example demonstrates the drawing API. To actually visualize,
you need to install: pip install llama-index-utils-workflow

Workflow visualization helps understand complex execution paths and
debug workflow logic by showing all possible flows.

For more details, visit:
https://developers.llamaindex.ai/python/llamaagents/workflows/drawing/
-------------------------------------------------------
"""


# Define events for a multi-path workflow
class ProcessEvent(Event):
    data: str


class ValidationEvent(Event):
    is_valid: bool
    message: str


class DrawingExampleWorkflow(Workflow):
    @step
    async def start(self, ev: StartEvent) -> ProcessEvent:
        """Initial processing step"""
        return ProcessEvent(data="Initial data")

    @step
    async def process_data(self, ev: ProcessEvent) -> ValidationEvent:
        """Process and validate data"""
        # Simulate validation
        is_valid = len(ev.data) > 0
        return ValidationEvent(
            is_valid=is_valid,
            message="Data is valid" if is_valid else "Data is invalid"
        )

    @step
    async def handle_result(self, ev: ValidationEvent) -> StopEvent:
        """Handle validation result"""
        return StopEvent(result=ev.message)


async def main():
    workflow = DrawingExampleWorkflow(timeout=30, verbose=False)
    result = await workflow.run(data="test")
    print(f"Result: {result}\n")
    
    print("To visualize workflows:")
    print("  pip install llama-index-utils-workflow")
    print("\n  from llama_index.utils.workflow import draw_all_possible_flows")
    print("  draw_all_possible_flows(DrawingExampleWorkflow, filename='workflow.html')")
    
    # Uncomment to generate visualization:
    # from llama_index.utils.workflow import draw_all_possible_flows
    # draw_all_possible_flows(DrawingExampleWorkflow, filename="workflow.html")


if __name__ == "__main__":
    asyncio.run(main())
