import asyncio
from workflows import Workflow, Context, step
from workflows.events import Event, StartEvent, StopEvent


"""
-------------------------------------------------------
In this example, we explore LlamaIndex Workflows with the following features:
- Drawing all possible flows with draw_all_possible_flows()
- Drawing the most recent execution with draw_most_recent_execution()
- Using the WorkflowServer debugger UI for visualization

Workflows can be visualized using type annotations in step definitions.
There are two main approaches: (1) converting a workflow to HTML using
the llama-index-utils-workflow package, which provides both static
(draw_all_possible_flows) and execution-based (draw_most_recent_execution)
views; (2) using the WorkflowServer debugger UI, which provides live
visualization, event logging, and human-in-the-loop support at the
server's root / path.

For more details, visit:
https://developers.llamaindex.ai/python/llamaagents/workflows/drawing/
-------------------------------------------------------
"""


# --- 1. Define events for a multi-path workflow ---
class ProcessEvent(Event):
    data: str


class ValidationEvent(Event):
    is_valid: bool
    message: str


# --- 2. Define a workflow with multiple paths ---
class DrawableWorkflow(Workflow):
    @step
    async def start(self, ctx: Context, ev: StartEvent) -> ProcessEvent:
        """Initial processing step"""
        return ProcessEvent(data="Sample data to process")

    @step
    async def process_data(self, ev: ProcessEvent) -> ValidationEvent:
        """Process and validate data"""
        is_valid = len(ev.data) > 0
        return ValidationEvent(
            is_valid=is_valid,
            message="Data is valid" if is_valid else "Data is invalid",
        )

    @step
    async def handle_result(self, ev: ValidationEvent) -> StopEvent:
        """Handle validation result"""
        return StopEvent(result=ev.message)


# --- 3. Draw all possible flows (static) ---
async def main():
    workflow = DrawableWorkflow(timeout=30, verbose=False)

    # Run the workflow
    handler = workflow.run()
    await handler

    # --- 3a. Draw all possible flows to HTML ---
    from llama_index.utils.workflow import draw_all_possible_flows
    draw_all_possible_flows(DrawableWorkflow, filename="all_paths.html")

    # --- 3b. Draw the most recent execution ---
    from llama_index.utils.workflow import draw_most_recent_execution
    draw_most_recent_execution(handler, filename="most_recent.html")

    # --- 4. Alternative: Use the WorkflowServer debugger UI ---
    # from workflows.server import WorkflowServer
    # server = WorkflowServer()
    # server.add_workflow("drawable", DrawableWorkflow())
    # await server.serve("0.0.0.0", "8080")
    # Then open http://localhost:8080 to see the debugger UI


if __name__ == "__main__":
    asyncio.run(main())
