import asyncio
from workflows import Workflow, step
from workflows.events import Event, StartEvent, StopEvent


"""
-------------------------------------------------------
In this example, we explore LlamaIndex Workflows with the following features:
- OpenTelemetry integration for distributed tracing
- Custom spans and events for detailed observability
- Third-party observability tools (Arize Phoenix, Langfuse)
- Automatic instrumentation of workflow steps and LLM calls

Observability is crucial for debugging workflows. The framework provides
extensive instrumentation that tracks input/output of every step.

For more details, visit:
https://developers.llamaindex.ai/python/llamaagents/workflows/observability/
-------------------------------------------------------
"""


# Note: This example shows the API usage. To actually see traces,
# you need to set up an observability backend.


class ProcessingEvent(Event):
    data: str


class ObservabilityWorkflow(Workflow):
    @step
    async def step_one(self, ev: StartEvent) -> ProcessingEvent:
        """First step - automatically traced"""
        print("Step 1: Processing data...")
        await asyncio.sleep(0.5)
        return ProcessingEvent(data="Processed in step 1")

    @step
    async def step_two(self, ev: ProcessingEvent) -> StopEvent:
        """Second step - automatically traced"""
        print(f"Step 2: Received {ev.data}")
        await asyncio.sleep(0.3)
        return StopEvent(result="Workflow completed")


async def main():
    # All workflow steps are automatically instrumented
    workflow = ObservabilityWorkflow(timeout=30, verbose=True)
    result = await workflow.run()
    print(f"\nResult: {result}")
    
    print("\n" + "-" * 60)
    print("To enable OpenTelemetry tracing:")
    print("  pip install llama-index-observability-otel")
    print("\nFor third-party observability:")
    print("  • Arize Phoenix - Real-time tracing")
    print("  • Langfuse - Production monitoring")
    print("-" * 60)


if __name__ == "__main__":
    asyncio.run(main())
