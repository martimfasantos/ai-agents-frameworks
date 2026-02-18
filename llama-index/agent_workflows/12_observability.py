import asyncio
from workflows import Workflow, step
from workflows.events import Event, StartEvent, StopEvent


"""
-------------------------------------------------------
In this example, we explore LlamaIndex Workflows with the following features:
- OpenTelemetry integration for distributed tracing
- Custom spans and events using the LlamaIndex dispatcher
- Third-party observability tools (Arize Phoenix, Langfuse, Opik)
- Automatic instrumentation of workflow steps and LLM calls

Observability is key for debugging workflows. Beyond print() statements,
workflows ship with extensive instrumentation that tracks the input and
output of every step as OpenTelemetry spans. Use the
llama-index-observability-otel package to export traces. You can also
define custom spans and events using the LlamaIndex dispatcher
(@dispatcher.span and dispatcher.event()). Third-party tools like
Arize Phoenix, Langfuse, and Opik integrate seamlessly for production
monitoring and visualization.

For more details, visit:
https://developers.llamaindex.ai/python/llamaagents/workflows/observability/
-------------------------------------------------------
"""


# --- 1. Define a simple workflow (all steps are auto-instrumented) ---
class ProcessingEvent(Event):
    data: str


class ObservableWorkflow(Workflow):
    @step
    async def step_one(self, ev: StartEvent) -> ProcessingEvent:
        """First step — automatically traced via OpenTelemetry"""
        await asyncio.sleep(0.5)
        return ProcessingEvent(data="Processed in step 1")

    @step
    async def step_two(self, ev: ProcessingEvent) -> StopEvent:
        """Second step — automatically traced via OpenTelemetry"""
        await asyncio.sleep(0.3)
        return StopEvent(result=f"Completed. Received: {ev.data}")


# --- 2. OpenTelemetry integration ---
# pip install llama-index-observability-otel
#
# from llama_index.observability.otel import LlamaIndexOpenTelemetry
#
# instrumentor = LlamaIndexOpenTelemetry(
#     span_exporter=your_span_exporter,
#     service_name_or_resource="your_service_name",
# )
# instrumentor.start_registering()
#
# All workflow steps, LLM calls, and events are automatically captured as
# OpenTelemetry spans with attributes: span names, start/end times,
# input/output data, and nested span relationships.


# --- 3. Custom spans and events with the dispatcher ---
# from llama_index_instrumentation import get_dispatcher
# from llama_index_instrumentation.base import BaseEvent
#
# dispatcher = get_dispatcher()
#
# class MyCustomEvent(BaseEvent):
#     data: str
#
# @dispatcher.span
# def my_traced_function(data: str) -> None:
#     """This function gets its own span in traces"""
#     dispatcher.event(MyCustomEvent(data=data))
#     print(f"Processing: {data}")


# --- 4. Third-party observability tools ---
# Arize Phoenix:
#   Real-time tracing and visualization for workflows.
#   pip install arize-phoenix
#   See: https://docs.arize.com/phoenix/integrations/frameworks/llamaindex/
#
# Langfuse:
#   Production monitoring directly integrated with workflows instrumentation.
#   pip install langfuse
#   See: https://github.com/langfuse/langfuse
#
# Opik:
#   Receives traces through the OpenTelemetry pipeline.
#   Configure your OTLP exporter to point to Opik's endpoint.
#   See: https://www.comet.com/docs/opik/integrations/opentelemetry-python-sdk


# --- 5. Run the workflow with verbose output ---
async def main():
    # verbose=True provides basic step logging to the console
    workflow = ObservableWorkflow(timeout=30, verbose=True)
    result = await workflow.run()
    print(f"\nResult: {result}")

    print("\n" + "-" * 60)
    print("To enable full OpenTelemetry tracing:")
    print("  uv add llama-index-observability-otel")
    print("\nSupported observability backends:")
    print("  • Arize Phoenix — real-time tracing & visualization")
    print("  • Langfuse — production monitoring")
    print("  • Opik — OpenTelemetry-based trace collection")
    print("-" * 60)


if __name__ == "__main__":
    asyncio.run(main())
