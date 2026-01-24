import asyncio
from llama_index.llms.openai import OpenAI
from workflows import Workflow, Context, step
from workflows.events import Event, StartEvent, StopEvent
from settings import settings


"""
-------------------------------------------------------
In this example, we explore LlamaIndex Workflows with the following features:
- Streaming events to provide real-time progress updates
- Using Context.write_event_to_stream() to emit progress events
- Streaming LLM responses token by token
- Handling workflow termination events

Workflows support event streaming to provide users with real-time feedback
during long-running operations.

For more details, visit:
https://developers.llamaindex.ai/python/llamaagents/workflows/streaming/
-------------------------------------------------------
"""


# Define progress event for streaming
class ProgressEvent(Event):
    msg: str


class ProcessingEvent(Event):
    data: str


class StreamingWorkflow(Workflow):
    @step
    async def step_one(self, ctx: Context, ev: StartEvent) -> ProcessingEvent:
        """First step - emit progress event"""
        ctx.write_event_to_stream(ProgressEvent(msg="Processing..."))
        await asyncio.sleep(0.5)
        return ProcessingEvent(data="Data from step 1")

    @step
    async def step_two(self, ctx: Context, ev: ProcessingEvent) -> StopEvent:
        """Second step - stream LLM response"""
        llm = OpenAI(
            model=settings.OPENAI_MODEL_NAME,
            api_key=settings.OPENAI_API_KEY.get_secret_value()
        )
        
        generator = await llm.astream_complete(
            "Write a short haiku about workflows."
        )
        
        full_response = ""
        async for response in generator:
            ctx.write_event_to_stream(ProgressEvent(msg=response.delta))
            full_response += response.delta
        
        return StopEvent(result=full_response)


async def main():
    workflow = StreamingWorkflow(timeout=60, verbose=False)
    handler = workflow.run()
    
    # Listen for streamed events
    async for event in handler.stream_events():
        if isinstance(event, ProgressEvent):
            print(event.msg, end="", flush=True)
    
    final_result = await handler
    print(f"\n\nResult: {final_result}")


if __name__ == "__main__":
    asyncio.run(main())
