import asyncio
import json

from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.config import get_stream_writer
from typing_extensions import TypedDict

from settings import settings

load_dotenv()

"""
-----------------------------------------------------------------------
In this example, we explore LangGraph with the following features:
- Custom streaming with get_stream_writer()
- Emitting progress updates, status messages, and structured data
- Consuming custom stream events alongside standard outputs

The custom streaming API lets nodes emit arbitrary data to the caller
during execution — progress bars, status updates, intermediate results,
or structured events. This is useful for long-running tasks where you
want to report progress without waiting for the node to complete.

For more details, visit:
https://docs.langchain.com/oss/python/langgraph/streaming
-----------------------------------------------------------------------
"""


# --- 1. Define state ---
class ProcessingState(TypedDict):
    data: list[str]
    results: list[str]


# --- 2. Define nodes that emit custom stream data ---
def ingestion_node(state: ProcessingState) -> dict:
    """Simulate data ingestion with progress updates."""
    writer = get_stream_writer()
    items = state["data"]

    writer({"type": "status", "message": "Starting data ingestion..."})

    processed = []
    for i, item in enumerate(items):
        processed.append(item.upper())
        writer(
            {
                "type": "progress",
                "step": "ingestion",
                "current": i + 1,
                "total": len(items),
                "item": item,
            }
        )

    writer(
        {
            "type": "status",
            "message": f"Ingestion complete: {len(processed)} items processed",
        }
    )
    return {"data": processed}


def analysis_node(state: ProcessingState) -> dict:
    """Analyze processed data and emit structured results."""
    writer = get_stream_writer()

    writer({"type": "status", "message": "Starting analysis..."})

    results = []
    for item in state["data"]:
        result = f"Analyzed: {item} (length={len(item)}, words={len(item.split())})"
        results.append(result)
        writer(
            {
                "type": "analysis_result",
                "item": item,
                "length": len(item),
                "words": len(item.split()),
            }
        )

    writer({"type": "status", "message": f"Analysis complete: {len(results)} results"})
    return {"results": results}


# --- 3. Build the graph ---
builder = StateGraph(ProcessingState)
builder.add_node("ingestion", ingestion_node)
builder.add_node("analysis", analysis_node)
builder.add_edge(START, "ingestion")
builder.add_edge("ingestion", "analysis")
builder.add_edge("analysis", END)

graph = builder.compile()


# --- 4. Consume custom stream events ---
async def main():
    print("=== Custom Streaming with get_stream_writer() ===\n")

    input_data = {
        "data": [
            "hello world",
            "langgraph streaming",
            "custom events are powerful",
        ],
        "results": [],
    }

    # Use stream_mode="custom" to get only custom events,
    # or combine with other modes
    async for chunk in graph.astream(input_data, stream_mode=["custom", "updates"]):
        namespace, data = chunk

        if namespace == "custom":
            event_type = data.get("type", "unknown")
            if event_type == "status":
                print(f"[STATUS] {data['message']}")
            elif event_type == "progress":
                pct = data["current"] / data["total"] * 100
                print(
                    f"[PROGRESS] {data['step']}: {data['current']}/{data['total']} ({pct:.0f}%) - {data['item']}"
                )
            elif event_type == "analysis_result":
                print(
                    f"[ANALYSIS] {data['item']} -> length={data['length']}, words={data['words']}"
                )
        elif namespace == "updates":
            for node_name in data:
                print(f"\n[NODE COMPLETE] {node_name}")

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
