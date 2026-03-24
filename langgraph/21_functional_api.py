from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.func import entrypoint, task
from langgraph.checkpoint.memory import InMemorySaver

from settings import settings

load_dotenv()

"""
-----------------------------------------------------------------------
In this example, we explore LangGraph with the following features:
- Functional API with @entrypoint and @task decorators
- Parallel task execution using futures
- entrypoint.final for decoupling return value from checkpoint state
- Persistence in functional workflows

The Functional API is an alternative to the Graph API that uses Python
decorators instead of explicit graph construction. @entrypoint defines
the workflow entry point, @task defines individual units of work that
are automatically checkpointed. Tasks return futures that can be
resolved with .result(), enabling easy parallel execution.

For more details, visit:
https://docs.langchain.com/oss/python/langgraph/functional-api
-----------------------------------------------------------------------
"""

llm = ChatOpenAI(model=settings.OPENAI_MODEL_NAME)
checkpointer = InMemorySaver()

# --------------------------------------------------------------
# Example 1: Basic Functional API
# --------------------------------------------------------------
print("=== Example 1: Basic Functional API ===\n")


@task
def analyze_topic(topic: str) -> str:
    """Analyze a single topic — automatically checkpointed."""
    response = llm.invoke(
        [
            SystemMessage(
                content="Provide a one-sentence analysis of this topic's importance."
            ),
            HumanMessage(content=topic),
        ]
    )
    return f"{topic}: {response.content}"


@entrypoint(checkpointer=checkpointer)
def research_pipeline(topics: list[str]) -> list[str]:
    """Research multiple topics in parallel using futures."""
    # Launch all tasks — they return futures
    futures = [analyze_topic(topic) for topic in topics]

    # Resolve all futures — tasks may run in parallel
    results = [f.result() for f in futures]

    return results


topics = ["Quantum Computing", "Gene Editing", "Nuclear Fusion"]
config = {"configurable": {"thread_id": "func-demo-1"}}

results = research_pipeline.invoke(topics, config=config)
for r in results:
    print(f"  {r}")


# --------------------------------------------------------------
# Example 2: entrypoint.final
# --------------------------------------------------------------
print("\n\n=== Example 2: entrypoint.final ===\n")


@task
def generate_report(topic: str) -> dict:
    """Generate a report with both content and metadata."""
    response = llm.invoke(
        [
            SystemMessage(content="Write a one-sentence report summary."),
            HumanMessage(content=f"Report on: {topic}"),
        ]
    )
    return {
        "topic": topic,
        "summary": response.content,
        "word_count": len(response.content.split()),
    }


checkpointer2 = InMemorySaver()


@entrypoint(checkpointer=checkpointer2)
def report_workflow(topic: str) -> entrypoint.final[str, dict]:
    """
    Return the summary to the caller, but save the full report dict
    to the checkpoint for later inspection.
    """
    report_future = generate_report(topic)
    report = report_future.result()

    # final(value=..., save=...):
    # - value is returned to the caller
    # - save is what gets persisted in the checkpoint
    return entrypoint.final(
        value=report["summary"],  # Caller gets just the summary string
        save=report,  # Checkpoint stores the full report dict
    )


config2 = {"configurable": {"thread_id": "func-demo-2"}}

# Caller receives just the summary
summary = report_workflow.invoke("Artificial Intelligence Ethics", config=config2)
print(f"Returned to caller: {summary}")

# But the checkpoint has the full report
state = report_workflow.get_state(config2)
print(f"Saved in checkpoint: {state.values}")


# --------------------------------------------------------------
# Example 3: Sequential Pipeline with Dependencies
# --------------------------------------------------------------
print("\n\n=== Example 3: Sequential Pipeline ===\n")


@task
def draft_content(topic: str) -> str:
    """Draft initial content."""
    response = llm.invoke(
        [
            SystemMessage(content="Write a brief 2-sentence draft about the topic."),
            HumanMessage(content=topic),
        ]
    )
    return response.content


@task
def review_content(draft: str) -> str:
    """Review and improve the draft."""
    response = llm.invoke(
        [
            SystemMessage(
                content="Improve this draft to be more engaging. Keep it to 2 sentences."
            ),
            HumanMessage(content=draft),
        ]
    )
    return response.content


@task
def add_metadata(content: str, topic: str) -> dict:
    """Add metadata to the final content."""
    return {
        "topic": topic,
        "content": content,
        "word_count": len(content.split()),
        "status": "published",
    }


checkpointer3 = InMemorySaver()


@entrypoint(checkpointer=checkpointer3)
def content_pipeline(topic: str) -> dict:
    """Sequential pipeline: draft -> review -> publish."""
    # Each step depends on the previous
    draft = draft_content(topic).result()
    print(f"  Draft: {draft[:80]}...")

    reviewed = review_content(draft).result()
    print(f"  Reviewed: {reviewed[:80]}...")

    result = add_metadata(reviewed, topic).result()
    return result


config3 = {"configurable": {"thread_id": "func-demo-3"}}

output = content_pipeline.invoke("The Future of Space Travel", config=config3)
print(f"\n  Final output:")
print(f"    Topic: {output['topic']}")
print(f"    Content: {output['content']}")
print(f"    Words: {output['word_count']}")
print(f"    Status: {output['status']}")
