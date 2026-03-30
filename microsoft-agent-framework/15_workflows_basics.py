import asyncio
from dataclasses import dataclass

from dotenv import load_dotenv

from agent_framework import (
    Executor,
    WorkflowBuilder,
    WorkflowContext,
    handler,
    executor,
)

from settings import settings

load_dotenv()

"""
-------------------------------------------------------
In this example, we explore Microsoft Agent Framework
with the following features:
- Class-based Executors with @handler methods
- Function-based executors with @executor decorator
- WorkflowBuilder for connecting executors with edges
- Conditional edges for branching logic

Workflows are the low-level building blocks for complex
agent pipelines. Executors process messages, edges define
data flow, and conditional edges enable branching based
on runtime values — giving you full control over execution.

For more details, visit:
https://learn.microsoft.com/en-us/agent-framework/workflows/workflows?pivots=programming-language-python
-------------------------------------------------------
"""


# --- 1. Define data types ---
@dataclass
class TextInput:
    text: str


@dataclass
class AnalysisResult:
    original: str
    word_count: int
    char_count: int
    is_long: bool


@dataclass
class Summary:
    message: str


# --- 2. Define a class-based executor ---
class TextAnalyzer(Executor):
    """Analyzes text and produces metrics."""

    @handler
    async def handle(
        self, message: TextInput, ctx: WorkflowContext[AnalysisResult]
    ) -> None:
        words = message.text.split()
        result = AnalysisResult(
            original=message.text,
            word_count=len(words),
            char_count=len(message.text),
            is_long=len(words) > 10,
        )
        print(
            f"[TextAnalyzer] Analyzed: {result.word_count} words, {result.char_count} chars"
        )
        await ctx.send_message(result)


# --- 3. Define function-based executors ---
@executor(id="short-formatter")
async def format_short(message: AnalysisResult, ctx: WorkflowContext[Summary]) -> None:
    """Formats short text analysis."""
    summary = Summary(
        message=f"Short text ({message.word_count} words): '{message.original}'"
    )
    print(f"[ShortFormatter] {summary.message}")
    await ctx.yield_output(summary)


@executor(id="long-formatter")
async def format_long(message: AnalysisResult, ctx: WorkflowContext[Summary]) -> None:
    """Formats long text analysis."""
    summary = Summary(
        message=(
            f"Long text detected! {message.word_count} words, "
            f"{message.char_count} characters. Preview: '{message.original[:50]}...'"
        )
    )
    print(f"[LongFormatter] {summary.message}")
    await ctx.yield_output(summary)


async def main() -> None:
    # --- 4. Build a workflow with conditional edges ---
    analyzer = TextAnalyzer(id="text-analyzer")

    workflow = (
        WorkflowBuilder(start_executor=analyzer)
        # Route to short formatter if text is short
        .add_edge(analyzer, format_short, condition=lambda msg: not msg.is_long)
        # Route to long formatter if text is long
        .add_edge(analyzer, format_long, condition=lambda msg: msg.is_long)
        .build()
    )

    # --- 5. Run with a short text ---
    print("=== Short Text ===")
    result = await workflow.run(TextInput(text="Hello world"))
    outputs = result.get_outputs()
    for output in outputs:
        print(f"Output: {output}")

    print()

    # --- 6. Run with a long text ---
    print("=== Long Text ===")
    result = await workflow.run(
        TextInput(
            text="This is a much longer piece of text that contains many more words and should trigger the long path"
        )
    )
    outputs = result.get_outputs()
    for output in outputs:
        print(f"Output: {output}")


if __name__ == "__main__":
    asyncio.run(main())
