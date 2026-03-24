from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from typing_extensions import TypedDict

from settings import settings

load_dotenv()

"""
-----------------------------------------------------------------------
In this example, we explore LangGraph with the following features:
- Composing graphs using subgraphs (graph as a node)
- Shared state between parent and child graphs
- Different state schemas for parent and subgraph
- Modular agent architecture with subgraph composition

Subgraphs let you compose complex systems from smaller, self-contained
graphs. A compiled graph can be added as a node in a parent graph. The
subgraph can share state keys with the parent or use a completely
different schema with a translation layer. This enables modular
architectures where teams build independent components.

For more details, visit:
https://docs.langchain.com/oss/python/langgraph/use-subgraphs
-----------------------------------------------------------------------
"""

llm = ChatOpenAI(model=settings.OPENAI_MODEL_NAME)

# --------------------------------------------------------------
# Example 1: Subgraph with Shared State
# --------------------------------------------------------------
print("=== Example 1: Subgraph with Shared State ===\n")


# Parent and child share the same MessagesState
def researcher(state: MessagesState):
    """Research subgraph: gathers information."""
    response = llm.invoke(
        [
            SystemMessage(
                content="You are a researcher. Provide 3 key facts about the topic in the last message. Be very concise — one line per fact."
            ),
        ]
        + state["messages"]
    )
    return {"messages": [response]}


def fact_checker(state: MessagesState):
    """Fact-checking node within the research subgraph."""
    response = llm.invoke(
        [
            SystemMessage(
                content="You are a fact-checker. Review the facts in the conversation and add a brief verification note. Be concise."
            ),
        ]
        + state["messages"]
    )
    return {"messages": [response]}


# Build the research subgraph
research_builder = StateGraph(MessagesState)
research_builder.add_node("research", researcher)
research_builder.add_node("fact_check", fact_checker)
research_builder.add_edge(START, "research")
research_builder.add_edge("research", "fact_check")
research_builder.add_edge("fact_check", END)

research_subgraph = research_builder.compile()


# Writer node in parent
def writer(state: MessagesState):
    """Write a summary using the researched and verified facts."""
    response = llm.invoke(
        [
            SystemMessage(
                content="You are a writer. Using the research and fact-checks in the conversation, write a concise 2-3 sentence summary. Only output the summary."
            ),
        ]
        + state["messages"]
    )
    return {"messages": [response]}


# Build the parent graph with subgraph as a node
parent_builder = StateGraph(MessagesState)
parent_builder.add_node("research_team", research_subgraph)  # Subgraph as node
parent_builder.add_node("writer", writer)
parent_builder.add_edge(START, "research_team")
parent_builder.add_edge("research_team", "writer")
parent_builder.add_edge("writer", END)

parent_graph = parent_builder.compile()

result = parent_graph.invoke(
    {
        "messages": [
            HumanMessage(content="Tell me about the James Webb Space Telescope")
        ],
    }
)

for msg in result["messages"]:
    role = msg.__class__.__name__.replace("Message", "")
    content = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
    print(f"[{role}] {content}\n")


# --------------------------------------------------------------
# Example 2: Subgraph with Different State
# --------------------------------------------------------------
print("\n=== Example 2: Subgraph with Different State ===\n")


class AnalysisState(TypedDict):
    """Subgraph uses its own state for internal processing."""

    text: str
    word_count: int
    sentiment: str
    summary: str


def count_words(state: AnalysisState) -> dict:
    words = len(state["text"].split())
    return {"word_count": words}


def analyze_sentiment(state: AnalysisState) -> dict:
    response = llm.invoke(
        [
            SystemMessage(
                content="Classify the sentiment of this text as 'positive', 'negative', or 'neutral'. Reply with one word only."
            ),
            HumanMessage(content=state["text"]),
        ]
    )
    return {"sentiment": response.content.strip().lower()}


def summarize_text(state: AnalysisState) -> dict:
    response = llm.invoke(
        [
            SystemMessage(content="Summarize this text in one sentence."),
            HumanMessage(content=state["text"]),
        ]
    )
    return {"summary": response.content}


# Build analysis subgraph
analysis_builder = StateGraph(AnalysisState)
analysis_builder.add_node("count", count_words)
analysis_builder.add_node("sentiment", analyze_sentiment)
analysis_builder.add_node("summarize", summarize_text)
analysis_builder.add_edge(START, "count")
analysis_builder.add_edge("count", "sentiment")
analysis_builder.add_edge("sentiment", "summarize")
analysis_builder.add_edge("summarize", END)

analysis_subgraph = analysis_builder.compile()


# Parent graph calls subgraph with state translation
class ParentState(TypedDict):
    input_text: str
    analysis_result: str


def call_analysis(state: ParentState) -> dict:
    """Call the analysis subgraph, translating state."""
    result = analysis_subgraph.invoke(
        {"text": state["input_text"], "word_count": 0, "sentiment": "", "summary": ""}
    )
    formatted = f"Words: {result['word_count']}, Sentiment: {result['sentiment']}, Summary: {result['summary']}"
    return {"analysis_result": formatted}


parent2_builder = StateGraph(ParentState)
parent2_builder.add_node("analyze", call_analysis)
parent2_builder.add_edge(START, "analyze")
parent2_builder.add_edge("analyze", END)

parent2_graph = parent2_builder.compile()

result = parent2_graph.invoke(
    {
        "input_text": "LangGraph is an excellent framework for building stateful AI agents. It provides powerful tools for graph-based workflows, persistence, and streaming. The developer experience is smooth and well-documented.",
        "analysis_result": "",
    }
)

print(f"Input: {result['input_text'][:80]}...")
print(f"Analysis: {result['analysis_result']}")
