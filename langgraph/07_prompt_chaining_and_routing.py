from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from settings import settings

load_dotenv()

"""
-----------------------------------------------------------------------
In this example, we explore LangGraph with the following features:
- Prompt chaining: sequential LLM calls where each builds on the last
- LLM-based routing with structured output
- Conditional edges driven by LLM classification

Prompt chaining passes the output of one LLM call as input to the next,
useful for multi-step reasoning (e.g., research -> draft -> polish).
LLM-based routing uses structured output to classify input and route it
to specialized handlers — combining the LLM's understanding with graph
control flow.

For more details, visit:
https://docs.langchain.com/oss/python/langgraph/workflows-agents
-----------------------------------------------------------------------
"""

llm = ChatOpenAI(model=settings.OPENAI_MODEL_NAME)

# --------------------------------------------------------------
# Example 1: Prompt Chaining
# --------------------------------------------------------------
print("=== Example 1: Prompt Chaining ===\n")


class ChainState(TypedDict):
    topic: str
    outline: str
    draft: str
    final: str


def outline_node(state: ChainState) -> dict:
    """Step 1: Generate an outline."""
    response = llm.invoke(
        [
            SystemMessage(
                content="Create a brief 3-point outline for a short paragraph about the given topic. Just list the 3 points, one per line."
            ),
            HumanMessage(content=state["topic"]),
        ]
    )
    return {"outline": response.content}


def draft_node(state: ChainState) -> dict:
    """Step 2: Write a draft based on the outline."""
    response = llm.invoke(
        [
            SystemMessage(
                content="Write a concise paragraph (3-4 sentences) based on this outline. Do not include any heading."
            ),
            HumanMessage(content=state["outline"]),
        ]
    )
    return {"draft": response.content}


def polish_node(state: ChainState) -> dict:
    """Step 3: Polish the draft."""
    response = llm.invoke(
        [
            SystemMessage(
                content="Improve this paragraph: make it more engaging and concise. Return only the improved paragraph."
            ),
            HumanMessage(content=state["draft"]),
        ]
    )
    return {"final": response.content}


chain_builder = StateGraph(ChainState)
chain_builder.add_node("outline", outline_node)
chain_builder.add_node("draft", draft_node)
chain_builder.add_node("polish", polish_node)
chain_builder.add_edge(START, "outline")
chain_builder.add_edge("outline", "draft")
chain_builder.add_edge("draft", "polish")
chain_builder.add_edge("polish", END)

chain_graph = chain_builder.compile()
result = chain_graph.invoke(
    {"topic": "The future of renewable energy", "outline": "", "draft": "", "final": ""}
)

print(f"Topic: {result['topic']}")
print(f"\nOutline:\n{result['outline']}")
print(f"\nDraft:\n{result['draft']}")
print(f"\nPolished:\n{result['final']}")


# --------------------------------------------------------------
# Example 2: LLM-based Routing
# --------------------------------------------------------------
print("\n\n=== Example 2: LLM-based Routing ===\n")


class RouteDecision(BaseModel):
    """The routing decision for a user query."""

    category: str = Field(description="One of: 'technical', 'billing', 'general'")
    reasoning: str = Field(
        description="Brief explanation of why this category was chosen"
    )


class RouterState(TypedDict):
    query: str
    category: str
    reasoning: str
    response: str


router_llm = llm.with_structured_output(RouteDecision)


def classify_node(state: RouterState) -> dict:
    """Classify the query using structured output."""
    decision = router_llm.invoke(
        [
            SystemMessage(
                content="Classify the user query into one of: 'technical', 'billing', 'general'."
            ),
            HumanMessage(content=state["query"]),
        ]
    )
    return {"category": decision.category, "reasoning": decision.reasoning}


def technical_handler(state: RouterState) -> dict:
    resp = llm.invoke(
        [
            SystemMessage(
                content="You are a technical support agent. Answer concisely in 1-2 sentences."
            ),
            HumanMessage(content=state["query"]),
        ]
    )
    return {"response": resp.content}


def billing_handler(state: RouterState) -> dict:
    resp = llm.invoke(
        [
            SystemMessage(
                content="You are a billing support agent. Answer concisely in 1-2 sentences."
            ),
            HumanMessage(content=state["query"]),
        ]
    )
    return {"response": resp.content}


def general_handler(state: RouterState) -> dict:
    resp = llm.invoke(
        [
            SystemMessage(
                content="You are a general support agent. Answer concisely in 1-2 sentences."
            ),
            HumanMessage(content=state["query"]),
        ]
    )
    return {"response": resp.content}


def route_query(state: RouterState) -> str:
    """Route based on the classified category."""
    category = state["category"].lower()
    if category == "technical":
        return "technical"
    elif category == "billing":
        return "billing"
    return "general"


router_builder = StateGraph(RouterState)
router_builder.add_node("classify", classify_node)
router_builder.add_node("technical", technical_handler)
router_builder.add_node("billing", billing_handler)
router_builder.add_node("general", general_handler)

router_builder.add_edge(START, "classify")
router_builder.add_conditional_edges(
    "classify",
    route_query,
    {
        "technical": "technical",
        "billing": "billing",
        "general": "general",
    },
)
router_builder.add_edge("technical", END)
router_builder.add_edge("billing", END)
router_builder.add_edge("general", END)

router_graph = router_builder.compile()

queries = [
    "My server keeps returning 503 errors after the latest deployment",
    "I was double-charged on my last invoice",
    "What are your business hours?",
]

for query in queries:
    result = router_graph.invoke(
        {"query": query, "category": "", "reasoning": "", "response": ""}
    )
    print(f"Query: {result['query']}")
    print(f"  Routed to: {result['category']} ({result['reasoning']})")
    print(f"  Response: {result['response']}\n")
