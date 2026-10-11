import asyncio
from dataclasses import dataclass, field
from typing import Annotated
from uuid import uuid4

from dotenv import load_dotenv

from agent_framework import (
    Agent,
    InMemoryCollection,
    VectorStoreField,
    create_vector_search_tool,
    vectorstoremodel,
)
from agent_framework.openai import OpenAIChatClient, OpenAIEmbeddingClient

from settings import settings

load_dotenv()

"""
-------------------------------------------------------
In this example, we explore Microsoft Agent Framework
with the following features:
- The shared vector-store abstractions added in v1.18.0
- @vectorstoremodel and VectorStoreField to describe a record
- InMemoryCollection with an embedding generator
- create_vector_search_tool to hand search to an agent

Until v1.18.0 the framework had no vector-store abstraction of its own,
so retrieval meant hand-rolling it in a ContextProvider — which is what
08_rag.py does with keyword matching. These APIs give a real semantic
search: the collection embeds records on upsert, and
create_vector_search_tool turns the collection into a tool the model
calls when it decides it needs to look something up. The same record
type and tool work against Azure AI Search, Redis, Qdrant or pgvector by
swapping the collection class.

For more details, visit:
https://learn.microsoft.com/en-us/agent-framework/agents/rag?pivots=programming-language-python
-------------------------------------------------------
"""


# --- 1. Describe the record: which field is the key, the text, the vector ---
@vectorstoremodel(collection_name="policies")
@dataclass
class Policy:
    text: Annotated[str, VectorStoreField("data")]
    topic: Annotated[str, VectorStoreField("data")]
    # A vector field carries its own source text on the way in; the collection
    # embeds that value in place, which is why the type allows both.
    embedding: Annotated[
        list[float] | str | None,
        VectorStoreField("vector", dimensions=1536),
    ] = None
    id: Annotated[str, VectorStoreField("key")] = field(
        default_factory=lambda: str(uuid4())
    )

    def __post_init__(self) -> None:
        if self.embedding is None:
            self.embedding = self.text


POLICIES = [
    Policy(
        topic="refund",
        text=(
            "Customers may request a full refund within 30 days of purchase. After 30 days "
            "a 15% restocking fee applies. Digital products are non-refundable once downloaded."
        ),
    ),
    Policy(
        topic="shipping",
        text=(
            "Standard shipping takes 5-7 business days. Express shipping costs $12.99 and "
            "arrives in 2-3 days. Orders over $50 ship free."
        ),
    ),
    Policy(
        topic="warranty",
        text=(
            "All products carry a 1-year limited warranty against manufacturing defects. "
            "A 3-year extended warranty costs $29.99. Accidental damage is not covered."
        ),
    ),
]


async def main():
    # --- 2. A collection that embeds records as they go in ---
    async with InMemoryCollection(
        record_type=Policy,
        embedding_generator=OpenAIEmbeddingClient(
            model="text-embedding-3-small",
            api_key=settings.OPENAI_API_KEY.get_secret_value(),
        ),
    ) as collection:
        await collection.ensure_collection_exists()
        await collection.upsert(POLICIES)

        print("=== Vector store RAG ===\n")

        # --- 3. Semantic search finds the right record without keyword overlap ---
        # "money back" shares no words with the refund policy text.
        results = await collection.search("can I get my money back", top=2)
        print("Direct search for 'can I get my money back':")
        # Each result is a dict of the deserialized record plus its score. The
        # in-memory collection defaults to cosine *distance*, so lower is closer
        # — the ranking is ascending, not descending.
        async for result in results.results:
            print(f"  [{result['record'].topic}] distance={result['score']:.3f}")

        # --- 4. Give the collection to an agent as a tool ---
        search_tool = create_vector_search_tool(
            collection,
            name="search_policies",
            description="Search company policy documents to answer a customer question.",
            top=2,
        )

        agent = Agent(
            client=OpenAIChatClient(
                model=settings.OPENAI_MODEL_NAME,
                api_key=settings.OPENAI_API_KEY.get_secret_value(),
            ),
            instructions=(
                "Answer customer questions using the search_policies tool. "
                "Answer in one sentence, citing the figure from the policy."
            ),
            tools=[search_tool],
        )

        for question in (
            "I bought something three weeks ago and changed my mind — what happens?",
            "How long until my parcel turns up if I don't pay extra?",
        ):
            reply = await agent.run(question)
            print(f"\nQ: {question}")
            print(f"A: {reply.text.strip()}")


if __name__ == "__main__":
    asyncio.run(main())
