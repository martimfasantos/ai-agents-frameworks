import asyncio
from typing import Any

from dotenv import load_dotenv

from agent_framework import (
    Agent,
    AgentSession,
    BaseContextProvider,
    Message,
    SessionContext,
    SupportsAgentRun,
)
from agent_framework.openai import OpenAIChatClient

from settings import settings

load_dotenv()

"""
-------------------------------------------------------
In this example, we explore Microsoft Agent Framework
with the following features:
- RAG using custom context providers
- Injecting retrieved context via extend_instructions
- BaseContextProvider.before_run for pre-query retrieval

Context providers let you inject relevant documents or
data into the agent's context before each LLM call. This
is the framework's built-in RAG pattern — you implement
the retrieval logic in before_run, and the framework
handles injection into the conversation context.

For more details, visit:
https://learn.microsoft.com/en-us/agent-framework/agents/rag?pivots=programming-language-python
-------------------------------------------------------
"""


# --- 1. Simulated knowledge base ---
KNOWLEDGE_BASE = {
    "refund": (
        "Refund Policy: Customers may request a full refund within 30 days of purchase. "
        "After 30 days, a 15% restocking fee applies. Digital products are non-refundable "
        "once downloaded. To request a refund, contact support@example.com with your order number."
    ),
    "shipping": (
        "Shipping Policy: Standard shipping takes 5-7 business days. Express shipping "
        "(2-3 days) costs $12.99. Free shipping on orders over $50. International shipping "
        "is available to 40+ countries with delivery in 10-15 business days."
    ),
    "warranty": (
        "Warranty Policy: All products come with a 1-year limited warranty covering "
        "manufacturing defects. Extended warranty (3 years) available for $29.99. "
        "Warranty does not cover accidental damage or normal wear and tear."
    ),
}


# --- 2. Create a custom context provider ---
class PolicyContextProvider(BaseContextProvider):
    """Retrieves relevant policy documents and injects them as extra instructions."""

    async def before_run(
        self,
        *,
        agent: SupportsAgentRun,
        session: AgentSession,
        context: SessionContext,
        state: dict[str, Any],
    ) -> None:
        # Get the latest user message for keyword matching
        messages = context.get_messages(include_input=True)
        query = ""
        if messages:
            last_msg = messages[-1]
            if last_msg.text:
                query = last_msg.text.lower()

        # Simple keyword-based retrieval (in production, use vector search)
        matched_docs = []
        for keyword, doc in KNOWLEDGE_BASE.items():
            if keyword in query:
                matched_docs.append(doc)

        if not matched_docs:
            matched_docs = list(KNOWLEDGE_BASE.values())

        # Inject retrieved context as additional instructions
        context_text = "\n\n".join(
            f"[Document {i + 1}]: {doc}" for i, doc in enumerate(matched_docs)
        )
        context.extend_instructions(
            source_id="policy-rag",
            instructions=f"Reference documents:\n{context_text}",
        )
        print(f"[RAG] Retrieved {len(matched_docs)} document(s)")


async def main() -> None:
    # --- 3. Create the agent with the context provider ---
    client = OpenAIChatClient(
        model_id=settings.OPENAI_MODEL_NAME,
        api_key=settings.OPENAI_API_KEY.get_secret_value(),
    )

    agent = client.as_agent(
        name="support-agent",
        instructions=(
            "You are a customer support agent. Answer questions using ONLY the "
            "provided reference documents. If the answer is not in the documents, "
            "say you don't have that information. Be concise and helpful."
        ),
        context_providers=[PolicyContextProvider(source_id="policy-rag")],
    )

    # --- 4. Ask questions that trigger context retrieval ---
    print("=== Question 1: Refund ===")
    result = await agent.run("Can I get a refund for a product I bought 2 weeks ago?")
    print(f"Answer: {result.text}\n")

    print("=== Question 2: Shipping ===")
    result = await agent.run("How long does international shipping take?")
    print(f"Answer: {result.text}\n")


if __name__ == "__main__":
    asyncio.run(main())
