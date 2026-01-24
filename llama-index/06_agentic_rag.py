import asyncio
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, SummaryIndex, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import FunctionAgent
from settings import settings


"""
-------------------------------------------------------
In this example, we explore LlamaIndex with the following features:
- Multiple query engines (vector search vs summary)
- QueryEngineTool for wrapping engines as agent tools
- Agentic RAG pattern with tool selection
- similarity_top_k for controlling retrieval

This demonstrates LlamaIndex's strength: giving agents the ability to intelligently
select different retrieval strategies based on the question type.

For more details, visit:
https://www.llamaindex.ai/blog/agentic-rag-with-llamaindex-2721b8a49ff6
-------------------------------------------------------
"""


async def main():
    # --- 1. Configure the LLM and embedding model ---
    llm = OpenAI(
        model=settings.OPENAI_MODEL_NAME,
        api_key=settings.OPENAI_API_KEY.get_secret_value()
    )
    Settings.llm = llm
    Settings.embed_model = OpenAIEmbedding(
        model=settings.OPENAI_EMBEDDINGS_MODEL,
        api_key=settings.OPENAI_API_KEY.get_secret_value()
    )

    print("LLM and embedding models configured")
    print("-" * 50)

    # --- 2. Load documents and create index ---
    documents = SimpleDirectoryReader(
        input_files=["res/metagpt.pdf"]
    ).load_data()

    print(f"Loaded {len(documents)} document(s)")
    print("-" * 50)

    # --- 3. Create multiple query engines with different strategies ---
    # Vector index for semantic search (good for specific facts)
    vector_index = VectorStoreIndex.from_documents(documents)
    vector_engine = vector_index.as_query_engine(
        similarity_top_k=3,  # Return top 3 most relevant chunks
        response_mode="compact"  # Combine chunks into a single response
    )

    # Summary index for high-level overviews (good for summarization)
    summary_index = SummaryIndex.from_documents(documents)
    summary_engine = summary_index.as_query_engine(
        response_mode="tree_summarize"  # Hierarchical summarization
    )

    print("Created vector and summary query engines")
    print("-" * 50)

    # --- 4. Wrap query engines as tools ---
    # The agent can now intelligently choose which tool to use
    vector_tool = QueryEngineTool(
        query_engine=vector_engine,
        metadata=ToolMetadata(
            name="vector_search",
            description=(
                "Useful for searching specific information, facts, and details "
                "from the MetaGPT research paper. Use this when you need precise "
                "information or technical details."
            )
        )
    )

    summary_tool = QueryEngineTool(
        query_engine=summary_engine,
        metadata=ToolMetadata(
            name="document_summary",
            description=(
                "Useful for getting high-level overviews and summaries of the "
                "MetaGPT research paper. Use this when you need general understanding "
                "or a broad summary."
            )
        )
    )

    print("Created QueryEngineTools with different strategies")
    print("-" * 50)

    # --- 5. Create agent with both tools ---
    agent = FunctionAgent(
        llm=llm,
        tools=[vector_tool, summary_tool],
        verbose=True
    )

    print("Agent created with multiple query strategies")
    print("-" * 50)

    # --- 6. Ask specific question (should use vector_search) ---
    query1 = "What are the specific roles defined in MetaGPT?"
    response1 = await agent.achat(query1)

    print(f"Query 1: {query1}")
    print(f"Response: {response1}")
    print("-" * 50)

    # --- 7. Ask for overview (should use document_summary) ---
    query2 = "Give me a high-level overview of what MetaGPT is about"
    response2 = await agent.achat(query2)

    print(f"Query 2: {query2}")
    print(f"Response: {response2}")
    print("-" * 50)

    """
    Expected output:
    - Query 1: Agent selects vector_search for specific information
    - Query 2: Agent selects document_summary for high-level overview
    - Demonstrates intelligent tool selection based on query type
    - Shows LlamaIndex's agentic RAG strength
    """


if __name__ == "__main__":
    asyncio.run(main())
