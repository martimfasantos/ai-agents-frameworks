import asyncio
from llama_index.core import SimpleDirectoryReader, SummaryIndex, VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.tools import QueryEngineTool
from llama_index.core.selectors import LLMSingleSelector
from settings import settings


"""
-------------------------------------------------------
In this example, we explore LlamaIndex with the following features:
- Router query engines for intelligent query routing
- Multiple index types (Vector and Summary)
- LLM-based query engine selection
- Query routing based on query characteristics
- Dynamic index selection for optimal results

Router query engines are a LlamaIndex-specific strength that enables automatic
selection of the most appropriate query strategy based on the user's question.
This is particularly useful when you have different types of indices optimized
for different query patterns.

For more details, visit:
https://docs.llamaindex.ai/en/stable/module_guides/querying/router/
-------------------------------------------------------
"""


async def main():
    # --- 1. Configure the LLM and embedding model ---
    Settings.llm = OpenAI(
        model=settings.OPENAI_MODEL_NAME,
        api_key=settings.OPENAI_API_KEY.get_secret_value()
    )
    Settings.embed_model = OpenAIEmbedding(
        model=settings.OPENAI_EMBEDDINGS_MODEL,
        api_key=settings.OPENAI_API_KEY.get_secret_value()
    )

    print("LLM and embedding models configured")
    print("-" * 50)

    # --- 2. Load and process documents ---
    documents = SimpleDirectoryReader(
        input_files=["res/metagpt.pdf"]
    ).load_data()

    splitter = SentenceSplitter(chunk_size=1024)
    nodes = splitter.get_nodes_from_documents(documents)

    print(f"Loaded and split {len(documents)} document(s) into {len(nodes)} nodes")
    print("-" * 50)

    # --- 3. Create multiple index types ---
    # Summary index: Good for summarization and overview questions
    summary_index = SummaryIndex(nodes)

    # Vector index: Good for specific information retrieval
    vector_index = VectorStoreIndex(nodes)

    print("Created Summary Index and Vector Index")
    print("-" * 50)

    # --- 4. Define query engines for each index ---
    summary_query_engine = summary_index.as_query_engine(
        response_mode="tree_summarize",
        use_async=True,
    )
    vector_query_engine = vector_index.as_query_engine()

    print("Created query engines")
    print("-" * 50)

    # --- 5. Create query engine tools with metadata ---
    # The description helps the LLM understand when to use each tool
    summary_tool = QueryEngineTool.from_defaults(
        query_engine=summary_query_engine,
        description=(
            "Useful for summarization questions related to MetaGPT. "
            "Use this for questions asking for overviews, summaries, or general information."
        ),
    )

    vector_tool = QueryEngineTool.from_defaults(
        query_engine=vector_query_engine,
        description=(
            "Useful for retrieving specific context from the MetaGPT paper. "
            "Use this for questions asking about specific details, mechanisms, or implementations."
        ),
    )

    print("Created query engine tools with descriptive metadata")
    print("-" * 50)

    # --- 6. Create the router query engine ---
    # The router uses an LLM to select the most appropriate query engine
    query_engine = RouterQueryEngine(
        selector=LLMSingleSelector.from_defaults(),
        query_engine_tools=[
            summary_tool,
            vector_tool,
        ],
        verbose=True  # Show which engine is selected
    )

    print("Created router query engine")
    print("-" * 50)

    # --- 7. Run a summarization query ---
    # This should route to the summary index
    print("Example 1: Summarization query (should use Summary Index)")
    print("-" * 50)

    query1 = "What is the summary of the document?"
    response1 = query_engine.query(query1)

    print(f"Query: {query1}")
    print("-" * 50)
    print(f"Response: {response1}")
    print("-" * 50)
    print(f"Number of source nodes: {len(response1.source_nodes)}")
    print("-" * 50)

    """
    Expected output:
    Selecting query engine 0: The question asks for a summary of the document...
    
    Response: A comprehensive summary of the MetaGPT framework...
    Number of source nodes: High (summary index processes all nodes)
    """

    # --- 8. Run a specific detail query ---
    # This should route to the vector index
    print("Example 2: Specific detail query (should use Vector Index)")
    print("-" * 50)

    query2 = "How do agents share information with other agents?"
    response2 = query_engine.query(query2)

    print(f"Query: {query2}")
    print("-" * 50)
    print(f"Response: {response2}")
    print("-" * 50)
    print(f"Number of source nodes: {len(response2.source_nodes)}")
    print("-" * 50)

    """
    Expected output:
    Selecting query engine 1: The question pertains to specific mechanisms...
    
    Response: Agents share information through a shared message pool...
    Number of source nodes: Low (vector index retrieves only relevant nodes)
    """

    # --- 9. Run additional queries to demonstrate routing ---
    print("Example 3: Testing routing with different query types")
    print("-" * 50)

    test_queries = [
        "What are the main contributions of this paper?",
        "Explain the specific architecture of the message pool.",
        "Give me an overview of MetaGPT's key features."
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"Test Query {i}: {query}")
        response = query_engine.query(query)
        print(f"Response preview: {str(response)[:150]}...")
        print("-" * 50)

    """
    Expected output:
    - Different queries route to different engines based on their nature
    - Summarization/overview queries → Summary Index
    - Specific detail queries → Vector Index
    - Demonstrates intelligent routing based on query characteristics
    """


if __name__ == "__main__":
    asyncio.run(main())
