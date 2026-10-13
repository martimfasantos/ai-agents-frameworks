import shutil

from dotenv import load_dotenv

from agno.knowledge import Knowledge
from agno.knowledge.embedder.openai import OpenAIEmbedder
from agno.knowledge.query_transformer.hyde import HyDE
from agno.models.openai import OpenAIChat
from agno.vectordb.lancedb import LanceDb

from settings import settings

load_dotenv()

"""
-------------------------------------------------------
In this example, we explore Agno with the following features:
- Query transformation on Knowledge, added in Agno v3.1.2
- HyDE (Hypothetical Document Embeddings) as the query transformer
- Plain search vs HyDE search over the same LanceDb table

A question and the passage that answers it rarely share wording, so the
question's embedding can land closer to documents that merely repeat its
words. HyDE asks an LLM to write a hypothetical answer first and searches with
that instead. The invented answer is never shown to the user and does not need
to be correct; it only has to look like the documents being searched for.

For more details, visit:
https://docs.agno.com/knowledge/concepts/search-and-retrieval/query-transformation
-------------------------------------------------------
"""

LANCE_URI = "/tmp/agno_hyde_example"
QUESTION = "Why did our sales numbers fall?"

shutil.rmtree(LANCE_URI, ignore_errors=True)

# --- 1. One vector table, shared by both knowledge bases ---
vector_db = LanceDb(
    uri=LANCE_URI,
    table_name="company_wiki",
    embedder=OpenAIEmbedder(
        id="text-embedding-3-small",
        api_key=settings.OPENAI_API_KEY.get_secret_value(),
    ),
)

plain_knowledge = Knowledge(vector_db=vector_db)
hyde_knowledge = Knowledge(
    vector_db=vector_db,
    query_transformer=HyDE(model=OpenAIChat(id=settings.OPENAI_MODEL_NAME)),
)

# --- 2. Add mock wiki pages ---
# Only the first page answers the question, and it shares none of its words.
documents = [
    (
        "Q3 business review",
        "Three enterprise accounts postponed their contract renewals to next year, and two mid-market customers churned to a cheaper competitor after the price increase.",
    ),
    (
        "Sales dashboard guide",
        "The sales dashboard shows numbers for each region. If the numbers fall to zero, the nightly data sync failed; press Refresh to reload them.",
    ),
    (
        "Sales team offsite",
        "The sales team offsite is in Lisbon in October. Numbers are limited, so register early.",
    ),
    (
        "Expense policy",
        "Expenses above 500 EUR need manager approval before purchase.",
    ),
]
for name, text in documents:
    plain_knowledge.insert(name=name, text_content=text)


# --- 3. Search the same question both ways ---
def show_results(label: str, knowledge: Knowledge) -> None:
    print(f"=== {label} ===")
    for rank, document in enumerate(knowledge.search(QUESTION, max_results=2), start=1):
        print(f"  {rank}. {document.name}: {document.content[:60]}…")
    print()


print(f"Question: {QUESTION}\n")
show_results("Plain search (embeds the question)", plain_knowledge)
show_results("HyDE search (embeds a hypothetical answer)", hyde_knowledge)
