from dotenv import load_dotenv

from agno.agent import Agent
from agno.knowledge import Knowledge
from agno.knowledge.embedder.openai import OpenAIEmbedder
from agno.models.openai import OpenAIChat
from agno.utils.pprint import pprint_run_response
from agno.vectordb.lancedb import LanceDb

from settings import settings

load_dotenv()

"""
-------------------------------------------------------
In this example, we explore Agno with the following features:
- Knowledge base with vector search (RAG)
- LanceDb as the vector database
- OpenAIEmbedder for embedding generation
- Adding documents to the knowledge base at startup
- Agent querying the knowledge base automatically

RAG (Retrieval-Augmented Generation) lets agents ground their
answers in your own data. Agno's Knowledge class wraps a
vector database and embedder. When attached to an agent,
relevant documents are retrieved automatically and injected
into the LLM's context.

For more details, visit:
https://docs.agno.com/knowledge/introduction
-------------------------------------------------------
"""

# --- 1. Configure the vector database and embedder ---
vector_db = LanceDb(
    uri="/tmp/agno_knowledge_example",
    table_name="recipes",
    embedder=OpenAIEmbedder(
        id="text-embedding-3-small",
        api_key=settings.OPENAI_API_KEY.get_secret_value(),
    ),
)

# --- 2. Create the knowledge base ---
knowledge = Knowledge(
    vector_db=vector_db,
)

# --- 3. Add documents to the knowledge base ---
# In production, you'd load PDFs, web pages, etc. via readers.
# Here we add a few sample documents directly using insert().
documents = [
    (
        "Pastéis de Nata",
        "Pastéis de nata are Portuguese custard tarts. The recipe originates from the Jerónimos Monastery in Belém, Lisbon. The pastry is made with puff pastry and a custard filling of egg yolks, sugar, milk, and vanilla.",
    ),
    (
        "Bacalhau à Brás",
        "Bacalhau à Brás is a classic Portuguese dish made with shredded salt cod, onions, and thinly sliced fried potatoes, all bound together with scrambled eggs and garnished with olives and parsley.",
    ),
    (
        "Francesinha",
        "Francesinha is a Portuguese sandwich from Porto. It contains cured ham, linguiça, fresh sausage, steak, and is covered with melted cheese and a thick spiced tomato-beer sauce. Typically served with french fries.",
    ),
    (
        "Caldo Verde",
        "Caldo verde is a traditional Portuguese soup made with thinly sliced collard greens (couve), potatoes, onion, garlic, and olive oil. It is often served with a slice of chouriço sausage.",
    ),
]

# Insert each document into the knowledge base
for name, text in documents:
    knowledge.insert(name=name, text_content=text)

# --- 4. Create the agent with knowledge ---
agent = Agent(
    model=OpenAIChat(id=settings.OPENAI_MODEL_NAME),
    knowledge=knowledge,
    instructions=[
        "You are a Portuguese cuisine expert.",
        "Use the knowledge base to answer questions about Portuguese food.",
        "If the knowledge base doesn't have the answer, say so honestly.",
    ],
    markdown=True,
)

# --- 5. Query the agent ---
run_output = agent.run("What is a Francesinha and where does it come from?")
pprint_run_response(run_output)

print("\n" + "=" * 60 + "\n")

run_output = agent.run("How do you make pastéis de nata?")
pprint_run_response(run_output)
