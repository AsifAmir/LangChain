import os
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Sample documents — multiple docs on same topic to test MMR diversity
docs = [
    # LangChain related (similar to each other)
    Document(page_content="LangChain is a framework for building applications powered by large language models."),
    Document(page_content="LangChain provides tools to connect LLMs with external data sources and APIs."),
    Document(page_content="LangChain supports chains, agents, and memory for building LLM applications."),
    Document(page_content="LangChain makes it easy to build chatbots, summarizers, and question answering systems."),

    # Vector database related (similar to each other)
    Document(page_content="FAISS is an in-memory vector store developed by Facebook for fast similarity search."),
    Document(page_content="Chroma is a lightweight vector database for storing and searching document embeddings."),
    Document(page_content="Pinecone is a cloud-based vector database that scales to billions of vectors."),
    Document(page_content="Vector databases store embeddings and allow semantic search over large document collections."),

    # Embeddings related (similar to each other)
    Document(page_content="Embeddings are numerical vector representations of text that capture semantic meaning."),
    Document(page_content="OpenAI embeddings convert text into high-dimensional vectors for semantic search."),
    Document(page_content="Google Gemini embeddings support up to 768 dimensions for document representation."),
    Document(page_content="Sentence transformers generate embeddings that place similar sentences close together in vector space."),

    # Retrieval related (similar to each other)
    Document(page_content="MMR retrieval balances relevance and diversity to avoid returning repetitive results."),
    Document(page_content="Similarity search retrieves the most semantically similar documents to a given query."),
    Document(page_content="Hybrid search combines keyword search and semantic search for better retrieval accuracy."),
    Document(page_content="Retrievers in LangChain fetch relevant documents from a vector store based on a query."),
]

# Initialize embedding model
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    output_dimensionality=768
)

# delete folder if exists to start fresh
if os.path.exists("my_faiss_db"):
    import shutil
    shutil.rmtree("my_faiss_db")

# Initialize FAISS vector store
vector_store = FAISS.from_documents(
    documents=docs, 
    embedding=embeddings
    )

# save to disk
vector_store.save_local("my_faiss_db")

# Enable MMR in the retriever
# MMR (Maximal Marginal Relevance) balances relevance and diversity in search results
# It avoids returning repetitive results by penalizing documents too similar to already selected ones
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,           # number of results to return
        "fetch_k": 10,    # candidates to consider before MMR re-ranking
        "lambda_mult": 0.5  # 0 = max diversity, 1 = max relevance
    }
)

# Query the retriever
query = "What is langchain?"
results = retriever.invoke(query)
print(f"Query: {query}")
print("\n--- Retriever Results ---")
for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)