import os
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# delete folder if exists to start fresh
if os.path.exists("my_chroma_db"):
    import shutil
    shutil.rmtree("my_chroma_db")

# Step 1: Your source documents
documents = [
    Document(page_content="LangChain helps developers build LLM applications easily."),
    Document(page_content="Chroma is a vector database optimized for LLM-based search."),
    Document(page_content="Embeddings convert text into high-dimensional vectors."),
    Document(page_content="Google Generative AI provides powerful embedding models."),
]

# Step 2: Initialize embedding model
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    output_dimensionality=768
)

# Step 3: Create Chroma vector store in memory
vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="sample",
    persist_directory="my_chroma_db"
)

# Step 4: Convert vector_store into a retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

# Step 5: Query the retriever
query = "What is Chroma used for?"
results = retriever.invoke(query)

# Step 6: Print results
print("\n--- Retriever Results ---")
for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)

# Step 7: Direct similarity search on vector store
results = vector_store.similarity_search(query, k=2)

# Step 8: Print similarity search results
print("\n--- Similarity Search Results ---")
for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)