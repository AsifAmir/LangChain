from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
)

# Single text
vector = embeddings.embed_query("What is the meaning of life?")
print("--- Single text ---")
print(f"Dimensions  : {len(vector)}")      # 384
print(f"First 5     : {vector[:5]}")

# Multiple texts (batch)
vectors = embeddings.embed_documents([
    "What is the meaning of life?",
    "How does AI work?",
    "What is a vector?"
])

print("--- Multiple texts ---")
for i, vector in enumerate(vectors):
    print(f"Document {i} - Dimensions: {len(vector)}, First 5: {vector[:5]}")

print(f"Dimensionality of the embedding vector: {len(vectors[0])}")