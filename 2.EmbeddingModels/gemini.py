from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    output_dimensionality=768
)

# Single query embedding
vector = embeddings.embed_query("hello, world!")
print("--- Single Query ---")
print(f"First 5 dims : {vector[:5]}")
print(f"Dimensionality: {len(vector)}")

# Multiple document embeddings
documents = [
    "The cat is on the table.",
    "The dog is in the garden.",
    "The bird is flying in the sky."
]

print("\n--- Document Embeddings ---")
vectors = embeddings.embed_documents(documents)

for i, vector in enumerate(vectors):
    print(f"Document {i}: {vector[:5]}...")

print(f"\nDimensionality: {len(vectors[0])}")