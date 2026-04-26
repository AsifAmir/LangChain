from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dotenv import load_dotenv

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    output_dimensionality=768
)

documents = [
    "Artificial intelligence is transforming industries by automating complex tasks and enabling data-driven decision-making.",
    "Machine learning algorithms improve their performance over time by learning patterns from large datasets.",
    "Natural language processing allows computers to understand, interpret, and generate human language effectively.",
    "Deep learning models, inspired by the human brain, excel at recognizing patterns in images, audio, and text.",
    "Cloud computing provides scalable infrastructure that supports the deployment of AI models at enterprise scale.",
]


query = "How do computers understand human language?"

# Generate embeddings
query_vector   = embeddings.embed_query(query)
doc_vectors    = embeddings.embed_documents(documents)

# Convert to numpy arrays for cosine similarity
query_array = np.array(query_vector).reshape(1, -1)      # shape: (1, 768)
docs_array  = np.array(doc_vectors)                       # shape: (4, 768)

# Calculate similarity scores
similarities = cosine_similarity(query_array, docs_array)[0]  # shape: (4,)

# Rank documents by similarity
ranked = sorted(zip(similarities, documents), reverse=True)

# Print results
print(f"Query: {query}\n")
print("--- Similarity Scores (ranked) ---")
for score, doc in ranked:
    print(f"{score:.4f} → {doc}")

# Best match
print(f"\n Most similar document: '{ranked[0][1]}'")