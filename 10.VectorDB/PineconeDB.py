import time
import os
from pinecone import Pinecone, ServerlessSpec
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Create LangChain documents for football players
doc1 = Document(
    page_content="Erling Haaland is a Norwegian striker playing for Manchester City in the English Premier League. Known for his incredible goal-scoring ability and physical presence, he broke the Premier League single-season goals record in his debut season.", 
    metadata={"id": "doc1", "team": "Manchester City", "league": "Premier League"}
    )
doc2 = Document(
    page_content="Kylian Mbappe is a French forward playing for Real Madrid in La Liga. Renowned for his blistering pace and clinical finishing, he is widely regarded as one of the best players in the world.", 
    metadata={"id": "doc2", "team": "Real Madrid", "league": "La Liga"}
    )
doc3 = Document(
    page_content="Julian Alvarez is an Argentine forward playing for Atletico Madrid in La Liga. Known for his work rate, intelligent movement, and clinical finishing, he won the FIFA World Cup with Argentina in 2022.", 
    metadata={"id": "doc3", "team": "Atletico Madrid", "league": "La Liga"}
    )
doc4 = Document(
    page_content="Harry Kane is an English striker playing for Bayern Munich in the Bundesliga. Known for his intelligent movement, hold-up play, and prolific goal scoring, he is one of the most complete forwards in Europe.", 
    metadata={"id": "doc4", "team": "Bayern Munich", "league": "Bundesliga"}
    )
doc5 = Document(
    page_content="Lamine Yamal is a Spanish winger playing for FC Barcelona in La Liga. At just 17 years old, he has already established himself as one of the most exciting young talents in world football.", 
    metadata={"id": "doc5", "team": "FC Barcelona", "league": "La Liga"}
    )
doc6 = Document(
    page_content="Nicolo Barella is an Italian midfielder playing for Inter Milan in Serie A. Known for his box-to-box energy, passing range, and never-say-die attitude, he is the engine of both Inter Milan and the Italian national team.", 
    metadata={"id": "doc6", "team": "Inter Milan", "league": "Serie A"}
    )
doc7 = Document(
    page_content="Khvicha Kvaratskhelia is a Georgian winger playing for Paris Saint-Germain in Ligue 1. Formerly at Napoli in Serie A, he is known for his unpredictable dribbling, creativity, and ability to unlock defenses with ease.", 
    metadata={"id": "doc7", "team": "Paris Saint-Germain", "league": "Ligue 1"}
    )
doc8 = Document(
    page_content="Florian Wirtz is a German attacking midfielder playing for Bayer Leverkusen in the Bundesliga. Known for his creativity, dribbling, and match-winning moments, he was a key part of Leverkusen's historic unbeaten title winning season.",
    metadata={"id": "doc8", "team": "Bayer Leverkusen", "league": "Bundesliga"}
    )
doc9 = Document(
    page_content="Vinicius Junior is a Brazilian winger playing for Real Madrid in La Liga. Known for his dribbling skills, flair, and match-winning performances, he won the Ballon d'Or in 2024.",
    metadata={"id": "doc9", "team": "Real Madrid", "league": "La Liga"}
    )

docs = [doc1, doc2, doc3, doc4, doc5, doc6, doc7, doc8, doc9]

# Initialize embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    output_dimensionality=768
)

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "football-players"

# delete index if exists to start fresh
if index_name in pc.list_indexes().names():
    pc.delete_index(index_name)
    print("Existing index deleted!")

# create new index
pc.create_index(
    name=index_name,
    dimension=768,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1")
)

# wait for index to be ready
while not pc.describe_index(index_name).status["ready"]:
    time.sleep(1)
print("Index created successfully!")

index = pc.Index(index_name)

# embed and upsert using LangChain documents
vectors = []
for doc in docs:
    embedding = embeddings.embed_query(doc.page_content)
    vectors.append({
        "id": doc.metadata["id"],
        "values": embedding,
        "metadata": {"text": doc.page_content, **doc.metadata}
    })

index.upsert(vectors=vectors)
print(f"Upserted {len(vectors)} documents!")
time.sleep(2)

# display all documents
print("\n--- All Documents ---")
for doc in docs:
    result = index.fetch(ids=[doc.metadata["id"]])
    vector = result["vectors"][doc.metadata["id"]]
    print(f"ID       : {doc.metadata['id']}")
    print(f"Content  : {vector['metadata']['text']}")
    print(f"Metadata : team={vector['metadata']['team']}, league={vector['metadata']['league']}")
    print("-" * 60)

# query
query = "Which player is known for his incredible goal-scoring ability and physical presence?"
query_embedding = embeddings.embed_query(query)

# similarity search
print("\n--- Similarity Search Results ---")
results = index.query(vector=query_embedding, top_k=3, include_metadata=True)
for i, match in enumerate(results["matches"]):
    print(f"Result {i}: {match['metadata']['text']}...")
    print(f"Score    : {match['score']:.4f}")

# search with metadata filter
print("\n--- Similarity Search with Metadata Filter (La Liga) ---")
results_filtered = index.query(
    vector=query_embedding,
    top_k=3,
    include_metadata=True,
    filter={"league": {"$eq": "La Liga"}}
)
for i, match in enumerate(results_filtered["matches"]):
    print(f"Result {i}: {match['metadata']['text']}...")
    print(f"League   : {match['metadata']['league']}")

# update doc8 using updated LangChain document
print("\nUpdating doc8...")
updated_doc8 = Document(
    page_content="Florian Wirtz is a German attacking midfielder playing for Liverpool in the Premier League. Known for his creativity, dribbling, and match-winning moments, he was a key part of Leverkusen's historic unbeaten title winning season before his move to Liverpool.",
    metadata={"id": "doc8", "team": "Liverpool", "league": "Premier League"}
)
updated_embedding = embeddings.embed_query(updated_doc8.page_content)
index.upsert(vectors=[{
    "id": updated_doc8.metadata["id"],
    "values": updated_embedding,
    "metadata": {"text": updated_doc8.page_content, **updated_doc8.metadata}
}])
print("Document updated successfully!")
time.sleep(2)

# delete doc9
print("\nDeleting doc9...")
index.delete(ids=[doc9.metadata["id"]])
print("Document deleted successfully!")
time.sleep(2)

# verify final state
print("\n--- Final Documents ---")
all_ids = [doc.metadata["id"] for doc in docs]
final = index.fetch(ids=all_ids)
for i, (doc_id, vector) in enumerate(final["vectors"].items()):
    print(f"Document {i+1}")
    print(f"Content  : {vector['metadata']['text']}")
    print(f"Metadata : team={vector['metadata']['team']}, league={vector['metadata']['league']}")
    print("-" * 60)