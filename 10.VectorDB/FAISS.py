import os
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Create LangChain documents for football players
doc1 = Document(
    page_content="Erling Haaland is a Norwegian striker playing for Manchester City in the English Premier League. Known for his incredible goal-scoring ability and physical presence, he broke the Premier League single-season goals record in his debut season.",
    metadata={"team": "Manchester City", "league": "Premier League"}
)
doc2 = Document(
    page_content="Kylian Mbappe is a French forward playing for Real Madrid in La Liga. Renowned for his blistering pace and clinical finishing, he is widely regarded as one of the best players in the world.",
    metadata={"team": "Real Madrid", "league": "La Liga"}
)
doc3 = Document(
    page_content="Julian Alvarez is an Argentine forward playing for Atletico Madrid in La Liga. Known for his work rate, intelligent movement, and clinical finishing, he won the FIFA World Cup with Argentina in 2022.",
    metadata={"team": "Atletico Madrid", "league": "La Liga"}
)
doc4 = Document(
    page_content="Harry Kane is an English striker playing for Bayern Munich in the Bundesliga. Known for his intelligent movement, hold-up play, and prolific goal scoring, he is one of the most complete forwards in Europe.",
    metadata={"team": "Bayern Munich", "league": "Bundesliga"}
)
doc5 = Document(
    page_content="Lamine Yamal is a Spanish winger playing for FC Barcelona in La Liga. At just 17 years old, he has already established himself as one of the most exciting young talents in world football.",
    metadata={"team": "FC Barcelona", "league": "La Liga"}
)
doc6 = Document(
    page_content="Nicolo Barella is an Italian midfielder playing for Inter Milan in Serie A. Known for his box-to-box energy, passing range, and never-say-die attitude, he is the engine of both Inter Milan and the Italian national team.",
    metadata={"team": "Inter Milan", "league": "Serie A"}
)
doc7 = Document(
    page_content="Khvicha Kvaratskhelia is a Georgian winger playing for Paris Saint-Germain in Ligue 1. Formerly at Napoli in Serie A, he is known for his unpredictable dribbling, creativity, and ability to unlock defenses with ease.",
    metadata={"team": "Paris Saint-Germain", "league": "Ligue 1"}
)
doc8 = Document(
    page_content="Florian Wirtz is a German attacking midfielder playing for Bayer Leverkusen in the Bundesliga. Known for his creativity, dribbling, and match-winning moments, he was a key part of Leverkusen's historic unbeaten title winning season.",
    metadata={"team": "Bayer Leverkusen", "league": "Bundesliga"}
)
doc9 = Document(
    page_content="Vinicius Junior is a Brazilian winger playing for Real Madrid in La Liga. Known for his dribbling skills, flair, and match-winning performances, he won the Ballon d'Or in 2024.",
    metadata={"team": "Real Madrid", "league": "La Liga"}
)

docs = [doc1, doc2, doc3, doc4, doc5, doc6, doc7, doc8, doc9]

# Initialize embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    output_dimensionality=768
)

# delete folder if exists to start fresh
if os.path.exists("my_faiss_db"):
    import shutil
    shutil.rmtree("my_faiss_db")

# Initialize FAISS vector store
vector_store = FAISS.from_documents(docs, embeddings)

# save to disk
vector_store.save_local("my_faiss_db")

# display all documents
print("--- All Documents ---")
for i, (doc_id, doc) in enumerate(vector_store.docstore._dict.items()):
    print(f"Document {i+1}  ID: {doc_id}")
    print(f"Content  : {doc.page_content}")
    print(f"Metadata : {doc.metadata}")
    print("-" * 60)

# query
query = "Which player is known for his incredible goal-scoring ability and physical presence?"

# similarity search
results = vector_store.similarity_search(query, k=3)
print("\n--- Similarity Search Results ---")
for i, result in enumerate(results):
    print(f"Result {i}: {result.page_content}...")

# search with similarity score
results_with_scores = vector_store.similarity_search_with_score(query, k=3)
print("\n--- Similarity Search with Scores ---")
for i, (result, score) in enumerate(results_with_scores):
    print(f"Result {i}: {result.page_content}... (Score: {score:.4f})")

# search with metadata filter
print("\n--- Similarity Search with Metadata Filter ---")
results_with_filter = vector_store.similarity_search(
    query, k=3, filter={"league": "La Liga"}
)
for i, result in enumerate(results_with_filter):
    print(f"Result {i}: {result.page_content}... (League: {result.metadata['league']})")

# update doc8 — FAISS has no update, so delete + re-add
print("\nUpdating doc8...")
all_ids = list(vector_store.docstore._dict.keys())
doc8_id = all_ids[7]
vector_store.delete([doc8_id])

updated_doc8 = Document(
    page_content="Florian Wirtz is a German attacking midfielder playing for Liverpool in the Premier League. Known for his creativity, dribbling, and match-winning moments, he was a key part of Leverkusen's historic unbeaten title winning season before his move to Liverpool.",
    metadata={"team": "Liverpool", "league": "Premier League"}
)
vector_store.add_documents([updated_doc8])
print("Document updated successfully!")

# delete doc9
print("\nDeleting doc9...")
all_ids = list(vector_store.docstore._dict.keys())
doc9_id = all_ids[8]
vector_store.delete([doc9_id])
print("Document deleted successfully!")

# verify final state
print("\n--- Final Documents ---")
for i, (doc_id, doc) in enumerate(vector_store.docstore._dict.items()):
    print(f"Document {i+1}")
    print(f"Content  : {doc.page_content}")
    print(f"Metadata : {doc.metadata}")
    print("-" * 60)