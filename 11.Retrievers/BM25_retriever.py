from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

# ── Sample documents ──────────────────────────────────────────────────────────
# BM25 works directly on raw text (no embeddings needed),
# so we only need the Document objects, no vector store.
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

docs = [doc1, doc2, doc3, doc4, doc5, doc6, doc7]

# ── BM25 Retriever ────────────────────────────────────────────────────────────
# BM25Retriever.from_documents() tokenises each document internally.
# No embedding model or vector store is required — BM25 is purely
# keyword/term-frequency based (Okapi BM25 ranking algorithm).
#
# k=3 → return the top 3 most relevant documents for any query.
# You can change k later via: bm25_retriever.k = 5
bm25_retriever = BM25Retriever.from_documents(docs, k=3)

# ── Query ─────────────────────────────────────────────────────────────────────
# BM25 scores documents by matching query terms against the corpus.
# Terms like "photosynthesis" get a high IDF (rare across docs = valuable).
# Documents containing it multiple times get boosted TF scores.
query = "Which player is known for his incredible goal-scoring ability and physical presence?"
results = bm25_retriever.invoke(query)

print(f"Query: {query}\n")

# ── Print results ─────────────────────────────────────────────────────────────
# Results are ranked by BM25 score (highest relevance first).
for i, doc in enumerate(results):
    print(f"--- Result {i+1} ({doc.metadata}) ---")
    print(doc.page_content)
    print()