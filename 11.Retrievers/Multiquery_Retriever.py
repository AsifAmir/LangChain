import os
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

# Sample documents
docs = [
    # Targeted Health Docs (The "Gold" results)
    Document(page_content="Optimizing mitochondrial density through zone 2 exercise significantly enhances cellular ATP production and metabolic stamina.", metadata={"topic": "health"}),
    Document(page_content="Stabilizing postprandial glucose spikes prevents the afternoon dip in cognitive function and physical vigor.", metadata={"topic": "health"}),
    Document(page_content="Alignment of the circadian rhythm with natural light exposure regulates cortisol secretion and sleep-wake homeostasis.", metadata={"topic": "health"}),
    
    # Distractor Docs (Sharing keywords like "energy" and "balance" but in wrong context)
    Document(page_content="The energy grid requires a delicate balance of load shedding to prevent total system collapse during peak summer hours.", metadata={"topic": "tech"}),
    Document(page_content="In chemical reactions, activation energy is the minimum energy required to reach the transition state for a balanced equation.", metadata={"topic": "science"}),
    Document(page_content="Accounting balance sheets track the flow of capital energy through liquid assets and long-term liabilities.", metadata={"topic": "finance"}),
    Document(page_content="Kinetic energy is conserved in a perfectly elastic collision between two balanced masses.", metadata={"topic": "physics"}),
]

# Initialize embedding model
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    output_dimensionality=768
)

# Define the llm for the MultiQueryRetriever
# model = ChatHuggingFace(llm=HuggingFaceEndpoint(
#     repo_id="openai/gpt-oss-20b",
#     task="text-generation",
#     temperature=0.7, # optional, set the temperature for text generation
#     provider="auto",  # let Hugging Face choose the best provider for you
#     ))

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite", # specifies the model to use
    temperature=0.3, # controls the randomness of the output, with higher values producing more creative responses
    # max_output_tokens=256 # limits the response to 256 tokens, which is approximately 128-150 words
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

# Create retrievers
similarity_retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

"""
DEFINITION: MultiQueryRetriever
The MultiQueryRetriever automates the process of prompt tuning by using an LLM to generate 
multiple queries from different perspectives for a single user input query. 

How it works:
1. It takes the original user query (e.g., "How to improve energy?").
2. The LLM generates N variations (e.g., "Ways to boost stamina", "Daily habits for vitality").
3. It executes a vector search for EACH variation against the database.
4. It performs a 'Unique Union' of all retrieved documents, effectively overcoming 
   semantic distance limitations where a single phrasing might miss relevant data.
"""

# Create MultiQueryRetriever
multiquery_retriever = MultiQueryRetriever.from_llm(
    retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
    llm=model
)

# Query
query = "I'm always wiped out in the afternoon and feel shaky. How do I fix my battery and stay steady?"
print(f"Query: {query}")

# Retrieve results
similarity_results = similarity_retriever.invoke(query)
multiquery_results= multiquery_retriever.invoke(query)

# Print results
print("\n--- Retriever Results ---")
for i, doc in enumerate(similarity_results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)

print("*"*150)

print("\n--- MultiQueryRetriever Results ---")
for i, doc in enumerate(multiquery_results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)