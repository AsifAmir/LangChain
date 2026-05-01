import os
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

# Sample documents
docs = [
    Document(page_content=(
        """The Grand Canyon is one of the most visited natural wonders in the world.
        Photosynthesis is the process by which green plants convert sunlight into energy.
        Millions of tourists travel to see it every year. The rocks date back millions of years."""
    ), metadata={"source": "Doc1"}),

    Document(page_content=(
        """In medieval Europe, castles were built primarily for defense.
        The chlorophyll in plant cells captures sunlight during photosynthesis.
        Knights wore armor made of metal. Siege weapons were often used to breach castle walls."""
    ), metadata={"source": "Doc2"}),

    Document(page_content=(
        """Basketball was invented by Dr. James Naismith in the late 19th century.
        It was originally played with a soccer ball and peach baskets. NBA is now a global league."""
    ), metadata={"source": "Doc3"}),

    Document(page_content=(
        """The history of cinema began in the late 1800s. Silent films were the earliest form.
        Thomas Edison was among the pioneers. Photosynthesis does not occur in animal cells.
        Modern filmmaking involves complex CGI and sound design."""
    ), metadata={"source": "Doc4"})
]

# Initialize embedding model
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    output_dimensionality=768
)

# Define the llm for the contextual compression retriever
model = ChatHuggingFace(llm=HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="text-generation",
    temperature=0.3, # optional, set the temperature for text generation
    provider="auto",  # let Hugging Face choose the best provider for you
    ))


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
# Wrap the vector store as a LangChain retriever.
# search_kwargs={"k": 5} → fetch the top 5 most similar chunks
# using embedding (cosine) similarity for any incoming query.
base_retriever = vector_store.as_retriever(search_type="similarity" ,search_kwargs={"k": 5})

# Set up the compressor using an LLM
llm = model
compressor = LLMChainExtractor.from_llm(llm)

# Create the contextual compression retriever
# LLMChainExtractor sends each raw chunk + the original query to the LLM
# and asks it: "Extract only the sentences relevant to this query."
# → Irrelevant sentences are stripped out.
# → Chunks with zero relevant content are dropped entirely.
# NOTE: This makes one extra LLM call per retrieved chunk (5 calls for k=5).
compression_retriever = ContextualCompressionRetriever(
    base_retriever=base_retriever,
    base_compressor=compressor
)

# Query the retriever
query = "What is photosynthesis?"
compressed_results = compression_retriever.invoke(query)
print(f"Query: {query}")

# Print results
for i, doc in enumerate(compressed_results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)
