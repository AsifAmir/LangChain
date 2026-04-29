from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    output_dimensionality=768
)

# Initialize the splitter
text_splitter = SemanticChunker(
    embeddings, breakpoint_threshold_type="standard_deviation", # percentile
    breakpoint_threshold_amount=1
)

# Define a sample text
sample = """
The old lighthouse stood at the edge of the rocky shore, its light sweeping across the dark water every few seconds. Fishermen had relied on it for generations, navigating safely through storms and fog. The smell of salt and seawater filled the air around it.

The summer Olympics bring athletes from every corner of the world together in one place. Competitors train for years, pushing their bodies to the limit for a chance at a gold medal. Millions of fans watch from stadiums and living rooms, united by the thrill of sport.

Climate change is one of the most serious challenges facing the world today. Rising temperatures are melting glaciers and causing sea levels to creep higher each year. Governments, scientists, and ordinary citizens must work together to reduce emissions and protect the planet for future generations.
"""
# Split the text into chunks
docs = text_splitter.create_documents([sample])
print(f'number of chunks: {len(docs)}')

# Print each chunk
for i, doc in enumerate(docs):
    print(f'Chunk {i+1}:\n{doc.page_content}\n')