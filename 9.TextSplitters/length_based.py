from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Load the PDF document
loader = PyPDFLoader('../Books/Building_LLMs_for_Production_-_Louis-Francois_Bouchard.pdf')
# lazy_load() method returns a generator that yields documents one at a time, which is more memory efficient for large files
docs = list(loader.lazy_load()) # Convert the generator to a list to access documents by index
# docs = loader.load() # Load all documents at once (not memory efficient for large files)

# print the content of the second document
# print(docs[1].page_content)

# Initialize the CharacterTextSplitter with specified chunk size and overlap
splitter = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20, #Usually 10%-20% of the chunk size as overlap
    separator=''
)

# Split the documents into chunks
result = splitter.split_documents(docs)

# print the content of the first chunk
print(result[0].page_content)


# chunks visualizer website: https://chunkviz.up.railway.app/