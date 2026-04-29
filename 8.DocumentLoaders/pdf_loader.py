from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('../Books/Building_LLMs_for_Production_-_Louis-Francois_Bouchard.pdf')

docs = loader.load()

print(f'Type of loaded documents: {type(docs)}')

# print(docs)

print(f'Number of documents loaded: {len(docs)}')

print(docs[1].page_content)
print(docs[1].metadata)