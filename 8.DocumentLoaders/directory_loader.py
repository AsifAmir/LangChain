from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path='../Books',
    glob='*.pdf',
    loader_cls=PyPDFLoader
)

# different from load() method, lazy_load() method returns a generator that yields documents one at a time, which is more memory efficient for large directories
docs = loader.lazy_load()

# print all the metadata of the loaded documents
for document in docs:
    print(document.metadata)