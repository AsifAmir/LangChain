from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv

load_dotenv()

# Define the models
model = ChatHuggingFace(llm=HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="text-generation",
    temperature=0.7,
    provider="auto",
    ))

# Define the prompt template
prompt = PromptTemplate(
    template='Write a summary for the following text - \n {text}',
    input_variables=['text']
)

# Define the output parser
parser = StrOutputParser()

# Define the document loader
loader = TextLoader('open_source_llm.txt', encoding='utf-8')
docs = loader.load()

# print the type of the loaded documents
print(type(docs))

# print the document in list format including page content and metadata
# print(docs)

# print the number of documents loaded
# print(len(docs))

# print the first document's page content
print(docs[0].page_content)

# print the first document's metadata
print(docs[0].metadata)

# Create the chain
chain = prompt | model | parser

# Invoke the chain with the first document's page content
response = chain.invoke({'text':docs[0].page_content})

# print the response
print(f'Response: {response}')

# Full chain for graph visualization
chain.get_graph().print_ascii()