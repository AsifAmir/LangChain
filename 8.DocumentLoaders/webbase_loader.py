from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

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
    template='Answer the following question \n {question} from the following text - \n {text}',
    input_variables=['question','text']
)

# Define the output parser
parser = StrOutputParser()

# Define the document loader
url = 'https://en.wikipedia.org/wiki/Attention_Is_All_You_Need'
loader = WebBaseLoader(url)
docs = loader.load()


# print(type(docs))
# print(len(docs))
# print(docs[0].page_content)
# print(docs[0].metadata)

# define the chain
chain = prompt | model | parser

# question and text for the chain
question = 'What is the topic of the content that we are talking about?'
text = docs[0].page_content

# invoke the chain
response = chain.invoke({'question':question, 'text':text})

# print the response
print(f'Response: {response}')