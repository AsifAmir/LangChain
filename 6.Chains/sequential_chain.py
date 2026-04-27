from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# prompt templates. These will define the prompts we want to send to the model.
prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

# Define the model
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-1B-Instruct",
    task="text-generation",
    temperature=0.3, # optional, set the temperature for text generation
    provider="auto",  # let Hugging Face choose the best provider for you
    )

model = ChatHuggingFace(llm=llm)

# output parser. This will take the output from the model and convert it to a string.
parser = StrOutputParser()

# chaining the prompts together. The output of one will be the input of the next.
chain = prompt1 | model | parser | prompt2 | model | parser

# run the chain with the initial input. The output of one will be the input of the next.
result = chain.invoke({'topic': 'Artificial Intelligence'})

# print the result
print(result)

# print the graph of the chain. This will show the flow of the data through the chain.
chain.get_graph().print_ascii()