from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# prompt template. This will define the prompt we want to send to the model.
prompt = PromptTemplate(
    template='Generate 5 interesting facts about {topic}',
    input_variables=['topic']
)

# Define the model
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    task="text-generation",
    temperature=0.3, # optional, set the temperature for text generation
    provider="auto",  # let Hugging Face choose the best provider for you
    )

model = ChatHuggingFace(llm=llm)

# output parser. This will take the output from the model and convert it to a string.
parser = StrOutputParser()

# chaining the prompt, model and parser together. The output of one will be the input of the next.
chain = prompt | model | parser

result = chain.invoke({'topic':'Artificial Intelligence'})

print(result)

# print the graph of the chain. This will show the flow of the data through the chain.
chain.get_graph().print_ascii()