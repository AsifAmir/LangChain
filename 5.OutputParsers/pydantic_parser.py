from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

# Define the model
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    task="text-generation",
    # temperature=0.3, # optional, set the temperature for text generation
    provider="auto",  # let Hugging Face choose the best provider for you
    )

model = ChatHuggingFace(llm=llm)

# pydantic model. This will define the schema for the output we want from the model.
class Person(BaseModel):

    name: str = Field(description='Name of the person')
    age: int = Field(gt=18, description='Age of the person')
    city: str = Field(description='Name of the city the person belongs to')


# output parser. This will take the output from the model and convert it to a pydantic object.
parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template='Generate the name, age and city of a fictional {place} person \n {format_instruction}',
    input_variables=['place'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

chain = template | model | parser

final_result = chain.invoke({'place':'Spanish'})

print(final_result)

# pros: pydantic output parser enforces schema, so the output will always be in the correct format.