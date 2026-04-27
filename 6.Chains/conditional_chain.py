from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser


load_dotenv()

# Define the models
model = ChatHuggingFace(llm=HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="text-generation",
    temperature=0.7, # optional, set the temperature for text generation
    provider="auto",  # let Hugging Face choose the best provider for you
    ))


class Review(BaseModel):

    sentiment: Literal['positive', 'negative'] = Field(description='Give the sentiment of the feedback')

# output parser. This will take the output from the model and convert it to a pydantic object.
parser2 = PydanticOutputParser(pydantic_object=Review)

template = PromptTemplate(
    template='Given the following feedback, classify the sentiment as positive or negative \n {feedback} \n {format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction':parser2.get_format_instructions()} # this will add the format instructions to the prompt, so that the model knows how to format the output
)

# chaining the prompt, model and parser2.
classifier_chain = template | model | parser2

# result = classifier_chain.invoke({'feedback':'The product is really good and I am satisfied with the quality'}).sentiment
# print(result)


prompt2 = PromptTemplate(
    template='Write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)

# output parser. This will take the output from the model and convert it to a string.
parser = StrOutputParser()

# branching the chain based on the sentiment of the feedback. If the sentiment is positive, it will go to prompt2, if it is negative, it will go to prompt3. If it cannot find the sentiment, it will return a default message.
branch_chain = RunnableBranch(
    (lambda x:x.sentiment == 'positive', prompt2 | model | parser),
    (lambda x:x.sentiment == 'negative', prompt3 | model | parser),
    RunnableLambda(lambda x: "could not find sentiment")
)

chain = classifier_chain | branch_chain

feedback = "The product is really good and I am satisfied with the quality"
# feedback = "The product is really bad and I am not satisfied with the quality"

print(chain.invoke({'feedback': feedback}))

chain.get_graph().print_ascii()