from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableSequence, RunnablePassthrough

load_dotenv()

# RunnablePassthrough passes the input as output without any modification.
# passthrough_chain = RunnablePassthrough()
# print(passthrough_chain.invoke({'Name': 'Darth'}))

# Define the models
model1 = ChatHuggingFace(llm=HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="text-generation",
    temperature=0.7,
    provider="auto",
    ))

model2 = ChatHuggingFace(llm=HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    temperature=0.7,
    provider="auto",
    ))

# Define the prompts
prompt1 = PromptTemplate(
    template='Generate a proper explanation about {topic} in 2-3 sentences.',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a short LinkedIn post content based on the following explanation: {text}',
    input_variables=['text']
)

parser = StrOutputParser()

# Split into two chains
explanation_chain = RunnableSequence(prompt1, model1, parser)

parallel_chain = RunnableParallel({
    'explanation': RunnablePassthrough(),  # explanation_chain,
    'linkedin_post': RunnableSequence(prompt2, model2, parser)
})

final_chain = RunnableSequence(explanation_chain, parallel_chain)

# provide the topic to the chain and get the explanation and LinkedIn post content
topic = "Ensemble Approach in Machine Learning"

# Run the chain and print the explanation
result = final_chain.invoke({'topic': topic})
print(result)
print(f"Explanation: {result['explanation']}")
print(f"LinkedIn Post: {result['linkedin_post']}")

# Full chain for graph visualization
final_chain.get_graph().print_ascii()