from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda

load_dotenv()

# RunnablePassthrough passes the input as output without any modification.
# passthrough_chain = RunnablePassthrough()
# print(passthrough_chain.invoke({'Name': 'Darth'}))

# Define the models
model = ChatHuggingFace(llm=HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="text-generation",
    temperature=0.7,
    provider="auto",
    ))

# Define the prompts
prompt = PromptTemplate(
    template='Generate a proper explanation about {topic} in 2-3 sentences.',
    input_variables=['topic']
)


# Output parser
parser = StrOutputParser()

# explanation_chain generates the explanation for the given topic
explanation_chain = RunnableSequence(prompt, model, parser)

# Define a function for word count
def word_count(text):
    return len(text.split())

# Split into two chains and run in parallel
# runnable_lambda allows us to use any function as a runnable.
parallel_chain = RunnableParallel({
    'explanation': RunnablePassthrough(),  # explanation_chain,
    'word_count': RunnableLambda(word_count),  # using the the word_count function
    # 'word_count': RunnableLambda(lambda x: len(x.split()))  # direct approach without defining a separate function
    
})

final_chain = RunnableSequence(explanation_chain, parallel_chain)

# provide the topic to the chain and get the explanation and word count
topic = "Ensemble Approach in Machine Learning"

# Run the chain
result = final_chain.invoke({'topic': topic})
print(result)
print(f"Explanation: {result['explanation']}")
print(f"Word Count: {result['word_count']}")

# Full chain for graph visualization
final_chain.get_graph().print_ascii()