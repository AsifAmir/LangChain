from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableBranch, RunnablePassthrough

load_dotenv()


# Define the models
llm=HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="text-generation",
    temperature=0.7,
    provider="auto",
    )

model = ChatHuggingFace(llm=llm)

# Define the prompts
prompt1 = PromptTemplate(
    template='Generate a short explanation about the following topic: {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Summarize the following text \n {text}',
    input_variables=['topic']
)

# Output parser
parser = StrOutputParser()

# explanation_chain generates the explanation for the given topic
# explanation_chain = RunnableSequence(prompt1, model, parser)
explanation_chain = prompt1 | model | parser # using the LCEL syntax for chaining. LCEL means LangChain Expression Language.

# RunnableBranch allows us to branch the execution based on a condition.
branch_chain = RunnableBranch(
    (lambda x: len(x.split())>100, prompt2 | model | parser), # if the explanation is more than 100 words, we summarize it.
    RunnablePassthrough()
)

# Final chain that combines the explanation and branching
final_chain = RunnableSequence(explanation_chain, branch_chain)

# provide the topic to the chain and get the explanation
topic = "Attention Is All You Need"

# Run the chain
result = final_chain.invoke({'topic': topic})
print(result)

# Full chain for graph visualization
# final_chain.get_graph().print_ascii()