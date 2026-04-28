from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableSequence

load_dotenv()

# Define the models
model = ChatHuggingFace(llm=HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="text-generation",
    temperature=0.7,
    provider="auto",
    ))


# Define the prompts
prompt1 = PromptTemplate(
    template='Generate a short blog post about {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a short LinkedIn post about {topic}',
    input_variables=['topic']
)

# Output parser
parser = StrOutputParser()

# Split into two chains and run in parallel
parallel_chain = RunnableParallel({
    'blog_post': RunnableSequence(prompt1, model, parser),
    'linkedin_post': RunnableSequence(prompt2, model, parser)
})

# provide the topic to the chain and get the explanation and LinkedIn post content
topic = "Ensemble Approach in Machine Learning"

# Run the chain and print the explanation
posts = parallel_chain.invoke({'topic': topic})
# print(posts)

print(f"{topic}\n")
print(f"Blog post: {posts['blog_post']}")
print(f"LinkedIn post: {posts['linkedin_post']}")

# Full chain for graph visualization
parallel_chain.get_graph().print_ascii()