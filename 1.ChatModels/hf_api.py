# pip install -U "langchain[huggingface]"
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    temperature=0.3, # optional, set the temperature for text generation
    max_new_tokens=50, # optional, set the maximum number of tokens to generate
    provider="auto",  # let Hugging Face choose the best provider for you
    )

model = ChatHuggingFace(llm=llm)

response = model.invoke("tell me about yourself")
print(response.content)