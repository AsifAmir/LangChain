import os
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv

load_dotenv() 


llm = HuggingFacePipeline.from_model_id(
    model_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    pipeline_kwargs=dict(
        temperature=0.3,
        max_new_tokens=100 
    )
)

chat_model = ChatHuggingFace(llm=llm)

response = chat_model.invoke("tell me about yourself")
print(response.content)