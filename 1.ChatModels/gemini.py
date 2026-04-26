# pip install -U "langchain[google-genai]"
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv
load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite", # specifies the model to use, in this case, a lightweight version of Gemini 2.5
    temperature=1.2, # controls the randomness of the output, with higher values producing more creative responses
    max_output_tokens=50 # limits the response to 50 tokens, which is approximately 25-30 words
    )

response = model.invoke("Write a short poem about the football team Barcelona.") # sends a prompt to the model and stores the response in the variable 'response'

# print(response) # prints the full response object, which includes metadata and content

print(response.content) # prints just the content of the response, which is the generated text


# Use case of temperature
# factual response: math, code, facts - use low temperature (0.0-0.3)
# balanced response: general questions, explanations - use medium temperature (0.5-0.7)
# creative response: stories, jokes, brainstorming - use high temperature (0.9-1.2)
# maximum randomness: use temperature above 1.2, but be cautious as it may produce incoherent responses