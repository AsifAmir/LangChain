from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite", # specifies the model to use
    temperature=0.3, # controls the randomness of the output, with higher values producing more creative responses
    max_output_tokens=256 # limits the response to 256 tokens, which is approximately 128-150 words
    )


# schema
# TypedDict is a way to define the structure of a dictionary in Python, it allows you to specify the keys and their corresponding value types.
class Review(TypedDict):

    key_themes: Annotated[list[str], "Write down all the key themes discussed in the review in a list"]
    summary: Annotated[str, "A brief summary of the review"]
    sentiment: Annotated[Literal["pos", "neg"], "Return sentiment of the review either negative, positive or neutral"]
    pros: Annotated[Optional[list[str]], "Write down all the pros inside a list"]
    cons: Annotated[Optional[list[str]], "Write down all the cons inside a list"]
    name: Annotated[Optional[str], "Write the name of the reviewer"]
    

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""I recently picked up the Lenovo IdeaPad Slim 3i for college, and honestly, it has been a solid companion for everyday tasks. The Intel Core i5 processor handles browsing, document editing, and light multitasking without breaking a sweat. Boot times are impressively fast thanks to the SSD, and the 8GB RAM keeps things smooth during my typical workflow.

The 15.6-inch Full HD display is decent for the price — colors are reasonably accurate and text is sharp enough for long study sessions. The slim form factor and lightweight build make it easy to carry around campus all day without feeling the strain. Battery life comfortably gets me through 6 to 7 hours of mixed use, which is enough for a full day of classes.

The keyboard has a comfortable key travel and is pleasant to type on for long periods. However, the trackpad feels a little plasticky and could be more responsive at times. The speaker quality is mediocre at best — fine for video calls, but do not expect anything impressive for music or movies.

Where it starts to show its budget nature is in the build quality. The plastic chassis flexes slightly under pressure and does not feel as premium as you might want. The port selection is adequate — a couple of USB-A ports, one USB-C, and an HDMI — but the lack of a Thunderbolt port is noticeable if you work with external displays or fast storage. Graphics are purely integrated, so do not even think about gaming beyond casual browser games.

Pros:
Lightweight and portable design, great for students
Fast SSD boot times and snappy everyday performance
Comfortable keyboard for extended typing sessions
Solid battery life for a full day of light use

Cons:
Plastic build feels budget-grade under pressure
Mediocre speakers and average trackpad responsiveness
No dedicated GPU — not suitable for gaming or video editing
Display brightness could be better in outdoor conditions

Review by Asif Amir
""")

# print(f"Result: {result}") # Output: A dictionary containing the structured output based on the defined schema

print(f"Summary: {result.get('summary')}") # Output: A brief summary of the review
print(f"Name: {result.get('name')}") # Output: Asif Amir
print(f"Sentiment: {result.get('sentiment')}") # Output: pos
print(f"Key Themes: {result.get('key_themes')}") # Output: ['Performance', 'Display', 'Portability', 'Battery Life', 'Build Quality', 'Input Devices', 'Audio Quality', 'Graphics Performance']
print(f"Pros: {result.get('pros')}") # Output: ['Lightweight and portable design, great for students', 'Fast SSD boot times and snappy everyday performance', 'Comfortable keyboard for extended typing sessions', 'Solid battery life for a full day of light use']
print(f"Cons: {result.get('cons')}") # Output: ['Plastic build feels budget-grade under pressure', 'Mediocre speakers and average trackpad responsiveness', 'No dedicated GPU — not suitable for gaming or video editing', 'Display brightness could be better in outdoor conditions']
