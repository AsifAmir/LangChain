from langchain_text_splitters import RecursiveCharacterTextSplitter,Language

text = """
# Project Name: Smart Student Tracker

A simple Python-based project to manage and track student data, including their grades, age, and academic status.


## Features

- Add new students with relevant info
- View student details
- Check if a student is passing
- Easily extendable class-based design


## 🛠 Tech Stack

- Python 3.10+
- No external dependencies


## Getting Started

1. Clone the repo  
   ```bash
   git clone https://github.com/your-username/student-tracker.git

"""

# Initialize the splitter
splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN,
    chunk_size=200,
    chunk_overlap=0,
)

# Perform the split
chunks = splitter.split_text(text)

# Print the number of chunks and the chunks themselves
print(f'number of chunks: {len(chunks)}')
print(chunks[0])
print(f'chunk length: {len(chunks[0])}')
# chunks visualizer website: https://chunkviz.up.railway.app/