# Textbook Q&A System

Ask questions about any PDF textbook and get AI answers. Built for the CASML Hackathon.

## What it does

- Upload a PDF textbook
- Ask questions about the content
- Get AI-powered answers with source references
- Works with tables, figures, and regular text

## How it works

1. **PDF Processing**: Breaks down your textbook into small chunks
2. **Smart Search**: Finds relevant parts for your question
3. **AI Answer**: Uses Qwen3 model to generate answers
4. **Show Sources**: Displays which parts of the book were used

## Tech used

- **AI Model**: Qwen3-0.6B (fast and efficient)
- **Search**: ChromaDB vector database
- **Interface**: Streamlit web app
- **PDF Reader**: pdfplumber (handles tables well)

## Setup

1. **Clone the repo**
```bash
git clone https://github.com/Uskamolla/Text_Book_QA.git
cd Text_Book_QA
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the app**
```bash
streamlit run app.py
```

4. **Use it**
   - Upload your PDF
   - Click "Process Document" 
   - Ask questions!

## Example questions

- "What are the main topics in chapter 1?"
- "Explain the key concepts"
- "What does Table 1 show?"
- "Summarize the conclusions"



## Requirements

- Python 3.8+
- 4GB RAM minimum
- Any PDF textbook

## How big are the models?

- First download: ~1.2GB (happens once)
- After that: instant startup from cache

## Project structure

```
Text_Book_QA/
├── app.py                 # Main web app
├── src/
│   ├── document_processor.py  # PDF handling
│   ├── vector_store.py       # Search system
│   └── qa_system.py          # AI model
├── requirements.txt          # Dependencies
└── README.md                # This file
```

## Made for

- Students studying from textbooks
- Researchers reading papers
- Anyone who wants to chat with their documents

## Note

- First run takes 10-15 minutes to process a book
- After that, questions are answered in seconds
- All data stays on your computer