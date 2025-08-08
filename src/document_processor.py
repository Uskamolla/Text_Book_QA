import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import re

class DocumentProcessor:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF file"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    # Clean the text
                    page_text = self.clean_text(page_text)
                    text += f"\n--- Page {page_num + 1} ---\n" + page_text
                
                return text
        except Exception as e:
            print(f"Error reading PDF: {str(e)}")
            return None
    
    def clean_text(self, text):
        """Clean extracted text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:()\-]', '', text)
        return text.strip()
    
    def create_chunks(self, text):
        """Split text into chunks"""
        if not text:
            return []
        
        # Create documents from text
        docs = [Document(page_content=text)]
        
        # Split into chunks
        chunks = self.text_splitter.split_documents(docs)
        
        # Add metadata to chunks
        for i, chunk in enumerate(chunks):
            chunk.metadata = {
                'chunk_id': i,
                'source': 'textbook.pdf',
                'chunk_size': len(chunk.page_content)
            }
        
        return chunks
    
    def process_pdf(self, pdf_path):
        """Main processing function"""
        print(f"Processing PDF: {pdf_path}")
        
        # Extract text
        text = self.extract_text_from_pdf(pdf_path)
        if not text:
            return []
        
        print(f"Extracted text length: {len(text)} characters")
        
        # Create chunks
        chunks = self.create_chunks(text)
        print(f"Created {len(chunks)} chunks")
        
        return chunks