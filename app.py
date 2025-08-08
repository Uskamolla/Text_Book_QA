import streamlit as st
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStore
from src.qa_system import QASystem

# Set page config
st.set_page_config(
    page_title="Textbook RAG System",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'qa_system' not in st.session_state:
    st.session_state.qa_system = None
if 'documents_loaded' not in st.session_state:
    st.session_state.documents_loaded = False

def main():
    st.title("📚 Textbook RAG Question Answering System")
    st.markdown("Ask questions about your textbook and get AI-powered answers!")
    
    # Sidebar for document processing
    with st.sidebar:
        st.header("📄 Document Processing")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload your textbook PDF",
            type=['pdf'],
            help="Upload a PDF file to process"
        )
        
        if uploaded_file is not None:
            # Save uploaded file
            pdf_path = os.path.join("data", "textbook.pdf")
            os.makedirs("data", exist_ok=True)
            
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.success("✅ PDF uploaded successfully!")
            
            # Process document button
            if st.button("🔄 Process Document", type="primary"):
                with st.spinner("Processing document..."):
                    try:
                        # Initialize components
                        processor = DocumentProcessor()
                        vector_store = VectorStore()
                        
                        # Process PDF
                        chunks = processor.process_pdf(pdf_path)
                        
                        if chunks:
                            # Clear existing collection
                            vector_store.clear_collection()
                            
                            # Add chunks to vector store
                            vector_store.add_documents(chunks)
                            
                            # Store in session state
                            st.session_state.vector_store = vector_store
                            st.session_state.documents_loaded = True
                            
                            st.success(f"✅ Document processed! Created {len(chunks)} chunks.")
                        else:
                            st.error("❌ Failed to process document.")
                            
                    except Exception as e:
                        st.error(f"❌ Error processing document: {str(e)}")
        
        # Collection info
        if st.session_state.vector_store:
            info = st.session_state.vector_store.get_collection_info()
            st.info(f"📊 Documents in collection: {info['document_count']}")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Ask Questions")
        
        # Initialize QA system if not done
        if st.session_state.qa_system is None:
            with st.spinner("Loading QA model..."):
                try:
                    st.session_state.qa_system = QASystem()
                    st.success("✅ QA model loaded!")
                except Exception as e:
                    st.error(f"❌ Error loading QA model: {str(e)}")
        
        # Question input
        question = st.text_input(
            "Enter your question:",
            placeholder="e.g., What is the main topic of chapter 3?",
            disabled=not st.session_state.documents_loaded
        )
        
        # Answer button
        if st.button("🔍 Get Answer", type="primary", disabled=not st.session_state.documents_loaded):
            if question.strip():
                with st.spinner("Generating answer..."):
                    try:
                        answer, context_chunks = st.session_state.qa_system.answer_question(
                            question, 
                            st.session_state.vector_store
                        )
                        
                        # Display answer
                        st.subheader("🤖 Answer:")
                        st.write(answer)
                        
                        # Store context for display in sidebar
                        st.session_state.last_context = context_chunks
                        
                    except Exception as e:
                        st.error(f"❌ Error generating answer: {str(e)}")
            else:
                st.warning("⚠️ Please enter a question.")
        
        # Sample questions
        if st.session_state.documents_loaded:
            st.subheader("💡 Try these sample questions:")
            sample_questions = [
                "What are the main concepts covered in this textbook?",
                "Explain the key points from the first chapter",
                "What are the important definitions mentioned?",
                "Summarize the main topics discussed"
            ]
            
            for q in sample_questions:
                if st.button(f"📝 {q}", key=f"sample_{q}"):
                    st.session_state.sample_question = q
                    st.rerun()
    
    with col2:
        st.header("📖 Context Sources")
        
        if hasattr(st.session_state, 'last_context') and st.session_state.last_context:
            st.subheader("🔍 Relevant Text Chunks:")
            
            for i, chunk in enumerate(st.session_state.last_context[:3]):
                with st.expander(f"Chunk {i+1} (Similarity: {chunk['similarity']:.3f})"):
                    st.write(chunk['content'][:300] + "..." if len(chunk['content']) > 300 else chunk['content'])
                    st.caption(f"Chunk ID: {chunk['metadata'].get('chunk_id', 'N/A')}")
        else:
            st.info("💭 Ask a question to see relevant context sources.")
    
    # Instructions
    if not st.session_state.documents_loaded:
        st.info("👆 Please upload and process a PDF document to start asking questions!")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>🚀 Built with Streamlit, LangChain, ChromaDB, and Qwen3-0.6B</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()