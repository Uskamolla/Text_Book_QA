import chromadb
from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np
import os

# Set cache directory to project folder
os.environ['TRANSFORMERS_CACHE'] = './models_cache'
os.environ['HF_HOME'] = './models_cache'

class VectorStore:
    def __init__(self, collection_name="textbook_chunks", embedding_model="Qwen/Qwen3-Embedding-0.6B"):
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize embedding model
        print(f"Loading embedding model: {embedding_model}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                embedding_model, 
                trust_remote_code=True,
                cache_dir="./models_cache"
            )
            self.embedding_model = AutoModel.from_pretrained(
                embedding_model,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                cache_dir="./models_cache"
            )
            print(f"Embedding model loaded on {self.device}")
        except Exception as e:
            print(f"Error loading embedding model: {str(e)}")
            self.embedding_model = None
            self.tokenizer = None
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path="./chroma_db")
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"Loaded existing collection: {collection_name}")
        except:
            self.collection = self.client.create_collection(name=collection_name)
            print(f"Created new collection: {collection_name}")
    
    def mean_pooling(self, model_output, attention_mask):
        """Mean pooling for sentence embeddings"""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def generate_embeddings(self, texts):
        """Generate embeddings for texts using Qwen embedding model"""
        if not self.embedding_model or not self.tokenizer:
            print("Embedding model not available")
            return []
            
        print(f"Generating embeddings for {len(texts)} texts...")
        embeddings = []
        
        # Process in batches to avoid memory issues
        batch_size = 8
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            try:
                # Tokenize
                encoded_input = self.tokenizer(
                    batch_texts, 
                    padding=True, 
                    truncation=True, 
                    max_length=512,
                    return_tensors='pt'
                )
                
                # Move to device
                if self.device == "cuda":
                    encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
                
                # Generate embeddings
                with torch.no_grad():
                    model_output = self.embedding_model(**encoded_input)
                    batch_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
                    
                    # Normalize embeddings
                    batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
                    
                    # Convert to CPU and list
                    batch_embeddings = batch_embeddings.cpu().numpy().tolist()
                    embeddings.extend(batch_embeddings)
                    
            except Exception as e:
                print(f"Error processing batch {i//batch_size + 1}: {str(e)}")
                # Add zero embeddings for failed batch
                for _ in batch_texts:
                    embeddings.append([0.0] * 768)  # Default embedding size
        
        print(f"Generated {len(embeddings)} embeddings")
        return embeddings
    
    def add_documents(self, chunks):
        """Add document chunks to vector store"""
        if not chunks:
            print("No chunks to add")
            return
        
        # Extract texts and metadata
        texts = [chunk.page_content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        
        # Generate embeddings
        embeddings = self.generate_embeddings(texts)
        
        # Add to collection
        try:
            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            print(f"Successfully added {len(chunks)} chunks to vector store")
        except Exception as e:
            print(f"Error adding documents: {str(e)}")
    
    def similarity_search(self, query, k=5):
        """Search for similar documents"""
        try:
            # Generate query embedding
            query_embeddings = self.generate_embeddings([query])
            if not query_embeddings:
                return []
            
            query_embedding = query_embeddings[0]
            
            # Search in collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results
            search_results = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    search_results.append({
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'similarity': 1 - results['distances'][0][i]  # Convert distance to similarity
                    })
            
            return search_results
        
        except Exception as e:
            print(f"Error in similarity search: {str(e)}")
            return []
    
    def get_collection_info(self):
        """Get information about the collection"""
        try:
            count = self.collection.count()
            return {"document_count": count}
        except:
            return {"document_count": 0}
    
    def clear_collection(self):
        """Clear all documents from collection"""
        try:
            self.collection.delete()
            self.collection = self.client.create_collection(name=self.collection_name)
            print("Collection cleared successfully")
        except Exception as e:
            print(f"Error clearing collection: {str(e)}")