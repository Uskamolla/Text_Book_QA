from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

# Set cache directory to project folder
os.environ['TRANSFORMERS_CACHE'] = './models_cache'
os.environ['HF_HOME'] = './models_cache'

class QASystem:
    def __init__(self, model_name="Qwen/Qwen3-0.6B"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load tokenizer and model
        print(f"Loading model: {model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                cache_dir="./models_cache"
            )
            
            # Set pad token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                cache_dir="./models_cache"
            )
            
            # Set pad token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print("Model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            self.model = None
            self.tokenizer = None
    
    def create_chat_messages(self, question, context_chunks):
        """Create chat messages using Qwen's format"""
        # Combine top context chunks
        context = "\n\n".join([chunk['content'][:400] for chunk in context_chunks[:2]])
        
        messages = [
            {
                "role": "system", 
                "content": "You are a helpful assistant that answers questions based on textbook content. Provide clear, concise, and accurate answers."
            },
            {
                "role": "user", 
                "content": f"""Based on the following context from a textbook, please answer the question:

Context:
{context}

Question: {question}

Please provide a direct and informative answer based on the context above."""
            }
        ]
        
        return messages
    
    def create_prompt(self, question, context_chunks):
        """Fallback method for compatibility"""
        messages = self.create_chat_messages(question, context_chunks)
        # Convert to simple prompt format for fallback
        context = "\n\n".join([chunk['content'][:400] for chunk in context_chunks[:2]])
        return f"""Based on the following context, please answer the question:

Context: {context}

Question: {question}

Answer:"""
    
    def generate_answer(self, question, context_chunks, max_length=200):
        """Generate answer using the model"""
        if not self.model or not self.tokenizer:
            return "Model not available. Please check the model loading."
        
        try:
            # Create prompt
            prompt = self.create_prompt(question, context_chunks)
            
            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
                padding=True
            )
            
            if self.device == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # Decode response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the answer part
            answer_start = full_response.find("Answer:") + len("Answer:")
            answer = full_response[answer_start:].strip()
            
            # Clean up the answer
            answer = answer.split("\n")[0].strip()  # Take first line only
            
            return answer if answer else "I couldn't generate a proper answer."
            
        except Exception as e:
            print(f"Error generating answer: {str(e)}")
            return f"Error generating answer: {str(e)}"
    
    def answer_question(self, question, vector_store, k=5):
        """Main function to answer a question"""
        # Search for relevant chunks
        context_chunks = vector_store.similarity_search(question, k=k)
        
        if not context_chunks:
            return "No relevant information found in the textbook.", []
        
        # Generate answer
        answer = self.generate_answer(question, context_chunks)
        
        return answer, context_chunks