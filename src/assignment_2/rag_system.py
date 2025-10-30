
import os
import json
import faiss
import numpy as np
from pathlib import Path
from typing import List, Dict
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI
import requests
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "docs", "chapter3_4.txt")
# Load environment variables from .env file
load_dotenv()

# Get API key, model URL, and model name from environment variables
api_key = os.getenv("OPENAI_API_KEY")
model_url = os.getenv("OPENAI_BASE_URL")

def get_available_models():
    """Fetches the available models from the LiteLLM API."""
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        response = requests.get(f"{model_url}/models", headers=headers)
        response.raise_for_status()  # Raise an exception for bad status codes
        models_data = response.json()
        # The model names are in the 'id' field of each object in the 'data' list
        model_ids = [model['id'] for model in models_data['data']]
        return model_ids
    except requests.exceptions.RequestException as e:
        print(f"Error fetching models: {e}")
        # Fallback to a default list if the API call fails
        return []

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

def get_embedding(text: str, model: str = "mxbai-embed-large") -> List[float]:
    """Generate embedding for input text"""
    response = client.embeddings.create(input=text, model=model)
    return response.data[0].embedding

class RAGSystem:
    def __init__(self, chunk_size: int = 500, embedding_model: str = "mxbai-embed-large", llm_model: str = "mistral-small"):
        """Initialize RAG system"""
        self.chunk_size = chunk_size
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.index = None
        self.chunks = []
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def load_and_chunk_text(self, file_path: str) -> List[Dict]:
        """Load text file and split into chunks"""
        with open(file_path, 'r') as f:
            text = f.read()
        
        tokens = self.tokenizer.encode(text)
        
        chunks = []
        for i in range(0, len(tokens), self.chunk_size):
            chunk_tokens = tokens[i:i + self.chunk_size]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append({
                "chunk_id": len(chunks),
                "source": file_path,
                "text": chunk_text
            })
        self.chunks = chunks
        return chunks

    def build_index(self, chunks: List[Dict]):
        """Create embeddings and build FAISS index"""
        embeddings = [get_embedding(chunk['text'], self.embedding_model) for chunk in chunks]
        
        dimension = len(embeddings[0])
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings))
        
        faiss.write_index(self.index, "faiss_index.bin")
        with open("chunks.json", 'w') as f:
            json.dump(self.chunks, f)


    def load_index(self, index_path: str, chunks_path: str):
        """Load existing FAISS index and chunks"""
        self.index = faiss.read_index(index_path)
        with open(chunks_path, 'r') as f:
            self.chunks = json.load(f)

    def get_top_k_similar(self, query: str, k: int = 3) -> List[Dict]:
        """Retrieve top-k most similar chunks"""
        query_embedding = get_embedding(query, self.embedding_model)
        distances, indices = self.index.search(np.array([query_embedding]), k)
        
        results = []
        for i in range(k):
            chunk_index = indices[0][i]
            results.append({
                "score": distances[0][i],
                "text": self.chunks[chunk_index]['text']
            })
        return results

    def generate_answer(self, query: str, context_chunks: List[Dict]) -> str:
        """Generate answer using retrieved context"""
        context = "\n".join([chunk['text'] for chunk in context_chunks])
        
        system_message = "You are a helpful assistant. Please answer the user's question based on the provided context."
        
        response = client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
            ]
        )
        return response.choices[0].message.content

    def query(self, question: str, k: int = 3) -> Dict:
        """Complete RAG pipeline: retrieve + generate"""
        retrieved_chunks = self.get_top_k_similar(question, k)
        answer = self.generate_answer(question, retrieved_chunks)
        return {
            "retrieved_chunks": retrieved_chunks,
            "answer": answer
        }

def main():
    available_models = get_available_models()
    if not available_models:
        print("No models available. Please check your OPENAI_API_KEY and OPENAI_BASE_URL.")
        return

    print("Available models:")
    for i, model in enumerate(available_models):
        print(f"{i+1}. {model}")

    # Select Embedding Model
    embedding_model_choice = input("Select an embedding model by number (default: mxbai-embed-large): ")
    try:
        embedding_model_idx = int(embedding_model_choice) - 1
        if 0 <= embedding_model_idx < len(available_models):
            embedding_model = available_models[embedding_model_idx]
        else:
            embedding_model = "mxbai-embed-large" # Default
    except ValueError:
        embedding_model = "mxbai-embed-large" # Default
    print(f"Using embedding model: {embedding_model}")

    # Select LLM
    llm_model_choice = input("Select an LLM for generation by number (default: mistral-small): ")
    try:
        llm_model_idx = int(llm_model_choice) - 1
        if 0 <= llm_model_idx < len(available_models):
            llm_model = available_models[llm_model_idx]
        else:
            llm_model = "mistral-small" # Default
    except ValueError:
        llm_model = "mistral-small" # Default
    print(f"Using LLM for generation: {llm_model}")

    # Initialize RAG system
    rag = RAGSystem(chunk_size=500, embedding_model=embedding_model, llm_model=llm_model)

    # Load and process documents
    chunks = rag.load_and_chunk_text(file_path)
    print(f"Created {len(chunks)} chunks")

    # Build index
    rag.build_index(chunks)

    # Test queries
    test_queries = [
        "What is the difference between instruct models and chat models?",
        "How does temperature affect model outputs?",
        "What is the ChatML format?",
    ]

    while True:
        print("\nSelect an option:")
        print("1. Ask a new question")
        print("2. Select a question from a list")
        print("3. Exit")
        choice = input("Enter your choice: ")

        if choice == '1':
            query = input("Enter your question: ")
        elif choice == '2':
            print("\nSelect a question:")
            for i, q in enumerate(test_queries):
                print(f"{i+1}. {q}")
            q_choice = input("Enter your choice: ")
            try:
                q_idx = int(q_choice) - 1
                if 0 <= q_idx < len(test_queries):
                    query = test_queries[q_idx]
                else:
                    print("Invalid choice.")
                    continue
            except ValueError:
                print("Invalid choice.")
                continue
        elif choice == '3':
            break
        else:
            print("Invalid choice.")
            continue

        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        result = rag.query(query, k=3)
        print("\nRetrieved chunks:")
        for i, chunk in enumerate(result['retrieved_chunks'], 1):
            print(f"\n[{i}) Similarity: {chunk['score']:.4f}")
            print(f"Text: {chunk['text'][:200]}...")
        print(f"\nAnswer:\n{result['answer']}")

if __name__ == "__main__":
    main()
