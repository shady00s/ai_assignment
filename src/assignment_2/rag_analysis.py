
import time
from typing import List, Dict
from rag_system import RAGSystem
import os


class RAGAnalyzer:
    def __init__(self):
        self.results = []
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.file_path = os.path.join(self.script_dir, "docs", "chapter3_4.txt")
        self.results_report_path = os.path.join(self.script_dir, "results", "results_report.md")

    def run_chunk_size_experiment(self, chunk_sizes: List[int], test_queries: List[str]):
        """Test different chunk sizes using mxbai-embed-large"""
        for size in chunk_sizes:
            print(f"\nTesting chunk size: {size}")
            rag = RAGSystem(chunk_size=size, embedding_model="mxbai-embed-large")
            chunks = rag.load_and_chunk_text(self.file_path)
            rag.build_index(chunks)
            for query in test_queries:
                result = rag.query(query, k=3)
                self.results.append({
                    "experiment": "chunk_size",
                    "chunk_size": size,
                    "query": query,
                    "retrieved_chunks": result["retrieved_chunks"],
                    "answer": result["answer"]
                })

    def run_k_value_experiment(self, k_values: List[int], test_queries: List[str]):
        """Test different k values with fixed chunk_size=500"""
        rag = RAGSystem(chunk_size=500, embedding_model="mxbai-embed-large")
        chunks = rag.load_and_chunk_text(self.file_path)
        rag.build_index(chunks)
        for k in k_values:
            print(f"\nTesting k={k}")
            for query in test_queries:
                result = rag.query(query, k=k)
                self.results.append({
                    "experiment": "k_value",
                    "k": k,
                    "query": query,
                    "retrieved_chunks": result["retrieved_chunks"],
                    "answer": result["answer"]
                })

    def run_model_comparison(self, test_queries: List[str]):
        """Compare bge-m3 vs mxbai-embed-large"""
        models = ["bge-m3", "mxbai-embed-large"]
        for model in models:
            print(f"\nBuilding RAG system with {model}...")
            rag = RAGSystem(chunk_size=500, embedding_model=model)
            chunks = rag.load_and_chunk_text(result.file_path)
            rag.build_index(chunks)
            for query in test_queries:
                start_time = time.time()
                result = rag.query(query, k=3)
                end_time = time.time()
                self.results.append({
                    "experiment": "model_comparison",
                    "model": model,
                    "query": query,
                    "retrieved_chunks": result["retrieved_chunks"],
                    "answer": result["answer"],
                    "retrieval_time": end_time - start_time
                })

    def generate_report(self):
        """Generate markdown analysis report"""
        with open(self.results_report_path, "w") as f:
            f.write("\n\n## Experimental Results\n")
            for result in self.results:
                f.write(f"\n### Experiment: {result['experiment']}\n")
                for key, value in result.items():
                    if key != 'experiment':
                        f.write(f"* **{key}:** {value}\n")

def main():
    analyzer = RAGAnalyzer()
    test_queries = [
        "What is the difference between instruct models and chat models?",
        "How does temperature affect model outputs?",
        "What is the ChatML format?",
        "Explain the RLHF training process",
        "What are system messages used for?",
    ]

    print("=" * 60)
    print("Experiment 1: Chunk Size Impact")
    print("=" * 60)
    analyzer.run_chunk_size_experiment([300, 500, 1000], test_queries)

    print("\n" + "=" * 60)
    print("Experiment 2: Retrieval Size (k) Impact")
    print("=" * 60)
    analyzer.run_k_value_experiment([1, 3, 5, 10], test_queries)

    print("\n" + "=" * 60)
    print("Experiment 3: Embedding Model Comparison")
    print("=" * 60)
    analyzer.run_model_comparison(test_queries)

    # Generate analysis report
    analyzer.generate_report()
    print("\nAnalysis report generated.  You can find it in" +  analyzer.results_report_path)


if __name__ == "__main__":
    main()
