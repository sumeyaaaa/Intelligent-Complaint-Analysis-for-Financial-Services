
import pandas as pd
import time
import pandas as pd
from langchain_chunking import batch_chunk_texts
import matplotlib.pyplot as plt

def load_complaints(csv_path: str, sample_size: int = 10000) -> list[str]:
    """
    Loads consumer complaint narratives from a CSV file.

    Args:
        csv_path (str): The path to the CSV file.
        sample_size (int): The number of rows to load.

    Returns:
        list[str]: A list of complaint text narratives.
    """
    print(f"Loading {sample_size} samples from {csv_path}...")
    sample_df = pd.read_csv(csv_path, nrows=sample_size)
    texts = sample_df["Consumer complaint narrative"].dropna().tolist()
    print(f"Loaded {len(texts)} non-empty narratives.")
    return texts


def evaluate_chunking_configs(texts: list[str], chunk_configs: list[dict]) -> tuple[pd.DataFrame, dict]:
    """
    Evaluates different chunking configurations and returns the results.

    Args:
        texts (list[str]): A list of texts to chunk.
        chunk_configs (list[dict]): A list of configurations to test.

    Returns:
        tuple[pd.DataFrame, dict]: A DataFrame with results and a dictionary with chunk examples.
    """
    results = []
    examples = {}

    for config in chunk_configs:
        size = config['chunk_size']
        overlap = config['overlap']

        print(f"Running chunking: size={size}, overlap={overlap}")
        start_time = time.time()
        all_chunks, counts = batch_chunk_texts(texts, chunk_size=size, chunk_overlap=overlap)
        duration = time.time() - start_time

        # Calculate metrics
        avg_chunks = sum(counts) / len(counts)
        chunk_lengths = [len(c) for c in all_chunks]
        avg_chunk_len = sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0
        
        complete_sentences = sum(1 for chunk in all_chunks if chunk.strip().endswith('.'))
        sentence_adherence_pct = (complete_sentences / len(all_chunks)) * 100 if all_chunks else 0

        results.append({
            "chunk_size": size,
            "overlap": overlap,
            "avg_chunks": avg_chunks,
            "avg_chunk_length": avg_chunk_len,
            "total_chunks": len(all_chunks),
            "time_sec": duration,
            "sentence_adherence_%": sentence_adherence_pct
        })

        examples[f"{size}_{overlap}"] = all_chunks[:3]

    return pd.DataFrame(results), examples



def generate_summary_report(results_df: pd.DataFrame, examples: dict):
    """Prints the summary table and example chunks to the console."""
    print("\n--- Chunking Evaluation Summary ---")
    print(results_df.to_string())

    for key, chunk_list in examples.items():
        print(f"\n🔍 Example Chunks for Config {key}:")
        for i, chunk in enumerate(chunk_list):
            print(f"--- Chunk {i+1} ---\n{chunk}\n")

def plot_chunking_results(results_df: pd.DataFrame, save_path: str = None):
    """
    Generates and shows plots for the chunking evaluation results.
    Optionally saves the plots to a file.
    """
    # Plot 1: Average number of chunks
    plt.figure(figsize=(10, 5))
    for overlap in results_df["overlap"].unique():
        subset = results_df[results_df["overlap"] == overlap]
        plt.plot(subset["chunk_size"], subset["avg_chunks"], marker='o', label=f"Overlap {overlap}")
    plt.title("Average Chunks per Complaint vs. Chunk Size")
    plt.xlabel("Chunk Size (characters)")
    plt.ylabel("Average Number of Chunks")
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(f"{save_path}_avg_chunks.png")
    plt.show()

    # Plot 2: Average chunk lengths
    plt.figure(figsize=(10, 5))
    for overlap in results_df["overlap"].unique():
        subset = results_df[results_df["overlap"] == overlap]
        plt.plot(subset["chunk_size"], subset["avg_chunk_length"], marker='o', label=f"Overlap {overlap}")
    plt.title("Average Chunk Length vs. Chunk Size")
    plt.xlabel("Chunk Size (characters)")
    plt.ylabel("Average Chunk Length (characters)")
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(f"{save_path}_avg_length.png")
    plt.show()