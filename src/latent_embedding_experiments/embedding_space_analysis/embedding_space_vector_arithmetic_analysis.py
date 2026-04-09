import json

import pandas as pd
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoTokenizer

from src.latent_embedding_experiments.algorithms.utils import emit

# --- CONFIGURATION ---
MODEL_ID = "meta-llama/Llama-3.1-405B-Instruct"
DISPLAY_K = 15
LOG_FILE = (
    "src/latent_embedding_experiments/logs/llama_405b_vector_arithmetic_analysis.txt"
)

# Define the equations you want to test: (Positive Concepts) - (Negative Concepts)
# Example 1: King - Man + Woman = Queen
# Example 2: Paris - France + Germany = Berlin
EQUATIONS = [
    {"name": "Gender/Royalty Pivot", "add": [" king", " woman"], "subtract": [" man"]},
    {
        "name": "Capital City Translation",
        "add": [" Paris", " Germany"],
        "subtract": [" France"],
    },
    {
        "name": "Conceptual Progression",
        "add": [" ocean", " sky"],
        "subtract": [" water"],
    },
]


def load_embeddings(model_id):
    """Downloads only the specific safetensors shard containing the embeddings."""
    emit(f"Fetching safetensors index for {model_id}...")
    index_path = hf_hub_download(
        repo_id=model_id, filename="model.safetensors.index.json"
    )
    with open(index_path, "r") as f:
        index_data = json.load(f)

    tensor_name = "model.embed_tokens.weight"
    tensor_file = index_data["weight_map"][tensor_name]

    emit(f"Downloading isolated embedding shard: {tensor_file}...")
    tensor_path = hf_hub_download(repo_id=model_id, filename=tensor_file)
    tensors = load_file(tensor_path)

    return tensors[tensor_name].to(torch.float32)


def get_concept_vector(concept, tokenizer, embeddings):
    """Encodes a word and averages its embeddings if it spans multiple tokens."""
    # We strip spaces visually for the log, but keep them for the tokenizer
    # as standard Llama tokenization often relies on the leading space.
    input_ids = tokenizer(concept, add_special_tokens=False).input_ids
    vectors = embeddings[input_ids]

    # If the concept is multiple tokens, average them into a single point
    if vectors.shape[0] > 1:
        return torch.mean(vectors, dim=0, keepdim=True)
    return vectors


def main():
    emit(f"Initializing tokenizer: {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    raw_embeddings = load_embeddings(MODEL_ID)
    norm_dictionary = F.normalize(raw_embeddings, p=2, dim=1)

    pd.set_option("display.width", 1000)

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        emit("LLAMA 405B VECTOR ARITHMETIC LOG", f)
        emit("Mode: Additive and Subtractive Conceptual Fusion\n", f)

        for eq in EQUATIONS:
            emit(f"\n{'='*80}", f)
            emit(f"EQUATION: {eq['name']}", f)
            emit(f"{'='*80}", f)

            add_concepts = eq.get("add", [])
            sub_concepts = eq.get("subtract", [])

            # Construct the visual string for the log
            add_str = " + ".join([f"'{c}'" for c in add_concepts])
            sub_str = " - ".join([f"'{c}'" for c in sub_concepts])
            equation_str = f"{add_str}" + (f" - {sub_str}" if sub_concepts else "")
            emit(f"Formula: [ {equation_str} ]\n", f)

            # --- SYNTHESIZING THE VECTOR ---
            result_vector = torch.zeros(
                (1, raw_embeddings.shape[1]), dtype=torch.float32
            )

            for concept in add_concepts:
                vec = get_concept_vector(concept, tokenizer, raw_embeddings)
                result_vector += vec

            for concept in sub_concepts:
                vec = get_concept_vector(concept, tokenizer, raw_embeddings)
                result_vector -= vec

            # Normalize the final fused vector
            result_unit = F.normalize(result_vector, p=2, dim=1)

            # --- DICTIONARY SEARCH ---
            cos_sims = torch.matmul(result_unit, norm_dictionary.T).squeeze(0)
            top_vals, top_idxs = torch.topk(cos_sims, DISPLAY_K)
            top_words = tokenizer.batch_decode(top_idxs)

            # Prevent the source words from cluttering the results
            clean_input_words = [
                c.strip().lower() for c in (add_concepts + sub_concepts)
            ]

            # --- RENDER TABLE ---
            header = f"{'Rank':<5} | {'CosSim':<8} | Token"
            emit(header, f)
            emit("-" * 40, f)

            rank_counter = 1
            for i, (word, val) in enumerate(zip(top_words, top_vals)):
                clean_word = word.replace("\n", "\\n").replace("\r", "\\r")
                word_lower = clean_word.strip().lower()

                # Flag words that were part of the input formula
                marker = (
                    "*"
                    if any(
                        src in word_lower or word_lower in src
                        for src in clean_input_words
                    )
                    else " "
                )

                emit(f"{rank_counter:<5} | {val:.4f}   | {marker} '{clean_word}'", f)
                rank_counter += 1

    emit(f"\nProcessing complete. Arithmetic results saved to {LOG_FILE}.")


if __name__ == "__main__":
    main()
