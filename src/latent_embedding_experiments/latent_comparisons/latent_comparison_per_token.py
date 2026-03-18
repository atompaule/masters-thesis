import json

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoTokenizer

from src.latent_embedding_experiments.algorithms.chimera import forge_mass_equalized_chimera
from src.latent_embedding_experiments.algorithms.solver import fast_geometric_solver

LOG_FILE = "src/latent_embedding_experiments/logs/llama_405b_latent_comparison_per_token.txt"
MODEL_ID = "meta-llama/Llama-3.1-405B-Instruct"
TEMPERATURE = 1.0


def load_embeddings(model_id):
    index_path = hf_hub_download(
        repo_id=model_id, filename="model.safetensors.index.json"
    )
    with open(index_path, "r") as f:
        index_data = json.load(f)

    tensor_name = "model.embed_tokens.weight"
    tensor_file = index_data["weight_map"][tensor_name]

    tensor_path = hf_hub_download(repo_id=model_id, filename=tensor_file)
    tensors = load_file(tensor_path)

    return tensors[tensor_name].to(torch.float32)


def compute_probabilities(logits, temperature):
    probs = F.softmax(torch.tensor(logits, dtype=torch.float32), dim=0)
    adjusted = probs ** (1.0 / temperature)
    adjusted = adjusted / adjusted.sum()
    return probs, adjusted


def build_candidate_vectors(embeddings, probs, adj_probs):
    avg_norm = torch.norm(embeddings, dim=1).mean().item()

    # Weighted embedding
    weighted = torch.sum(embeddings * probs.unsqueeze(1), dim=0, keepdim=True)
    weighted = F.normalize(weighted, dim=1) * avg_norm

    # Centroid
    centroid = torch.mean(embeddings, dim=0, keepdim=True)
    centroid = F.normalize(centroid, dim=1) * avg_norm

    # Top candidate
    top = embeddings[0:1]

    return weighted, centroid, top, avg_norm


def evaluate_vector(vector, normalized_dict, tokenizer, top_k_words, file, display_k):
    vector = F.normalize(vector, dim=1)
    sims = vector @ normalized_dict.T

    vals, idxs = torch.topk(sims.squeeze(0), display_k)
    words = tokenizer.batch_decode(idxs)

    file.write(f"{'Rank':<5} | {'CosSim':<8} | Token\n")
    file.write("-" * 40 + "\n")

    for i, (w, v) in enumerate(zip(words, vals)):
        marker = "*" if w in top_k_words else " "
        file.write(f"{i+1:<5} | {v:.4f}   | {marker} '{w}'\n")


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    raw_embeddings = load_embeddings(MODEL_ID)

    norm_dict = F.normalize(raw_embeddings, dim=1)
    dict_mean = norm_dict.mean(dim=0)
    dict_std = norm_dict.std(dim=0) + 1e-9

    dataset = load_dataset("arcee-ai/LLama-405B-Logits", split="train", streaming=True)
    sequence = next(iter(dataset))

    input_ids = np.array(sequence["input_ids"])
    token_ids = np.array(sequence["token_ids"])
    logits = np.array(sequence["top_values"])

    start, end = 500, 550
    k = 10
    display_k = k * 2  # Defines the expanded output for the tables

    with open(LOG_FILE, "w", encoding="utf-8") as f:

        context = tokenizer.decode(input_ids[: start + 1])

        for t in range(start, end):
            actual_token = tokenizer.decode([input_ids[t + 1]])
            context += actual_token

            f.write(f"\nStep {t}\n")
            f.write(f"Context: ...{context[-150:]}\n")
            f.write(f"Actual: {actual_token}\n\n")

            top_k_ids = token_ids[t][:k]
            top_k_logits = logits[t][:k]
            top_k_embs = raw_embeddings[top_k_ids]
            top_k_words = tokenizer.batch_decode(top_k_ids)

            probs, adj_probs = compute_probabilities(top_k_logits, TEMPERATURE)

            # Similarity between candidates (strictly K x K)
            norm_candidates = norm_dict[top_k_ids]
            sim_matrix = norm_candidates @ norm_candidates.T

            df = pd.DataFrame(
                sim_matrix.numpy(), index=top_k_words, columns=top_k_words
            )
            f.write(df.to_string() + "\n\n")

            weighted, centroid, top, avg_norm = build_candidate_vectors(
                top_k_embs, probs, adj_probs
            )

            chimera = forge_mass_equalized_chimera(
                top_k_ids, norm_dict, dict_mean, dict_std, avg_norm, adj_probs
            )

            temp_weighted = torch.sum(
                top_k_embs * adj_probs.unsqueeze(1), dim=0, keepdim=True
            )
            temp_weighted = F.normalize(temp_weighted, dim=1)

            solver = fast_geometric_solver(
                norm_candidates,
                top_k_ids,
                norm_dict,
                temp_weighted,
                avg_norm,
                adj_probs,
            )

            f.write("\nWeighted embedding:\n")
            evaluate_vector(weighted, norm_dict, tokenizer, top_k_words, f, display_k)

            f.write("\nCentroid:\n")
            evaluate_vector(centroid, norm_dict, tokenizer, top_k_words, f, display_k)

            f.write("\nTop token:\n")
            evaluate_vector(top, norm_dict, tokenizer, top_k_words, f, display_k)

            f.write("\nChimera:\n")
            evaluate_vector(chimera, norm_dict, tokenizer, top_k_words, f, display_k)

            f.write("\nSolver:\n")
            evaluate_vector(solver, norm_dict, tokenizer, top_k_words, f, display_k)

    print(f"Done. Results saved to {LOG_FILE}")


if __name__ == "__main__":
    main()
