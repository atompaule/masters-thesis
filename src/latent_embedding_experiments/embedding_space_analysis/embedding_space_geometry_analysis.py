import json

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoTokenizer

LOG_FILE = "src/latent_embedding_experiments/logs/01_llama_405b_embedding_space_analysis.txt"
MODEL_ID = "meta-llama/Llama-3.1-405B-Instruct"


def log(message, file):
    print(message)
    file.write(message + "\n")


def load_embeddings(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    index_path = hf_hub_download(
        repo_id=model_id, filename="model.safetensors.index.json"
    )

    with open(index_path, "r") as f:
        index_data = json.load(f)

    tensor_name = "model.embed_tokens.weight"
    tensor_file = index_data["weight_map"][tensor_name]

    tensor_path = hf_hub_download(repo_id=model_id, filename=tensor_file)
    tensors = load_file(tensor_path)

    embeddings = tensors[tensor_name].to(torch.float32)

    return tokenizer, embeddings


def analyze_norms(embeddings, tokenizer, log_file):
    norms = torch.norm(embeddings, dim=1)

    log(f"Mean norm: {norms.mean().item():.4f}", log_file)
    log(f"Median norm: {norms.median().item():.4f}", log_file)
    log(f"Std norm:  {norms.std().item():.4f}", log_file)

    top_vals, top_idxs = torch.topk(norms, 15)
    log("\nTop tokens by norm:", log_file)
    for val, idx in zip(top_vals, top_idxs):
        token = tokenizer.decode([idx])
        log(f"{val.item():.4f} | '{token}'", log_file)

    low_vals, low_idxs = torch.topk(norms, 15, largest=False)
    log("\nLowest tokens by norm:", log_file)
    for val, idx in zip(low_vals, low_idxs):
        token = tokenizer.decode([idx])
        log(f"{val.item():.4f} | '{token}'", log_file)


def analyze_anisotropy(embeddings, log_file, sample_size=10000):
    normalized = F.normalize(embeddings, dim=1)

    torch.manual_seed(42)
    sample_idxs = torch.randperm(normalized.shape[0])[:sample_size]
    sample = normalized[sample_idxs]

    sim_matrix = sample @ sample.T
    sim_matrix.fill_diagonal_(0.0)

    avg_sim = sim_matrix.sum().item() / (sample_size * (sample_size - 1))
    log(f"\nAverage cosine similarity: {avg_sim:.4f}", log_file)

    if avg_sim > 0.1:
        log("High anisotropy (narrow cone)", log_file)
    elif avg_sim > 0.01:
        log("Moderate anisotropy", log_file)
    else:
        log("Near-isotropic distribution", log_file)

    return normalized


def analyze_topology(normalized, tokenizer, log_file, batch_size=2048):
    vocab_size = normalized.shape[0]

    max_neighbor_sim = torch.zeros(vocab_size)
    density_counts = torch.zeros(vocab_size)

    for i in range(0, vocab_size, batch_size):
        end = min(i + batch_size, vocab_size)
        batch = normalized[i:end]

        sims = batch @ normalized.T

        # remove self-similarity
        for j in range(end - i):
            sims[j, i + j] = -1.0

        max_sims, _ = torch.max(sims, dim=1)
        max_neighbor_sim[i:end] = max_sims

        density = (sims > 0.85).sum(dim=1)
        density_counts[i:end] = density.float()

    # isolated tokens
    low_vals, low_idxs = torch.topk(max_neighbor_sim, 15, largest=False)
    log("\nMost isolated tokens:", log_file)
    for val, idx in zip(low_vals, low_idxs):
        token = tokenizer.decode([idx])
        log(f"{val.item():.4f} | '{token}'", log_file)

    # dense clusters
    high_vals, high_idxs = torch.topk(density_counts, 15)
    log("\nMost dense tokens:", log_file)
    for val, idx in zip(high_vals, high_idxs):
        token = tokenizer.decode([idx])
        log(f"{int(val.item())} neighbors | '{token}'", log_file)


def main():
    tokenizer, embeddings = load_embeddings(MODEL_ID)

    vocab_size, dim = embeddings.shape

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        log(f"Vocab size: {vocab_size}", f)
        log(f"Embedding dimension: {dim}", f)

        analyze_norms(embeddings, tokenizer, f)
        analyze_anisotropy(embeddings, f)

    print(f"Done. Results saved to {LOG_FILE}")


if __name__ == "__main__":
    main()
