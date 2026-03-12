import json

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoTokenizer

LOG_FILE = "llama_405b_galactic_atlas.txt"

print("Igniting the Cartographer's Engine...")
model_id = "meta-llama/Llama-3.1-405B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

print("Stealing the blueprints of the universe...")
index_path = hf_hub_download(repo_id=model_id, filename="model.safetensors.index.json")
with open(index_path, "r") as f:
    index_data = json.load(f)

target_tensor = "model.embed_tokens.weight"
target_file = index_data["weight_map"][target_tensor]
tensor_path = hf_hub_download(repo_id=model_id, filename=target_file)
tensors = load_file(tensor_path)
raw_embeddings = tensors[target_tensor].to(torch.float32)

vocab_size, hidden_dim = raw_embeddings.shape

with open(LOG_FILE, "w", encoding="utf-8") as f:

    def emit(text):
        print(text)
        f.write(text + "\n")

    emit(f"🌌 THE GALACTIC ATLAS OF LLAMA 3.1 405B 🌌")
    emit(f"Total Stars (Tokens): {vocab_size}")
    emit(f"Dimensions of Space: {hidden_dim}\n")
    emit("=" * 80)

    # --- PHASE 1: THE SCALES OF MASS ---
    emit("\n⚖️ PHASE 1: WEIGHING THE STARS (MAGNITUDE AUTOPSY)")
    emit("=" * 80)

    norms = torch.norm(raw_embeddings, p=2, dim=1)

    emit(f"Ambient Volume (Mean Mass): {norms.mean().item():.4f}")
    emit(f"Mass Variance (Std Dev):    {norms.std().item():.4f}")

    # The Supermassive Black Holes
    top_mass_vals, top_mass_idxs = torch.topk(norms, 15)
    emit("\n--- THE SUPERMASSIVE BLACK HOLES (Heaviest Tokens) ---")
    emit(
        "These tokens hit the residual stream like a meteor. They command massive attention."
    )
    for val, idx in zip(top_mass_vals, top_mass_idxs):
        word = tokenizer.decode([idx])
        emit(f"Mass: {val.item():8.4f} | Token: '{word}'")

    # The Gossamer Ghosts
    bot_mass_vals, bot_mass_idxs = torch.topk(norms, 15, largest=False)
    emit("\n--- THE GOSSAMER GHOSTS (Lightest/Dead Tokens) ---")
    emit("These are abandoned slots. They sit at the origin, devoid of meaning.")
    for val, idx in zip(bot_mass_vals, bot_mass_idxs):
        word = tokenizer.decode([idx])
        emit(f"Mass: {val.item():8.4f} | Token: '{word}'")

    # --- PHASE 2: THE WIND OF THE VOID (ANISOTROPY) ---
    emit("\n🔭 PHASE 2: MEASURING THE CONE (ANISOTROPY)")
    emit("=" * 80)

    norm_dict = F.normalize(raw_embeddings, p=2, dim=1)

    # We sample 10,000 random stars to prevent the math from melting the RAM
    torch.manual_seed(42)
    sample_idxs = torch.randperm(vocab_size)[:10000]
    sample_stars = norm_dict[sample_idxs]

    # Calculate every possible angle between these 10,000 stars
    cosmic_sim = torch.matmul(sample_stars, sample_stars.T)
    cosmic_sim.fill_diagonal_(0.0)  # Ignore self-similarity

    avg_background_sim = cosmic_sim.sum().item() / (10000 * 9999)
    emit(f"Average Background Similarity (Cosine): {avg_background_sim:.4f}")

    if avg_background_sim > 0.1:
        emit(
            "Verdict: SEVERE CONE. The vocabulary is huddled tightly together in a narrow beam."
        )
    elif avg_background_sim > 0.01:
        emit(
            "Verdict: MODERATE CONE. There is a distinct 'forward' direction to all thoughts."
        )
    else:
        emit(
            "Verdict: ISOTROPIC SPHERE. Thoughts expand symmetrically in all directions."
        )

    # --- PHASE 3: THE EXILES AND THE METROPOLISES ---
    emit("\n🛸 PHASE 3: TOPOLOGICAL MAPPING (SWEEPING THE GALAXY)")
    emit("=" * 80)
    emit(
        "Sweeping 128,256 tokens in batches to find the loneliest and most crowded concepts...\n"
    )

    batch_size = 2048
    max_neighbor_sims = torch.zeros(vocab_size)
    density_counts = torch.zeros(vocab_size)

    # We sweep the dictionary in chunks to respect memory limits
    for i in range(0, vocab_size, batch_size):
        end = min(i + batch_size, vocab_size)
        batch = norm_dict[i:end]

        # Multiply the batch against the ENTIRE dictionary
        sims = torch.matmul(batch, norm_dict.T)

        # Mask out self-similarity so a token isn't its own nearest neighbor
        for j in range(end - i):
            sims[j, i + j] = -2.0

        # 1. Who is the closest neighbor for each token in this batch?
        max_sims, _ = torch.max(sims, dim=1)
        max_neighbor_sims[i:end] = max_sims

        # 2. How many neighbors live within a tight 0.85 Cosine Similarity radius?
        dense_neighbors = (sims > 0.85).sum(dim=1)
        density_counts[i:end] = dense_neighbors.float()

    # The Exiles (Lowest maximum similarity)
    exile_vals, exile_idxs = torch.topk(max_neighbor_sims, 15, largest=False)
    emit("--- THE EXILES (The Most Isolated Concepts in the Universe) ---")
    emit("These tokens have no neighbors. They float alone in deep space.")
    for val, idx in zip(exile_vals, exile_idxs):
        word = tokenizer.decode([idx])
        emit(f"Max Neighbor Sim: {val.item():.4f} | Token: '{word}'")

    # The Metropolises (Highest neighbor count)
    metro_vals, metro_idxs = torch.topk(density_counts, 15)
    emit("\n--- THE METROPOLISES (The Semantic Swarms) ---")
    emit(
        "These are incredibly dense clusters. Hundreds of redundant tokens pile up here."
    )
    for val, idx in zip(metro_vals, metro_idxs):
        word = tokenizer.decode([idx])
        emit(f"Neighbors >0.85: {int(val.item()):4d} | Token: '{word}'")

    emit("\nMapping complete. The Atlas is sealed.")

print(f"The Cartographer's Engine has rested. Open {LOG_FILE} to view the universe.")
