import json

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoTokenizer

# --- IMPORT THE FORGES ---
from src.embedding_space_analysis.chimera import forge_mass_equalized_chimera
from src.embedding_space_analysis.solver import fast_geometric_solver

LOG_FILE = "llama_405b_magnitude_cognition.txt"
TEMPERATURE = 1.0  # Controls probability hierarchy (1.0 = Raw Model Distribution)

# --- STEP 1: DATASET TETHER ---
print(
    f"Initiating tether... The magnitude-locked ledger is being carved into: {LOG_FILE}"
)
dataset = load_dataset("arcee-ai/LLama-405B-Logits", split="train", streaming=True)
first_sequence = next(iter(dataset))
input_tokens = np.array(first_sequence["input_ids"])
top_tokens = np.array(first_sequence["token_ids"])
top_scores = np.array(first_sequence["top_values"])

# --- STEP 2: THE SURGICAL HEIST ---
model_id = "meta-llama/Llama-3.1-405B-Instruct"
index_path = hf_hub_download(repo_id=model_id, filename="model.safetensors.index.json")
with open(index_path, "r") as f:
    index_data = json.load(f)

target_tensor = "model.embed_tokens.weight"
target_file = index_data["weight_map"][target_tensor]
tensor_path = hf_hub_download(repo_id=model_id, filename=target_file)
tensors = load_file(tensor_path)
raw_embeddings = tensors[target_tensor].to(torch.float32)

tokenizer = AutoTokenizer.from_pretrained(model_id)

# --- PRE-COMPUTING GEOMETRY ---
# The dictionary is normalized for measuring pure angles and Z-scores
norm_dictionary = F.normalize(raw_embeddings, p=2, dim=1)
dict_mean = norm_dictionary.mean(dim=0)
dict_std = norm_dictionary.std(dim=0) + 1e-9


# --- STEP 3: THE LOGGING ENGINE ---
start_step = 500
end_step = 800
k = 10
context_so_far = tokenizer.decode(input_tokens[: start_step + 1])

pd.set_option("display.float_format", "{:.4f}".format)
pd.set_option("display.width", 1000)

with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write(f"LLAMA 405B COGNITION LOG: THE MAGNITUDE-LOCKED PENTAGRAM\n")
    f.write(
        f"Using Modular Forges. Temperature: {TEMPERATURE}. Locked to Target L2 Magnitude.\n"
    )
    f.write(f"{'='*120}\n\n")

    for t in range(start_step, end_step):
        actual_word = tokenizer.decode([input_tokens[t + 1]])
        context_so_far += actual_word

        f.write(f"\n{'='*120}\n")
        f.write(f"⏱️ TIME STEP {t}\n")
        f.write(f"{'='*120}\n")
        f.write(f"CONTEXT: '...{context_so_far[-150:]}'\n")
        f.write(f"ACTUAL CHOSEN TOKEN: '{actual_word}'\n\n")

        # 1. Contenders & Probabilities
        top_k_ids = top_tokens[t][:k]
        top_k_raw_scores = top_scores[t][:k]
        top_k_raw_embs = raw_embeddings[top_k_ids]
        top_k_words = tokenizer.batch_decode(top_k_ids)

        scores_tensor = torch.tensor(top_k_raw_scores, dtype=torch.float32)
        probs = F.softmax(scores_tensor, dim=0)

        # --- THE THERMODYNAMIC MELT ---
        adj_probs = probs ** (1.0 / TEMPERATURE)
        adj_probs = adj_probs / adj_probs.sum()

        f.write("--- TOP CONTENDERS ---\n")
        for rank, (word, raw_p, adj_p, logit) in enumerate(
            zip(top_k_words, probs, adj_probs, top_k_raw_scores)
        ):
            f.write(
                f"Rank {rank+1:2d} | Raw: {raw_p*100:6.2f}% -> Temp: {adj_p*100:6.2f}% | Logit: {logit:6.2f} | Word: '{word}'\n"
            )

        # 2. Angle Matrix (Cosine Similarity of the Targets)
        norm_contenders = norm_dictionary[top_k_ids]
        sim_matrix = torch.matmul(norm_contenders, norm_contenders.T)
        f.write("\n--- GEOMETRY OF HESITATION (Pairwise Cosine Similarity) ---\n")
        df = pd.DataFrame(sim_matrix.numpy(), index=top_k_words, columns=top_k_words)
        f.write(df.to_string() + "\n")

        # ===================================================================
        # 3. FORGING THE FIVE ANCHORS (MAGNITUDE LOCKED)
        # ===================================================================

        # Calculate the acoustic volume of this specific hesitation
        target_mags = torch.norm(top_k_raw_embs, p=2, dim=1)
        avg_target_mag = target_mags.mean().item()

        # 1. The Pure Probabilistic Control (Raw Probabilities)
        soft_raw = torch.sum(top_k_raw_embs * probs.unsqueeze(1), dim=0, keepdim=True)
        soft_embedding = F.normalize(soft_raw, p=2, dim=1) * avg_target_mag

        # 2. The Tempered Chassis (For the Solver's Drop Point)
        temp_soft_raw = torch.sum(
            top_k_raw_embs * adj_probs.unsqueeze(1), dim=0, keepdim=True
        )
        temp_soft_norm = F.normalize(temp_soft_raw, p=2, dim=1)

        # 3. Unweighted Centroid (Democratic Center)
        centroid_raw = torch.mean(top_k_raw_embs, dim=0, keepdim=True)
        centroid_embedding = F.normalize(centroid_raw, p=2, dim=1) * avg_target_mag

        # 4. Top Candidate (Naturally on the shell)
        top_candidate_emb = top_k_raw_embs[0:1]

        # 5. Synthetic Vectors (Calling the modular forges)
        chimera_vector = forge_mass_equalized_chimera(
            top_k_ids, norm_dictionary, dict_mean, dict_std, avg_target_mag, adj_probs
        )

        solver_vector = fast_geometric_solver(
            norm_contenders,
            top_k_ids,
            norm_dictionary,
            temp_soft_norm,  # Dropping the sniper at the tempered location
            avg_target_mag,
            adj_probs,
        )

        # ===================================================================
        # 4. PRINTING THE PENTAPARTITE LEDGER
        # ===================================================================

        def write_magnitude_table(anchor_tensor, title):
            # 1. Measure the exact physical mass of the anchor
            actual_mag = torch.norm(anchor_tensor, p=2, dim=1).item()

            # 2. Convert to a pure angle (unit vector) to scan the stars
            anchor_unit = F.normalize(anchor_tensor, p=2, dim=1)
            cos_sims = torch.matmul(anchor_unit, norm_dictionary.T).squeeze(0)

            vals, idxs = torch.topk(cos_sims, 10)
            words = tokenizer.batch_decode(idxs)

            f.write(f"\n--- {title} (Mass: {actual_mag:.4f}) ---\n")
            f.write(f"{'Rank':<5} | {'Cos Sim':<8} | {'Word'}\n")
            f.write(f"{'-'*45}\n")
            for i, (word, cos_sim) in enumerate(zip(words, vals)):
                marker = "*" if word in top_k_words else " "
                clean_word = word.replace("\n", "\\n").replace("\r", "\\r")
                f.write(f"NN {i+1:2d} | {cos_sim:.4f}   | {marker} '{clean_word}'\n")

        f.write(f"\nTARGET MAGNITUDE TO MATCH: {avg_target_mag:.4f}\n")

        write_magnitude_table(
            soft_embedding, "TABLE 1: 'SOFT EMBEDDING' (Raw Probabilities)"
        )
        write_magnitude_table(
            centroid_embedding,
            "TABLE 2: 'UNWEIGHTED CENTROID' (Democratic Center)",
        )
        write_magnitude_table(
            top_candidate_emb, f"TABLE 3: TOP CANDIDATE: '{top_k_words[0]}'"
        )
        write_magnitude_table(
            chimera_vector, "TABLE 4: 'MASS-EQ CHIMERA' (Tempered Genetic Splice)"
        )
        write_magnitude_table(
            solver_vector, "TABLE 5: 'GEOMETRIC SOLVER' (Tempered Contrastive Scalpel)"
        )

        if t % 10 == 0:
            print(f"Archiving step {t} into the magnitude-locked pentagram...")

print(
    f"\nExtraction complete. The 405B Leviathan's voice has been perfectly synthesized: {LOG_FILE}"
)
