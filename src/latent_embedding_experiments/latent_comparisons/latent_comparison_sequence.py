import math

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- IMPORTS ---
from src.latent_embedding_experiments.algorithms.chimera import forge_mass_equalized_chimera
from src.latent_embedding_experiments.algorithms.solver import fast_geometric_solver

# --- CONFIGURATION ---
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
PROMPT = "The fundamental difference between human consciousness and artificial intelligence is"
STEPS = 15
K = 10
DISPLAY_K = K * 2
TEMPERATURE = 2.0
LOG_FILE = "src/latent_embedding_experiments/logs/llama_8b_latent_comparison_sequence.txt"


def emit(text, file_handle=None):
    print(text)
    if file_handle:
        file_handle.write(text + "\n")


print(f"Loading model: {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16
)
device = model.device

# --- EMBEDDING DICTIONARY ---
raw_embeddings = model.get_input_embeddings().weight.detach().to(torch.float32)
norm_dictionary = F.normalize(raw_embeddings, p=2, dim=1)
dict_mean = norm_dictionary.mean(dim=0)
dict_std = norm_dictionary.std(dim=0) + 1e-9

# Create a bfloat16 version of raw embeddings for dot-product evaluation
raw_embeddings_bf16 = raw_embeddings.to(torch.bfloat16)

# --- INITIALIZATION ---
input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(device)

pd.set_option("display.float_format", "{:.4f}".format)
pd.set_option("display.width", 1000)

with open(LOG_FILE, "w", encoding="utf-8") as f:
    emit(f"LLAMA 8B LATENT COMPARISON SEQUENCE LOG", f)
    emit(
        f"Mode: Natural Autoregressive Generation with Synthetic Vector Evaluation.", f
    )
    emit(
        f"Steps: {STEPS} | Synthesis Top-K: {K} | Table Display Limit: {DISPLAY_K} | Temp: {TEMPERATURE}\n",
        f,
    )

    context_so_far = PROMPT

    with torch.no_grad():
        for step in range(STEPS):
            emit(f"\n{'='*160}", f)
            emit(f"STEP {step + 1} | Context: '...{context_so_far[-60:]}'", f)
            emit(f"{'='*160}", f)

            # 1. Forward Pass (Now extracting hidden states for Coconut)
            outputs = model(input_ids=input_ids, output_hidden_states=True)

            # 2. Extract Logits
            next_token_logits = outputs.logits[0, -1, :].to(torch.float32)
            probs = F.softmax(next_token_logits, dim=-1)

            top_k_probs, top_k_ids = torch.topk(probs, K)
            top_k_words = tokenizer.batch_decode(top_k_ids)
            target_ids_list = top_k_ids.tolist()

            # 3. Advance Context with Natural Token
            actual_next_token_id = top_k_ids[0:1]
            actual_next_word = top_k_words[0]
            context_so_far += actual_next_word
            input_ids = torch.cat([input_ids, actual_next_token_id.unsqueeze(0)], dim=1)

            # --- TEMPERATURE SCALING ---
            adj_probs = top_k_probs ** (1.0 / TEMPERATURE)
            adj_probs = adj_probs / adj_probs.sum()

            emit(f"--- PREDICTED LOGIT DISTRIBUTION ---", f)
            for rank, (word, raw_p, adj_p) in enumerate(
                zip(top_k_words, top_k_probs, adj_probs)
            ):
                clean_word = word.replace("\n", "\\n").replace("\r", "\\r")
                emit(
                    f"  Rank {rank+1:2d} | Raw: {raw_p*100:5.2f}% -> Scaled: {adj_p*100:5.2f}% | '{clean_word}'",
                    f,
                )

            # --- VECTOR SYNTHESIS ---
            top_k_raw_embs = raw_embeddings[top_k_ids]
            avg_target_mag = torch.norm(top_k_raw_embs, p=2, dim=1).mean().item()

            # A. Baseline
            v_baseline = top_k_raw_embs[0:1]

            # B. Soft Embedding
            raw_soft_raw = torch.sum(
                top_k_raw_embs * top_k_probs.unsqueeze(1), dim=0, keepdim=True
            )
            v_soft = F.normalize(raw_soft_raw, p=2, dim=1) * avg_target_mag
            temp_soft_norm = F.normalize(
                torch.sum(top_k_raw_embs * adj_probs.unsqueeze(1), dim=0, keepdim=True),
                p=2,
                dim=1,
            )

            # C. Centroid
            v_centroid = (
                F.normalize(torch.mean(top_k_raw_embs, dim=0, keepdim=True), p=2, dim=1)
                * avg_target_mag
            )

            # D. Mass-Equalized Chimera
            with torch.enable_grad():
                v_chimera = forge_mass_equalized_chimera(
                    top_k_ids,
                    norm_dictionary,
                    dict_mean,
                    dict_std,
                    avg_target_mag,
                    adj_probs,
                )

            # E. Geometric Solver
            target_embs_norm = norm_dictionary[top_k_ids]
            with torch.enable_grad():
                v_solver = fast_geometric_solver(
                    target_embs_norm,
                    top_k_ids,
                    norm_dictionary,
                    temp_soft_norm,
                    avg_target_mag,
                    adj_probs,
                )

            # F. CoLaR Vector
            v_colar = torch.sum(top_k_raw_embs, dim=0, keepdim=True) / math.sqrt(K)

            # G. Coconut Vector (Continuous Hidden State)
            v_coconut = (
                outputs.hidden_states[-1][0, -1, :].unsqueeze(0).to(torch.float32)
            )

            # --- COMPARATIVE ANALYSIS ---
            vectors_unnormalized = torch.cat(
                [
                    v_baseline,
                    v_soft,
                    v_centroid,
                    v_chimera,
                    v_solver,
                    v_colar,
                    v_coconut,
                ],
                dim=0,
            )
            vectors_unit = F.normalize(vectors_unnormalized, p=2, dim=1)
            mags = torch.norm(vectors_unnormalized, p=2, dim=1).cpu().numpy()

            labels = [
                "Baseline",
                "Soft",
                "Centroid",
                "Chimera",
                "Solver",
                "CoLaR",
                "Coconut",
            ]
            sim_matrix = torch.matmul(vectors_unit, vectors_unit.T)
            df = pd.DataFrame(sim_matrix.cpu().numpy(), index=labels, columns=labels)

            emit(f"\n--- CROSS-SIMILARITY MATRIX ---", f)
            emit(df.to_string(), f)

            col_width = 25

            # =====================================================================
            # TABLE 1: COSINE SIMILARITY
            # =====================================================================
            emit(
                f"\n--- [TABLE 1] TOP {DISPLAY_K} NEAREST NEIGHBORS (COSINE SIMILARITY) ---",
                f,
            )
            cos_sims_all = torch.matmul(vectors_unit, norm_dictionary.T)
            top_k_vals_cos, top_k_idx_cos = torch.topk(cos_sims_all, DISPLAY_K, dim=1)

            header_cos = f"{'Rank':<4} |"
            for label in labels:
                header_title = f"{label} (Ang)"
                header_cos += f" {header_title:<{col_width}} |"
            emit(header_cos[:-2], f)
            emit("-" * len(header_cos[:-2]), f)

            for rank in range(DISPLAY_K):
                row_str = f"{rank+1:<4} |"
                for i in range(len(labels)):
                    token_id = top_k_idx_cos[i, rank].item()
                    w = tokenizer.decode([token_id])
                    cw = w.replace("\n", "\\n").replace("\r", "\\r")
                    if len(cw) > 10:
                        cw = cw[:8] + ".."

                    marker = "*" if token_id in target_ids_list else " "
                    val = top_k_vals_cos[i, rank].item()
                    cell = f"{marker}'{cw}' ({val:.3f})"
                    row_str += f" {cell:<{col_width}} |"
                emit(row_str[:-2], f)

            # =====================================================================
            # TABLE 2: UNNORMALIZED DOT PRODUCT
            # =====================================================================
            emit(
                f"\n--- [TABLE 2] TOP {DISPLAY_K} TOKENS BY MAGNITUDE (UNNORMALIZED DOT PRODUCT) ---",
                f,
            )

            # Cast unnormalized vectors to bfloat16 to match the dictionary
            vectors_unnormalized_bf16 = vectors_unnormalized.to(torch.bfloat16)
            dot_prods_all = torch.matmul(
                vectors_unnormalized_bf16, raw_embeddings_bf16.T
            )
            top_k_vals_dot, top_k_idx_dot = torch.topk(dot_prods_all, DISPLAY_K, dim=1)

            header_dot = f"{'Rank':<4} |"
            for label, mag in zip(labels, mags):
                header_title = f"{label} (Mag: {mag:.1f})"
                header_dot += f" {header_title:<{col_width}} |"
            emit(header_dot[:-2], f)
            emit("-" * len(header_dot[:-2]), f)

            for rank in range(DISPLAY_K):
                row_str = f"{rank+1:<4} |"
                for i in range(len(labels)):
                    token_id = top_k_idx_dot[i, rank].item()
                    w = tokenizer.decode([token_id])
                    cw = w.replace("\n", "\\n").replace("\r", "\\r")
                    if len(cw) > 10:
                        cw = cw[:8] + ".."

                    marker = "*" if token_id in target_ids_list else " "
                    val = top_k_vals_dot[i, rank].item()

                    cell = f"{marker}'{cw}' ({val:>7.3f})"
                    row_str += f" {cell:<{col_width}} |"
                emit(row_str[:-2], f)

    emit(f"\nProcessing complete. Log file saved to {LOG_FILE}.", f)
