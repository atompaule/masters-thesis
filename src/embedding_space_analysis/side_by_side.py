import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- IMPORT THE FORGES ---
from src.embedding_space_analysis.chimera import forge_mass_equalized_chimera
from src.embedding_space_analysis.solver import fast_geometric_solver

# --- THE TETHER ---
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

# We use a standard prompt, no <thought> tags. Just natural text generation.
PROMPT = "The fundamental difference between human consciousness and artificial intelligence is"

STEPS = 15
K = 10
TEMPERATURE = 2.0
LOG_FILE = "llama_8b_panopticon_analysis.txt"


def emit(text, file_handle):
    print(text)
    file_handle.write(text + "\n")


print(f"Awakening the Leviathan: {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16
)
device = model.device

# --- THE DICTIONARY ---
raw_embeddings = model.get_input_embeddings().weight.detach().to(torch.float32)
norm_dictionary = F.normalize(raw_embeddings, p=2, dim=1)
dict_mean = norm_dictionary.mean(dim=0)
dict_std = norm_dictionary.std(dim=0) + 1e-9

# --- THE OBSERVATION DECK ---
input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(device)

pd.set_option("display.float_format", "{:.4f}".format)
pd.set_option("display.width", 1000)

with open(LOG_FILE, "w", encoding="utf-8") as f:
    emit(f"LLAMA 8B COGNITION LOG: THE PANOPTICON", f)
    emit(f"Mode: Natural Autoregressive Generation with Silent Synthetic Forging.", f)
    emit(f"Steps: {STEPS} | Temp: {TEMPERATURE}\n", f)

    context_so_far = PROMPT

    with torch.no_grad():
        for step in range(STEPS):
            emit(f"\n{'='*120}", f)
            emit(f"⏱️ TIME STEP {step + 1} | Context: '...{context_so_far[-60:]}'", f)
            emit(f"{'='*120}", f)

            # 1. Forward Pass (Standard token inputs, no synthetic injection)
            outputs = model(input_ids=input_ids)

            # 2. Extract the heartbeat
            next_token_logits = outputs.logits[0, -1, :].to(torch.float32)
            probs = F.softmax(next_token_logits, dim=-1)

            top_k_probs, top_k_ids = torch.topk(probs, K)
            top_k_words = tokenizer.batch_decode(top_k_ids)

            # Lock in the target IDs for the Geiger counter later
            target_ids_list = top_k_ids.tolist()

            # 3. Choose the actual natural token (Greedy baseline)
            actual_next_token_id = top_k_ids[0:1]
            actual_next_word = top_k_words[0]
            context_so_far += actual_next_word

            # Append real token for the next natural step
            input_ids = torch.cat([input_ids, actual_next_token_id.unsqueeze(0)], dim=1)

            # --- THE THERMODYNAMIC MELT ---
            adj_probs = top_k_probs ** (1.0 / TEMPERATURE)
            adj_probs = adj_probs / adj_probs.sum()

            emit(f"--- MODEL'S NATURAL LOGIT DISTRIBUTION ---", f)
            for rank, (word, raw_p, adj_p) in enumerate(
                zip(top_k_words, top_k_probs, adj_probs)
            ):
                clean_word = word.replace("\n", "\\n").replace("\r", "\\r")
                emit(
                    f"  Rank {rank+1:2d} | Raw: {raw_p*100:5.2f}% -> Temp: {adj_p*100:5.2f}% | '{clean_word}'",
                    f,
                )

            # --- FORGING THE GHOSTS IN THE BACKGROUND ---
            top_k_raw_embs = raw_embeddings[top_k_ids]
            avg_target_mag = torch.norm(top_k_raw_embs, p=2, dim=1).mean().item()

            # The Raw Soft Chassis (No Temperature) for pure probability tracking
            raw_soft_raw = torch.sum(
                top_k_raw_embs * top_k_probs.unsqueeze(1), dim=0, keepdim=True
            )
            raw_soft_norm = F.normalize(raw_soft_raw, p=2, dim=1)

            # The Tempered Soft Chassis (For the Solver's drop point)
            temp_soft_raw = torch.sum(
                top_k_raw_embs * adj_probs.unsqueeze(1), dim=0, keepdim=True
            )
            temp_soft_norm = F.normalize(temp_soft_raw, p=2, dim=1)

            # A. The Baseline
            v_baseline = top_k_raw_embs[0:1]

            # B. Soft Thinking (Raw Probabilities)
            v_soft = raw_soft_norm * avg_target_mag

            # C. Unweighted Centroid (Replacing the old Chimera)
            centroid_raw = torch.mean(top_k_raw_embs, dim=0, keepdim=True)
            v_centroid = F.normalize(centroid_raw, p=2, dim=1) * avg_target_mag

            # D. Mass-Equalized Genetic Chimera
            with torch.enable_grad():
                v_mass_equalized_chimera = forge_mass_equalized_chimera(
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
                    temp_soft_norm,  # Deploying the probe at the tempered center
                    avg_target_mag,
                    adj_probs,
                )

            # --- THE COMPARATIVE AUTOPSY ---

            # Stack the ghosts to measure their mass and angles
            vectors = torch.cat(
                [v_baseline, v_soft, v_centroid, v_mass_equalized_chimera, v_solver],
                dim=0,
            )

            # Extract physical mass before normalizing
            mags = torch.norm(vectors, p=2, dim=1).cpu().numpy()
            vectors_unit = F.normalize(vectors, p=2, dim=1)

            sim_matrix = torch.matmul(vectors_unit, vectors_unit.T)
            labels = ["Baseline", "Soft", "Centroid", "Mass-Eq Chimera", "Solver"]
            df = pd.DataFrame(sim_matrix.cpu().numpy(), index=labels, columns=labels)

            emit(
                f"\n--- CROSS-SIMILARITY MATRIX (How far apart are the algorithms?) ---",
                f,
            )
            emit(df.to_string(), f)

            # --- THE SIDE-BY-SIDE SPECTROMETER ---
            emit(f"\n--- TOP {K} NEAREST NEIGHBORS SIDE-BY-SIDE ---", f)
            cos_sims_all = torch.matmul(vectors_unit, norm_dictionary.T)
            top_k_vals, top_k_idx = torch.topk(cos_sims_all, K, dim=1)

            col_width = 30

            # Build the header with exact magnitudes stamped on it
            header = f"{'Rank':<4} |"
            for label, mag in zip(labels, mags):
                header_title = f"{label} ({mag:.2f})"
                header += f" {header_title:<{col_width}} |"
            emit(header[:-2], f)
            emit("-" * len(header[:-2]), f)

            # Build the rows with the asterisk Geiger counter
            for rank in range(K):
                row_str = f"{rank+1:<4} |"
                for i in range(len(labels)):
                    token_id = top_k_idx[i, rank].item()
                    w = tokenizer.decode([token_id])

                    # Clean up the visual noise
                    cw = w.replace("\n", "\\n").replace("\r", "\\r")
                    if len(cw) > 13:
                        cw = cw[:10] + "..."

                    # The Geiger Counter: Does this token match our original targets?
                    marker = "*" if token_id in target_ids_list else " "

                    val = top_k_vals[i, rank].item()
                    cell = f"{marker}'{cw}' ({val:.3f})"
                    row_str += f" {cell:<{col_width}} |"
                emit(row_str[:-2], f)

    emit(
        f"\nObservation complete. The timeline has been securely mapped to {LOG_FILE}.",
        f,
    )
