import math

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- IMPORT THE FORGES ---
from src.embedding_space_analysis.chimera import (
    forge_fluid_chimera,
    forge_mass_equalized_chimera,
)
from src.embedding_space_analysis.solver import fast_geometric_solver

# --- THE TETHER ---
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

THOUGHT_START_TAG = "<|begin_of_thought|>"
THOUGHT_END_TAG = "<|end_of_thought|>\nAnswer: "

SYSTEM_PROMPT = """You are a helpful assistant.

Once the <|end_of_thought|> tag is provided, provide a clear, natural language answer."""

USER_PROMPT = "Explain the fundamental difference between human consciousness and artificial intelligence."

FULL_PROMPT = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>

{USER_PROMPT}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{THOUGHT_START_TAG}"""

THOUGHT_STEPS = 15
SPEAK_STEPS = 30
K = 10
# Controls the severity of the probability hierarchy.
# 1.0 = Standard. >1.0 = Melts the top token, empowering the underdogs. <1.0 = Sharpens the top token.
TEMPERATURE = 2.0
LOG_FILE = "llama_8b_subconscious_gestation.txt"


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

# --- THE INJECTION CHAMBER ---
input_ids = tokenizer(FULL_PROMPT, return_tensors="pt").input_ids.to(device)
baseline_embeds = model.get_input_embeddings()(input_ids).to(torch.float32)

end_tag_ids = tokenizer(
    THOUGHT_END_TAG, return_tensors="pt", add_special_tokens=False
).input_ids.to(device)
end_tag_embeds = model.get_input_embeddings()(end_tag_ids).to(torch.float32)

algorithms = [
    "1. DISCRETE BASELINE",
    "2. SOFT THINKING (Raw)",
    "3. UNWEIGHTED CENTROID",
    "4. MASS-EQUALIZED GENETIC CHIMERA (Temp)",
    "5. GEOMETRIC SOLVER (Temp)",
    "6. COLAR (Scaled Formula)",
    "7. COCONUT",
]

with open(LOG_FILE, "w", encoding="utf-8") as f:
    emit(f"LLAMA 8B COGNITION LOG: THE SUBCONSCIOUS GESTATION", f)
    emit(f"Prompt Phase: System instructed to incubate continuous vectors.", f)
    emit(
        f"Thought Steps: {THOUGHT_STEPS} | Translation Steps: {SPEAK_STEPS} | Temp: {TEMPERATURE}\n",
        f,
    )

    with torch.no_grad():
        for algo_name in algorithms:
            # Skipping the raw baselines to focus on the synthetic geometry
            if "BASELINE" in algo_name or "COCONUT" in algo_name:
                continue

            emit(f"\n{'='*80}", f)
            emit(f"🌌 TIMELINE: {algo_name}", f)
            emit(f"{'='*80}", f)

            current_embeds = baseline_embeds.clone()

            # --- PHASE 1: THE DARK ROOM ---
            emit(f"\n[ENTERING SUBCONSCIOUS THOUGHT LOOP...]", f)
            for step in range(THOUGHT_STEPS):
                outputs = model(
                    inputs_embeds=current_embeds.to(torch.bfloat16),
                    output_hidden_states=True,
                )

                next_token_logits = outputs.logits[0, -1, :].to(torch.float32)
                probs = F.softmax(next_token_logits, dim=-1)

                top_k_probs, top_k_ids = torch.topk(probs, K)
                top_k_words = tokenizer.batch_decode(top_k_ids)

                # --- THE THERMODYNAMIC MELT ---
                adj_probs = top_k_probs ** (1.0 / TEMPERATURE)
                adj_probs = adj_probs / adj_probs.sum()

                emit(f"\n  --- Subconscious Pulse {step + 1} ---", f)
                emit(f"    [Model's Logit Predictions]", f)
                for rank, (word, raw_p, adj_p) in enumerate(
                    zip(top_k_words, top_k_probs, adj_probs)
                ):
                    clean_word = word.replace("\n", "\\n").replace("\r", "\\r")
                    emit(
                        f"      {rank+1:2d}. [Raw: {raw_p*100:5.2f}% | Tempered: {adj_p*100:5.2f}%] '{clean_word}'",
                        f,
                    )

                top_k_raw_embs = raw_embeddings[top_k_ids]
                avg_target_mag = torch.norm(top_k_raw_embs, p=2, dim=1).mean().item()

                # Calculate the core tempered chassis once
                soft_raw = torch.sum(
                    top_k_raw_embs * adj_probs.unsqueeze(1), dim=0, keepdim=True
                )
                soft_norm = F.normalize(soft_raw, p=2, dim=1)

                # Route to the appropriate mathematical forge
                if "SOFT THINKING" in algo_name:
                    # No Temp for Soft Thinking Control
                    raw_soft = torch.sum(
                        top_k_raw_embs * top_k_probs.unsqueeze(1), dim=0, keepdim=True
                    )
                    next_vector = F.normalize(raw_soft, p=2, dim=1) * avg_target_mag

                elif "CENTROID" in algo_name:
                    centroid_raw = torch.mean(top_k_raw_embs, dim=0, keepdim=True)
                    next_vector = F.normalize(centroid_raw, p=2, dim=1) * avg_target_mag

                elif "MASS-EQUALIZED" in algo_name:
                    with torch.enable_grad():
                        next_vector = forge_mass_equalized_chimera(
                            top_k_ids,
                            norm_dictionary,
                            dict_mean,
                            dict_std,
                            avg_target_mag,
                            adj_probs,
                        )

                elif "GEOMETRIC SOLVER" in algo_name:
                    target_embs_norm = norm_dictionary[top_k_ids]
                    with torch.enable_grad():
                        next_vector = fast_geometric_solver(
                            target_embs_norm,
                            top_k_ids,
                            norm_dictionary,
                            soft_norm,
                            avg_target_mag,
                            adj_probs,
                        )

                elif "COLAR" in algo_name:
                    # The CoLaR scaling: Summing the top K elements and dividing by sqrt(K)
                    c = K
                    colar_raw = torch.sum(top_k_raw_embs, dim=0, keepdim=True)
                    next_vector = colar_raw / math.sqrt(c)

                # --- THE X-RAY ---
                unit_next_vector = F.normalize(next_vector, p=2, dim=1)
                cos_sims = torch.matmul(unit_next_vector, norm_dictionary.T).squeeze(0)

                top_sim_vals, top_sim_idxs = torch.topk(cos_sims, K)
                top_sim_words = tokenizer.batch_decode(top_sim_idxs)

                # Note: We display the raw mass here so you can verify how well CoLaR preserves it natively
                actual_mag = torch.norm(next_vector, p=2, dim=1).item()
                emit(
                    f"\n    [Forged Geometry Nearest Neighbors | Mass: {actual_mag:.4f} | Target Mass: {avg_target_mag:.4f}]",
                    f,
                )

                for rank, (word, sim) in enumerate(zip(top_sim_words, top_sim_vals)):
                    clean_word = word.replace("\n", "\\n").replace("\r", "\\r")
                    marker = "*" if word in top_k_words else " "
                    emit(
                        f"      NN {rank+1:2d} | Sim: {sim:.4f} | {marker} '{clean_word}'",
                        f,
                    )

                current_embeds = torch.cat(
                    [current_embeds, next_vector.unsqueeze(0)], dim=1
                )

            emit(
                f"\n[INCUBATION COMPLETE. {THOUGHT_STEPS} vectors forged and injected.]",
                f,
            )

            # --- PHASE 2: THE AWAKENING ---
            current_embeds = torch.cat([current_embeds, end_tag_embeds], dim=1)

            # --- PHASE 3: THE TRANSLATION ---
            emit(f"\n[TRANSLATING SUBCONSCIOUS TO NATURAL LANGUAGE...]", f)

            generated_text = ""
            for step in range(SPEAK_STEPS):
                outputs = model(inputs_embeds=current_embeds.to(torch.bfloat16))

                next_token_id = torch.argmax(outputs.logits[0, -1, :])
                next_token_word = tokenizer.decode([next_token_id])

                generated_text += next_token_word

                real_vector = raw_embeddings[next_token_id : next_token_id + 1]
                current_embeds = torch.cat(
                    [current_embeds, real_vector.unsqueeze(0)], dim=1
                )

            emit(f"\nFinal Translated Output:\n'{generated_text}'\n", f)

    emit("\nThe grand experiment is complete. Check the logs.", f)
