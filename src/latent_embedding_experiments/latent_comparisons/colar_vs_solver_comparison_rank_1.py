import math

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- IMPORTS ---
from src.latent_embedding_experiments.algorithms.solver import geometric_solver

# --- CONFIGURATION ---
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

K = 10
TEMPERATURE = 2.0

# --- EVALUATION THRESHOLDS ---
RANK_THRESHOLD = 3  # CoLaR failure: Target token rank exceeds this value
SIM_GAP_THRESHOLD = (
    0.08  # CoLaR failure: Solver cosine similarity exceeds CoLaR by this margin
)

NUM_SEQUENCES = 5  # Number of GSM8K samples to evaluate
MAX_STEPS_PER_SEQ = 40  # Maximum generation steps analyzed per sequence

LOG_FILE = "src/latent_embedding_experiments/logs/llama_8b_colar_vs_solver_comparison_rank_1.txt"


def emit(text, file_handle):
    print(text)
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

# --- DATASET PREPARATION ---
print("Loading GSM8K dataset...")
dataset = load_dataset("gsm8k", "main", split="train")

with open(LOG_FILE, "w", encoding="utf-8") as f:
    emit(f"LATENT GEOMETRY EVALUATION LOG", f)
    emit(f"Evaluating {NUM_SEQUENCES} reasoning sequences from GSM8K.", f)
    emit(
        f"Failure Conditions: Target Rank > {RANK_THRESHOLD} OR Solver Sim Advantage > {SIM_GAP_THRESHOLD}\n",
        f,
    )

    total_fractures = 0
    total_steps_analyzed = 0

    with torch.no_grad():
        for seq_idx in range(NUM_SEQUENCES):
            question = dataset[seq_idx]["question"]

            # Format prompt
            prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

            emit(f"\n{'='*100}", f)
            emit(f"SEQUENCE {seq_idx + 1} | PROMPT: {question[:80]}...", f)
            emit(f"{'='*100}", f)

            for step in range(MAX_STEPS_PER_SEQ):
                total_steps_analyzed += 1

                # 1. Forward Pass
                outputs = model(input_ids=input_ids)
                next_token_logits = outputs.logits[0, -1, :].to(torch.float32)
                probs = F.softmax(next_token_logits, dim=-1)

                top_k_probs, top_k_ids = torch.topk(probs, K)
                top_1_id = top_k_ids[0].item()
                top_1_word = (
                    tokenizer.decode([top_1_id])
                    .replace("\n", "\\n")
                    .replace("\r", "\\r")
                )

                # Append actual predicted token to continue generation
                input_ids = torch.cat([input_ids, top_k_ids[0:1].unsqueeze(0)], dim=1)

                # --- VECTOR SYNTHESIS ---
                top_k_raw_embs = raw_embeddings[top_k_ids]
                avg_target_mag = torch.norm(top_k_raw_embs, p=2, dim=1).mean().item()

                adj_probs = top_k_probs ** (1.0 / TEMPERATURE)
                adj_probs = adj_probs / adj_probs.sum()

                # Base vector for Solver
                temp_soft_raw = torch.sum(
                    top_k_raw_embs * adj_probs.unsqueeze(1), dim=0, keepdim=True
                )
                temp_soft_norm = F.normalize(temp_soft_raw, p=2, dim=1)

                # A. CoLaR Vector
                v_colar = torch.sum(top_k_raw_embs, dim=0, keepdim=True) / math.sqrt(K)
                colar_unit = F.normalize(v_colar, p=2, dim=1)

                # B. Solver Vector (T=1 top-k masses, renormalized; solver applies p^(1/T))
                target_embs_norm = norm_dictionary[top_k_ids]
                solver_pool_probs = top_k_probs / top_k_probs.sum()
                with torch.enable_grad():
                    v_solver = geometric_solver(
                        target_embs_norm,
                        top_k_ids.tolist(),
                        norm_dictionary,
                        avg_target_mag,
                        solver_pool_probs,
                        temperature=TEMPERATURE,
                    )
                solver_unit = F.normalize(v_solver, p=2, dim=1)

                # --- EVALUATION ---
                colar_sims = torch.matmul(colar_unit, norm_dictionary.T).squeeze(0)
                solver_sims = torch.matmul(solver_unit, norm_dictionary.T).squeeze(0)

                colar_top1_sim = colar_sims[top_1_id].item()
                solver_top1_sim = solver_sims[top_1_id].item()

                # Calculate dictionary rank of the target token
                colar_top1_rank = (colar_sims > colar_top1_sim).sum().item() + 1
                solver_top1_rank = (solver_sims > solver_top1_sim).sum().item() + 1

                # --- FAILURE DETECTION ---
                rank_failure = colar_top1_rank > RANK_THRESHOLD
                sim_failure = (solver_top1_sim - colar_top1_sim) > SIM_GAP_THRESHOLD

                if rank_failure or sim_failure:
                    total_fractures += 1
                    reason = "RANK EXCEEDED" if rank_failure else "SIMILARITY GAP"
                    if rank_failure and sim_failure:
                        reason = "RANK & SIMILARITY FAILURE"

                    emit(
                        f"\n  [DEGRADATION DETECTED] | Seq: {seq_idx + 1} | Step: {step + 1} | Target: '{top_1_word}'",
                        f,
                    )
                    emit(f"      Type: {reason}", f)
                    emit(
                        f"      CoLaR  -> Cosine Sim: {colar_top1_sim:.4f} | Rank: {colar_top1_rank}",
                        f,
                    )
                    emit(
                        f"      Solver -> Cosine Sim: {solver_top1_sim:.4f} | Rank: {solver_top1_rank}",
                        f,
                    )

    # --- SUMMARY ---
    emit(f"\n{'='*100}", f)
    emit(f"EVALUATION COMPLETE", f)
    emit(f"{'='*100}", f)
    emit(f"Total Steps Analyzed:  {total_steps_analyzed}", f)
    emit(f"Total Degradations:    {total_fractures}", f)

    failure_rate = (
        (total_fractures / total_steps_analyzed) * 100
        if total_steps_analyzed > 0
        else 0
    )
    emit(f"Degradation Rate:      {failure_rate:.2f}%", f)

print(f"\nProcessing complete. Log file saved to {LOG_FILE}.")
