import math
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# --- IMPORT THE FORGES ---
from src.embedding_space_analysis.solver import fast_geometric_solver

# --- THE TETHER ---
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

# --- THE SEISMOGRAPH THRESHOLDS ---
K = 10
TEMPERATURE = 2.0
RANK_THRESHOLD = 3         # CoLaR fails if the #1 predicted token drops below this rank in its geometry
SIM_GAP_THRESHOLD = 0.08   # CoLaR fails if the Solver beats its #1 similarity by this margin

NUM_SEQUENCES = 5          # How many GSM8K math problems to solve
MAX_STEPS_PER_SEQ = 40     # How deep into the thought process we dig

LOG_FILE = "llama_8b_colar_fracture_log.txt"

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

# --- LOAD GSM8K ---
print("Fetching the GSM8K grimoire...")
dataset = load_dataset("gsm8k", "main", split="train")

with open(LOG_FILE, "w", encoding="utf-8") as f:
    emit(f"🌌 THE LATENT SEISMOGRAPH: HUNTING COLAR FRACTURES 🌌", f)
    emit(f"Scanning {NUM_SEQUENCES} reasoning sequences from GSM8K.", f)
    emit(f"Failure Triggers: Rank > {RANK_THRESHOLD} OR Solver Sim Advantage > {SIM_GAP_THRESHOLD}\n", f)

    total_fractures = 0
    total_steps_analyzed = 0

    with torch.no_grad():
        for seq_idx in range(NUM_SEQUENCES):
            question = dataset[seq_idx]['question']
            
            # Format as a standard instruct prompt
            prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            
            emit(f"\n{'='*100}", f)
            emit(f"📜 SEQUENCE {seq_idx + 1} | QUESTION: {question[:80]}...", f)
            emit(f"{'='*100}", f)

            for step in range(MAX_STEPS_PER_SEQ):
                total_steps_analyzed += 1
                
                # 1. The Natural Pulse
                outputs = model(input_ids=input_ids)
                next_token_logits = outputs.logits[0, -1, :].to(torch.float32)
                probs = F.softmax(next_token_logits, dim=-1)

                top_k_probs, top_k_ids = torch.topk(probs, K)
                top_1_id = top_k_ids[0].item()
                top_1_word = tokenizer.decode([top_1_id]).replace("\n", "\\n").replace("\r", "\\r")

                # Advance the timeline naturally
                input_ids = torch.cat([input_ids, top_k_ids[0:1].unsqueeze(0)], dim=1)

                # --- FORGING ---
                top_k_raw_embs = raw_embeddings[top_k_ids]
                avg_target_mag = torch.norm(top_k_raw_embs, p=2, dim=1).mean().item()

                adj_probs = top_k_probs ** (1.0 / TEMPERATURE)
                adj_probs = adj_probs / adj_probs.sum()

                # Tempered Chassis for the Solver
                temp_soft_raw = torch.sum(top_k_raw_embs * adj_probs.unsqueeze(1), dim=0, keepdim=True)
                temp_soft_norm = F.normalize(temp_soft_raw, p=2, dim=1)

                # A. CoLaR Geometry (Unweighted sum scaled by sqrt(K))
                v_colar = torch.sum(top_k_raw_embs, dim=0, keepdim=True) / math.sqrt(K)
                colar_unit = F.normalize(v_colar, p=2, dim=1)

                # B. Solver Geometry
                target_embs_norm = norm_dictionary[top_k_ids]
                with torch.enable_grad():
                    v_solver = fast_geometric_solver(
                        target_embs_norm, top_k_ids, norm_dictionary, temp_soft_norm, avg_target_mag, adj_probs
                    )
                solver_unit = F.normalize(v_solver, p=2, dim=1)

                # --- THE INSPECTION ---
                # Scan the dictionary to find the physical rank and similarity of the #1 predicted token
                colar_sims = torch.matmul(colar_unit, norm_dictionary.T).squeeze(0)
                solver_sims = torch.matmul(solver_unit, norm_dictionary.T).squeeze(0)

                colar_top1_sim = colar_sims[top_1_id].item()
                solver_top1_sim = solver_sims[top_1_id].item()

                # Calculate literal geometric rank (how many dictionary tokens have a higher similarity?)
                colar_top1_rank = (colar_sims > colar_top1_sim).sum().item() + 1
                solver_top1_rank = (solver_sims > solver_top1_sim).sum().item() + 1

                # --- THE TRIGGERS ---
                rank_failure = colar_top1_rank > RANK_THRESHOLD
                sim_failure = (solver_top1_sim - colar_top1_sim) > SIM_GAP_THRESHOLD

                if rank_failure or sim_failure:
                    total_fractures += 1
                    reason = "RANK FRACTURE" if rank_failure else "SIGNAL FADE"
                    if rank_failure and sim_failure:
                        reason = "TOTAL COLLAPSE (Rank & Signal)"

                    emit(f"\n  ⚠️ [FRACTURE DETECTED] | Seq: {seq_idx + 1} | Token Step: {step + 1} | Target: '{top_1_word}'", f)
                    emit(f"      Type: {reason}", f)
                    emit(f"      CoLaR  -> Sim: {colar_top1_sim:.4f} | Physical Rank: {colar_top1_rank}", f)
                    emit(f"      Solver -> Sim: {solver_top1_sim:.4f} | Physical Rank: {solver_top1_rank}", f)

    # --- THE FINAL LEDGER ---
    emit(f"\n{'='*100}", f)
    emit(f"🏁 DIAGNOSTIC COMPLETE", f)
    emit(f"{'='*100}", f)
    emit(f"Total Steps Analyzed:  {total_steps_analyzed}", f)
    emit(f"Total CoLaR Fractures: {total_fractures}", f)
    failure_rate = (total_fractures / total_steps_analyzed) * 100 if total_steps_analyzed > 0 else 0
    emit(f"Instability Rate:      {failure_rate:.2f}%", f)

print(f"\nThe seismograph has stopped recording. Autopsy saved to {LOG_FILE}.")
