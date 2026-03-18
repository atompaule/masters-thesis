import itertools
import math
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc

# --- THE TETHER ---
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
PROMPT = "The fundamental difference between human consciousness and artificial intelligence is"
TEMPERATURE = 2.0
K = 10
NUM_SAMPLES = 50
LOG_FILE = "llama_8b_hyperparameter_sweep.txt"


def emit(text, file_handle=None):
    print(text)
    if file_handle:
        file_handle.write(text + "\n")


print(f"Awakening the Leviathan for Sample Harvesting: {MODEL_ID}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16
)

device = model.device

gc.collect()
torch.cuda.empty_cache()

# --- EMBEDDINGS ---
raw_embeddings = model.get_input_embeddings().weight.detach().to(torch.float32)
norm_dictionary = F.normalize(raw_embeddings, p=2, dim=1)

# --- SAMPLE HARVEST ---
print("Harvesting live cognitive samples...")

samples = []
input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(device)

with torch.no_grad():

    for _ in range(NUM_SAMPLES):

        outputs = model(input_ids=input_ids)
        next_token_logits = outputs.logits[0, -1, :].to(torch.float32)

        probs = F.softmax(next_token_logits, dim=-1)

        top_k_probs, top_k_ids = torch.topk(probs, K)

        input_ids = torch.cat(
            [input_ids, top_k_ids[0:1].unsqueeze(0)],
            dim=1
        )

        adj_probs = top_k_probs ** (1.0 / TEMPERATURE)
        adj_probs = adj_probs / adj_probs.sum()

        target_embs_norm = norm_dictionary[top_k_ids]

        top_k_raw_embs = raw_embeddings[top_k_ids]

        avg_target_mag = torch.norm(
            top_k_raw_embs,
            p=2,
            dim=1
        ).mean().item()

        colar_raw = torch.sum(top_k_raw_embs, dim=0, keepdim=True) / math.sqrt(K)
        colar_dir = F.normalize(colar_raw, p=2, dim=1)

        soft_raw = torch.sum(
            top_k_raw_embs * adj_probs.unsqueeze(1),
            dim=0,
            keepdim=True
        )
        soft_dir = F.normalize(soft_raw, p=2, dim=1)

        top_1_dir = target_embs_norm[0:1]

        samples.append({
            "target_ids": top_k_ids.tolist(),
            "target_embs_norm": target_embs_norm,
            "colar_dir": colar_dir,
            "soft_dir": soft_dir,
            "top_1_dir": top_1_dir,
            "avg_target_mag": avg_target_mag,
            "adj_probs": adj_probs
        })


# --- SOLVER ---
def fast_geometric_solver_sweep(
    target_embs_norm,
    target_ids,
    dict_norm,
    sample,
    lr,
    epochs,
    num_interlopers,
    init_strategy
):

    k = len(target_ids)

    if init_strategy == "colar":
        start_vec = sample["colar_dir"]

    elif init_strategy == "soft":
        start_vec = sample["soft_dir"]

    elif init_strategy == "top_1":
        start_vec = sample["top_1_dir"]

    elif init_strategy == "random":
        start_vec = F.normalize(
            torch.randn(
                1,
                dict_norm.shape[1],
                device=dict_norm.device
            ),
            p=2,
            dim=1
        )

    with torch.no_grad():

        initial_sims = torch.matmul(
            sample["colar_dir"],
            dict_norm.T
        ).squeeze(0)

        _, danger_zone_idxs = torch.topk(initial_sims, num_interlopers)

        danger_mask = ~torch.isin(
            danger_zone_idxs,
            torch.tensor(target_ids, device=dict_norm.device)
        )

        local_interlopers = dict_norm[danger_zone_idxs[danger_mask]]

    probe = torch.nn.Parameter(start_vec.clone().squeeze(0))

    optimizer = optim.Adam([probe], lr=lr)

    for _ in range(epochs):

        optimizer.zero_grad()

        probe_n = F.normalize(probe.unsqueeze(0), p=2, dim=1)

        t_sims = torch.matmul(
            probe_n,
            target_embs_norm.T
        ).squeeze(0)

        pull_loss = -torch.sum(
            t_sims * sample["adj_probs"]
        ) * 2.0

        sim_diffs = t_sims[1:] - t_sims[:-1]

        p_gaps = (
            sample["adj_probs"][:-1]
            - sample["adj_probs"][1:]
        )

        margins = p_gaps * 0.1

        ranking_loss = torch.sum(
            F.relu(sim_diffs + margins)
        )

        i_sims = torch.matmul(
            probe_n,
            local_interlopers.T
        ).squeeze(0)

        hard_negs, _ = torch.topk(i_sims, k)

        push_loss = F.relu(
            hard_negs.mean()
            - t_sims.min()
            + 0.05
        ) * 1.5

        loss = (
            pull_loss
            + ranking_loss * 3.0
            + push_loss
        )

        loss.backward()
        optimizer.step()

    return (
        F.normalize(probe.unsqueeze(0), p=2, dim=1)
        .detach()
        * sample["avg_target_mag"]
    )


# --- EVALUATION ---
def evaluate_vector(vector, target_embs_norm):

    unit_vector = F.normalize(vector, p=2, dim=1)

    sims = torch.matmul(
        unit_vector,
        target_embs_norm.T
    ).squeeze(0)

    _, sorted_indices = torch.sort(
        sims,
        descending=True
    )

    correct_positions = (
        sorted_indices
        == torch.arange(K, device=device)
    ).sum().item()

    return sims.cpu().numpy(), correct_positions


def interloper_similarity(
    vector,
    target_ids,
    dict_norm
):

    unit_vector = F.normalize(vector, p=2, dim=1)

    sims = torch.matmul(
        unit_vector,
        dict_norm.T
    ).squeeze(0)

    mask = torch.ones_like(
        sims,
        dtype=torch.bool
    )

    mask[torch.tensor(target_ids, device=sims.device)] = False

    sims = sims[mask]

    top_negs, _ = torch.topk(sims, K)

    return top_negs.mean().item()


# --- SWEEP GRID ---
init_strategies = ["colar"]
learning_rates = [0.1]
epochs_list = [300]
interlopers_list = [1000]

grid = list(
    itertools.product(
        init_strategies,
        learning_rates,
        epochs_list,
        interlopers_list
    )
)

pd.set_option("display.float_format", "{:.4f}".format)
pd.set_option("display.width", 1000)


with open(LOG_FILE, "w", encoding="utf-8") as f:

    emit("🌌 THE HYPER-CALIBRATION MATRIX 🌌\n", f)

    # --- COLAR BASELINE ---
    colar_sim_sums = [0.0] * K
    colar_total_rank = 0
    colar_total_sim_sum = 0.0
    colar_total_interloper = 0.0

    for sample in samples:

        sims, rank_score = evaluate_vector(
            sample["colar_dir"],
            sample["target_embs_norm"]
        )

        colar_total_rank += rank_score

        colar_total_sim_sum += sims.sum()

        inter_sim = interloper_similarity(
            sample["colar_dir"],
            sample["target_ids"],
            norm_dictionary
        )

        colar_total_interloper += inter_sim

        for i in range(K):
            colar_sim_sums[i] += sims[i]

    colar_avg_rank = colar_total_rank / NUM_SAMPLES
    colar_avg_sim_sum = colar_total_sim_sum / NUM_SAMPLES
    colar_avg_interloper = colar_total_interloper / NUM_SAMPLES

    emit("=== COLAR BASELINE (UNOPTIMIZED) ===", f)
    emit(f"Average Rank Score: {colar_avg_rank:.1f} / {K}", f)
    emit(f"Average Sum of Target Similarities: {colar_avg_sim_sum:.4f}", f)
    emit(f"Average Interloper Similarity: {colar_avg_interloper:.4f}", f)

    emit("Average Similarities per Rank:", f)

    colar_sim_str = " | ".join(
        [f"R{i+1}: {colar_sim_sums[i]/NUM_SAMPLES:.3f}" for i in range(K)]
    )

    emit(f"{colar_sim_str}\n", f)

    # --- TABLE HEADER ---
    emit(
        f"{'Init':<8} | {'LR':<6} | {'Epochs':<6} | {'Moat Size':<9} | "
        f"{'Rank Score':<11} | {'Sum Sims':<8} | {'Interloper':<10} | "
        f"{'R1 Sim':<7} | {'R2 Sim':<7} | {'R10 Sim':<7}",
        f
    )

    emit("-" * 110, f)

    # --- SWEEP ---
    for init_strat, lr, epochs, interlopers in grid:

        sweep_sim_sums = [0.0] * K
        sweep_total_rank = 0
        sweep_total_sim_sum = 0.0
        sweep_total_interloper = 0.0

        for sample in samples:

            solver_vec = fast_geometric_solver_sweep(
                sample["target_embs_norm"],
                sample["target_ids"],
                norm_dictionary,
                sample,
                lr,
                epochs,
                interlopers,
                init_strat
            )

            sims, rank_score = evaluate_vector(
                solver_vec,
                sample["target_embs_norm"]
            )

            sweep_total_rank += rank_score

            sweep_total_sim_sum += sims.sum()

            inter_sim = interloper_similarity(
                solver_vec,
                sample["target_ids"],
                norm_dictionary
            )

            sweep_total_interloper += inter_sim

            for i in range(K):
                sweep_sim_sums[i] += sims[i]

        avg_rank = sweep_total_rank / NUM_SAMPLES
        avg_sim_sum = sweep_total_sim_sum / NUM_SAMPLES
        avg_interloper = sweep_total_interloper / NUM_SAMPLES

        r1_avg = sweep_sim_sums[0] / NUM_SAMPLES
        r2_avg = sweep_sim_sums[1] / NUM_SAMPLES
        r10_avg = sweep_sim_sums[9] / NUM_SAMPLES

        emit(
            f"{init_strat:<8} | {lr:<6.3f} | {epochs:<6} | {interlopers:<9} | "
            f"{avg_rank:>4.1f} / {K:<4} | {avg_sim_sum:<8.3f} | {avg_interloper:<10.3f} | "
            f"{r1_avg:<7.3f} | {r2_avg:<7.3f} | {r10_avg:<7.3f}",
            f
        )

emit(f"\nSweep complete. Calibration logs saved to {LOG_FILE}.")
