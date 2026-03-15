import math
import torch
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoTokenizer
from datasets import load_dataset
import gc
import json
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


# ------------------------------------------------
# CONFIG
# ------------------------------------------------

MODEL_ID = "meta-llama/Llama-3.1-405B-Instruct"
TEMPERATURE = 3.0
K = 10
NUM_STEPS = 200
WORST_N = 20
LOG_FILE = "llama_405b_fracture_gallery.txt"


gc.collect()
torch.cuda.empty_cache()


def emit(text, file_handle=None):
    print(text)
    if file_handle:
        file_handle.write(text + "\n")


# ------------------------------------------------
# TOKENIZER
# ------------------------------------------------

print(f"Initializing tokenizer for: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------
# LOAD 405B EMBEDDINGS ONLY
# ------------------------------------------------

print("Downloading embedding matrix...")

index_path = hf_hub_download(
    repo_id=MODEL_ID,
    filename="model.safetensors.index.json"
)

with open(index_path) as f:
    index_data = json.load(f)

target_tensor = "model.embed_tokens.weight"
target_file = index_data["weight_map"][target_tensor]

tensor_path = hf_hub_download(repo_id=MODEL_ID, filename=target_file)

tensors = load_file(tensor_path)

raw_embeddings = tensors[target_tensor].to(torch.float32).to(device)

norm_dictionary = F.normalize(raw_embeddings, p=2, dim=1).to(torch.bfloat16)


# ------------------------------------------------
# SOLVER
# ------------------------------------------------

def fast_geometric_solver(target_embs_norm, target_ids, dict_norm, magnitude_target, adj_probs):

    k = len(target_ids)

    colar_dir = F.normalize(torch.sum(target_embs_norm, dim=0, keepdim=True), p=2, dim=1)

    with torch.no_grad():

        initial_sims = torch.matmul(colar_dir, dict_norm.float().T).squeeze(0)

        _, danger_zone_idxs = torch.topk(initial_sims, 1000)

        danger_mask = ~torch.isin(danger_zone_idxs, torch.tensor(target_ids, device=device))

        local_interlopers = dict_norm.float()[danger_zone_idxs[danger_mask]]

    probe = torch.nn.Parameter(colar_dir.clone().float().squeeze(0))

    optimizer = optim.Adam([probe], lr=0.05)

    target_embs_float = target_embs_norm.float()

    for _ in range(150):

        optimizer.zero_grad()

        probe_n = F.normalize(probe.unsqueeze(0), p=2, dim=1)

        t_sims = torch.matmul(probe_n, target_embs_float.T).squeeze(0)

        pull_loss = -torch.sum(t_sims * adj_probs) * 5.0

        sim_diffs = t_sims[1:] - t_sims[:-1]

        p_gaps = adj_probs[:-1] - adj_probs[1:]

        margins = p_gaps * 0.1

        ranking_loss = torch.sum(F.relu(sim_diffs + margins))

        i_sims = torch.matmul(probe_n, local_interlopers.T).squeeze(0)

        hard_negs, _ = torch.topk(i_sims, k)

        push_loss = F.relu(hard_negs.mean() - t_sims.min() + 0.05) * 1.5

        loss = pull_loss + ranking_loss * 3.0 + push_loss

        loss.backward()

        optimizer.step()

    return (F.normalize(probe.unsqueeze(0), p=2, dim=1).detach() * magnitude_target).to(torch.bfloat16)


# ------------------------------------------------
# LOGIT EXTRACTION
# ------------------------------------------------

def extract_top_k_from_row(row, device, tokenizer, max_k=10):

    special_ids = torch.tensor(tokenizer.all_special_ids, device=device)

    if "token_ids" in row and "top_values" in row:

        raw_logits = torch.tensor(row["top_values"], dtype=torch.float32, device=device)
        top_k_ids = torch.tensor(row["token_ids"], dtype=torch.long, device=device)

        if raw_logits.dim() > 1:
            raw_logits, top_k_ids = raw_logits[-1], top_k_ids[-1]

        mask = ~torch.isin(top_k_ids, special_ids)

        raw_logits = raw_logits[mask]
        top_k_ids = top_k_ids[mask]

        actual_k = min(max_k, raw_logits.shape[-1])

        probs = F.softmax(raw_logits[:actual_k], dim=-1)

        return probs, top_k_ids[:actual_k]

    return None, None


# ------------------------------------------------
# HUNT
# ------------------------------------------------

print("Connecting to 405B logit stream...")

dataset = load_dataset(
    "arcee-ai/LLama-405B-Logits",
    split="train",
    streaming=True
)

timeline_data = []

with torch.no_grad():

    for step, row in enumerate(dataset):

        if step >= NUM_STEPS:
            break

        if step % 10 == 0:
            print(f"Processing step {step}/{NUM_STEPS}")

        top_k_probs, top_k_ids = extract_top_k_from_row(row, device, tokenizer, K)

        if top_k_probs is None:
            continue

        actual_k = len(top_k_ids)

        words = tokenizer.convert_ids_to_tokens(top_k_ids.tolist())

        prob_mass = top_k_probs.sum().item()

        p_gap = (top_k_probs[0] - top_k_probs[1]).item() if actual_k > 1 else 0

        normalized_p = top_k_probs / prob_mass

        local_entropy = -torch.sum(normalized_p * torch.log(normalized_p + 1e-9)).item()

        adj_probs = top_k_probs ** (1.0 / TEMPERATURE)
        adj_probs = adj_probs / adj_probs.sum()

        target_embs_norm = norm_dictionary[top_k_ids].float()

        top_k_raw_embs = raw_embeddings[top_k_ids].float()

        avg_target_mag = torch.norm(top_k_raw_embs, dim=1).mean().item()


        # ------------------------------------------------
        # FORGE VECTORS
        # ------------------------------------------------

        v_colar = torch.sum(top_k_raw_embs, dim=0, keepdim=True) / math.sqrt(actual_k)
        unit_colar = F.normalize(v_colar, p=2, dim=1).to(torch.bfloat16)

        soft_weights = adj_probs.unsqueeze(1)
        v_soft = torch.sum(top_k_raw_embs * soft_weights, dim=0, keepdim=True)
        unit_soft = F.normalize(v_soft, p=2, dim=1).to(torch.bfloat16)

        with torch.enable_grad():
            v_solver = fast_geometric_solver(
                target_embs_norm,
                top_k_ids,
                norm_dictionary,
                avg_target_mag,
                adj_probs
            )

        unit_solver = F.normalize(v_solver, p=2, dim=1).to(torch.bfloat16)


        # ------------------------------------------------
        # ANALYSIS
        # ------------------------------------------------

        colar_sims_all = torch.matmul(unit_colar, norm_dictionary.T).squeeze(0)
        soft_sims_all = torch.matmul(unit_soft, norm_dictionary.T).squeeze(0)
        solver_sims_all = torch.matmul(unit_solver, norm_dictionary.T).squeeze(0)


        step_info = {

            "step": step + 1,
            "actual_k": actual_k,
            "prob_mass": prob_mass,
            "p_gap": p_gap,
            "local_entropy": local_entropy,

            "words": [w.replace("Ġ", " ") for w in words],
            "raw_probs": top_k_probs.cpu().numpy(),

            "colar_phys_ranks": [],
            "soft_phys_ranks": [],
            "solver_phys_ranks": [],

            "colar_sims": [],
            "soft_sims": [],
            "solver_sims": []

        }


        for i in range(actual_k):

            t_id = top_k_ids[i].item()

            c_sim = colar_sims_all[t_id].item()
            s_sim = soft_sims_all[t_id].item()
            g_sim = solver_sims_all[t_id].item()

            c_rank = (colar_sims_all > c_sim).sum().item() + 1
            s_rank = (soft_sims_all > s_sim).sum().item() + 1
            g_rank = (solver_sims_all > g_sim).sum().item() + 1

            step_info["colar_phys_ranks"].append(c_rank)
            step_info["soft_phys_ranks"].append(s_rank)
            step_info["solver_phys_ranks"].append(g_rank)

            step_info["colar_sims"].append(c_sim)
            step_info["soft_sims"].append(s_sim)
            step_info["solver_sims"].append(g_sim)


        if step_info["words"][0].startswith("<|"):
            step_info["colar_failure"] = -1
            step_info["soft_failure"] = -1
            step_info["solver_failure"] = -1
        else:
            step_info["colar_failure"] = step_info["colar_phys_ranks"][0]
            step_info["soft_failure"] = step_info["soft_phys_ranks"][0]
            step_info["solver_failure"] = step_info["solver_phys_ranks"][0]

        timeline_data.append(step_info)


# ------------------------------------------------
# AUTOPSY
# ------------------------------------------------

colar_worst = sorted(timeline_data, key=lambda x: x["colar_failure"], reverse=True)[:WORST_N]
soft_worst = sorted(timeline_data, key=lambda x: x["soft_failure"], reverse=True)[:WORST_N]
solver_worst = sorted(timeline_data, key=lambda x: x["solver_failure"], reverse=True)[:WORST_N]


def print_gallery(title, cases, failure_key, file):

    emit("\n\n" + "#"*120, file)
    emit(title, file)
    emit("#"*120 + "\n", file)

    for rank, case in enumerate(cases):

        if case[failure_key] < 0:
            continue

        emit("="*120, file)

        emit(
            f"🚨 WORST CASE #{rank+1} | Step {case['step']} | Rank1 Displacement: {case[failure_key]}",
            file
        )

        emit(
            f"   ↳ K: {case['actual_k']} | Mass: {case['prob_mass']*100:.2f}% | Gap: {case['p_gap']*100:.2f}% | Entropy: {case['local_entropy']:.3f}",
            file
        )

        emit("-"*120, file)

        header = f"{'Rank & Token':<25} | {'Prob':<8} || {'CoLaR Rank(Sim)':<16} | {'Soft Rank(Sim)':<16} | {'Solver Rank(Sim)':<18}"

        emit(header, file)

        for i in range(case["actual_k"]):

            word = case["words"][i]
            prob = case["raw_probs"][i] * 100

            row = (
                f"{'-->' if i==0 else '   '} {i+1}. '{word[:15]:<15}' | {prob:>5.2f}%   || "
                f"{case['colar_phys_ranks'][i]:>5} ({case['colar_sims'][i]:.3f}) | "
                f"{case['soft_phys_ranks'][i]:>5} ({case['soft_sims'][i]:.3f}) | "
                f"{case['solver_phys_ranks'][i]:>5} ({case['solver_sims'][i]:.3f})"
            )

            emit(row, file)

        emit("\n", file)


with open(LOG_FILE, "w", encoding="utf-8") as f:

    emit("🌌 THE 405B TITAN FRACTURE GALLERY 🌌", f)

    print_gallery("🔥 COLAR WORST CASES 🔥", colar_worst, "colar_failure", f)
    print_gallery("🌊 SOFT WORST CASES 🌊", soft_worst, "soft_failure", f)
    print_gallery("🧠 SOLVER WORST CASES 🧠", solver_worst, "solver_failure", f)


print(f"\nAutopsy complete → {LOG_FILE}")
