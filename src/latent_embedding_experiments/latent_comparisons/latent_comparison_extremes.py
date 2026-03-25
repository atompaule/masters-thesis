import gc
import json
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoTokenizer

from src.latent_embedding_experiments.algorithms.solver import geometric_solver

# =========================================================
# CONFIG
# =========================================================


@dataclass
class Config:
    model_id: str = "meta-llama/Llama-3.1-405B-Instruct"
    temperature: float = 3.0
    k: int = 10
    num_steps: int = 100
    worst_n: int = 20
    log_file: str = (
        "src/latent_embedding_experiments/logs/llama_405b_latent_comparison_extremes.txt"
    )

    solver_steps: int = 300
    lr: float = 0.05
    danger_topk: int = 2000


CFG = Config()

gc.collect()
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# UTIL
# =========================================================


def emit(text, f=None):
    print(text)
    if f:
        f.write(text + "\n")


# =========================================================
# TOKENIZER
# =========================================================

print(f"Initializing tokenizer for: {CFG.model_id}")
tokenizer = AutoTokenizer.from_pretrained(CFG.model_id)


# =========================================================
# LOAD EMBEDDINGS
# =========================================================


def load_embeddings(model_id):
    print("Downloading embedding matrix...")

    index_path = hf_hub_download(
        repo_id=model_id, filename="model.safetensors.index.json"
    )

    with open(index_path) as f:
        index_data = json.load(f)

    tensor_name = "model.embed_tokens.weight"
    tensor_file = index_data["weight_map"][tensor_name]

    tensor_path = hf_hub_download(repo_id=model_id, filename=tensor_file)
    tensors = load_file(tensor_path)

    raw = tensors[tensor_name].to(torch.float32).to(device)

    return {
        "raw": raw,
        "norm": F.normalize(raw, dim=1).to(torch.float32),
    }


emb = load_embeddings(CFG.model_id)


# =========================================================
# EXTRACTION
# =========================================================


def extract_top_k(row, max_k):
    if "token_ids" not in row or "top_values" not in row:
        return None, None

    ids = torch.tensor(row["token_ids"], device=device)
    logits = torch.tensor(row["top_values"], device=device, dtype=torch.float32)

    if logits.dim() > 1:
        ids, logits = ids[-1], logits[-1]

    mask = ~torch.isin(ids, torch.tensor(tokenizer.all_special_ids, device=device))

    ids = ids[mask][:max_k]
    logits = logits[mask][:max_k]

    probs = F.softmax(logits, dim=-1)
    return probs, ids


# =========================================================
# VECTOR BUILDING
# =========================================================


def build_vectors(ids, probs):
    raw = emb["raw"][ids]

    k = len(ids)

    v_colar = raw.sum(dim=0, keepdim=True) / math.sqrt(k)
    v_soft = (raw * probs.unsqueeze(1)).sum(dim=0, keepdim=True)

    avg_target_mag = torch.norm(raw, p=2, dim=1).mean().item()
    target_norm = emb["norm"][ids]
    with torch.enable_grad():
        v_solver = geometric_solver(
            target_norm,
            ids.tolist(),
            emb["norm"],
            avg_target_mag,
            probs,
            temperature=CFG.temperature,
        )

    return v_colar, v_soft, v_solver


# =========================================================
# COSINE EVALUATION
# =========================================================


def evaluate_cosine(v, emb_norm):
    v_norm = F.normalize(v.float(), dim=1)
    return (v_norm @ emb_norm.T).squeeze(0)


def rank_of(val, arr):
    return (arr > val).sum().item() + 1


# =========================================================
# MAIN LOOP
# =========================================================

print("Connecting to dataset...")
dataset = load_dataset("arcee-ai/LLama-405B-Logits", split="train", streaming=True)

timeline = []

for step, row in enumerate(dataset):

    if step >= CFG.num_steps:
        break

    if step % 10 == 0:
        print(f"Step {step}/{CFG.num_steps}")

    probs, ids = extract_top_k(row, CFG.k)
    if probs is None:
        continue

    words = tokenizer.convert_ids_to_tokens(ids.tolist())
    words = [w.replace("Ġ", " ") for w in words]

    v_colar, v_soft, v_solver = build_vectors(ids, probs)

    colar_all = evaluate_cosine(v_colar, emb["norm"])
    soft_all = evaluate_cosine(v_soft, emb["norm"])
    solver_all = evaluate_cosine(v_solver, emb["norm"])

    case = {
        "step": step + 1,
        "words": words,
        "probs": probs.cpu().numpy(),
        "k": len(ids),
        "colar": [],
        "soft": [],
        "solver": [],
    }

    for i, tid in enumerate(ids):
        tid = tid.item()

        c = colar_all[tid].item()
        s = soft_all[tid].item()
        g = solver_all[tid].item()

        case["colar"].append((rank_of(c, colar_all), c))
        case["soft"].append((rank_of(s, soft_all), s))
        case["solver"].append((rank_of(g, solver_all), g))

    case["colar_failure"] = case["colar"][0][0]
    case["soft_failure"] = case["soft"][0][0]
    case["solver_failure"] = case["solver"][0][0]

    timeline.append(case)


# =========================================================
# REPORTING
# =========================================================


def worst(data, key):
    return sorted(data, key=lambda x: x[key], reverse=True)[: CFG.worst_n]


def format_row(rank, token, prob, c, s, g):
    return (
        f"{rank:>2}. {token:<18} | "
        f"{prob:>6.2f}% | "
        f"{c[0]:>6} ({c[1]:>6.3f}) | "
        f"{s[0]:>6} ({s[1]:>6.3f}) | "
        f"{g[0]:>6} ({g[1]:>6.3f})"
    )


def print_gallery(title, cases, key, f):

    emit("\n" + "=" * 110, f)
    emit(title, f)
    emit("=" * 110 + "\n", f)

    header = (
        f"{'Rank Token':<22} | {'Prob':<8} | "
        f"{'CoLaR Rank (Cos)':<20} | "
        f"{'Soft Rank (Cos)':<20} | "
        f"{'Solver Rank (Cos)':<20}"
    )

    for i, case in enumerate(cases):

        emit("-" * 110, f)
        emit(f"🚨 Case #{i+1} | Step {case['step']} | Failure Rank: {case[key]}", f)
        emit(header, f)
        emit("-" * 110, f)

        for j in range(case["k"]):
            row = format_row(
                j + 1,
                case["words"][j][:18],
                case["probs"][j] * 100,
                case["colar"][j],
                case["soft"][j],
                case["solver"][j],
            )
            emit(row, f)

        emit("\n", f)


with open(CFG.log_file, "w", encoding="utf-8") as f:

    emit("🌌 FRACTURE GALLERY — COSINE SPACE 🌌", f)

    print_gallery(
        "🔥 CoLaR Worst Cases 🔥", worst(timeline, "colar_failure"), "colar_failure", f
    )
    print_gallery(
        "🌊 Soft Worst Cases 🌊", worst(timeline, "soft_failure"), "soft_failure", f
    )
    print_gallery(
        "🧠 Solver Worst Cases 🧠",
        worst(timeline, "solver_failure"),
        "solver_failure",
        f,
    )


print(f"\nDone → {CFG.log_file}")
