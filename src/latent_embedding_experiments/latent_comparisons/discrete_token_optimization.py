import math
from dataclasses import dataclass

import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer

# =========================================================
# CONFIG
# =========================================================


@dataclass
class Config:
    model_id: str = "meta-llama/Llama-3.1-8B-Instruct"
    display_k: int = 15
    log_file: str = (
        "src/latent_embedding_experiments/logs/llama_8b_discrete_token_optimization.txt"
    )

    solver_steps: int = 300
    lr: float = 0.05
    danger_topk: int = 2000  # The radius of synonyms we intend to burn away


CFG = Config()


# =========================================================
# UTIL
# =========================================================


def emit(text, f=None):
    print(text)
    if f:
        f.write(text + "\n")


def clean_token(token: str, max_len=14):
    token = token.replace("\n", "\\n").replace("\r", "\\r")
    return token[:max_len] + ".." if len(token) > max_len else token


def format_cell(token, value, width=32, marker=False):
    prefix = "*" if marker else " "
    content = f"{prefix}'{token}' ({value:.3f})"
    return f"{content:<{width}}"


# =========================================================
# MODEL
# =========================================================

print(f"Awakening model: {CFG.model_id}")
tokenizer = AutoTokenizer.from_pretrained(CFG.model_id)
model = AutoModelForCausalLM.from_pretrained(
    CFG.model_id, device_map="auto", torch_dtype=torch.bfloat16
)

device = model.device

raw_embeddings = model.get_input_embeddings().weight.detach().to(torch.float32)
norm_dictionary = F.normalize(raw_embeddings, dim=1)


# =========================================================
# THE ISOLATION FORGE
# =========================================================


def isolate_discrete_token(target_id, dict_norm, magnitude):
    target_norm = dict_norm[target_id].unsqueeze(0)

    # 1. Identify the halo of interlopers to repel
    with torch.no_grad():
        sims = target_norm @ dict_norm.T
        _, idxs = torch.topk(sims.squeeze(0), CFG.danger_topk + 1)

        # Remove the target itself from the interloper list
        mask = idxs != target_id
        interlopers = dict_norm[idxs[mask]]

    # 2. Plant the seed exactly on the target
    probe = torch.nn.Parameter(target_norm.clone().squeeze(0))
    opt = optim.Adam([probe], lr=CFG.lr)

    for _ in range(CFG.solver_steps):
        opt.zero_grad()

        p = F.normalize(probe.unsqueeze(0), dim=1)

        # The Anchor: How close are we to the true target?
        sim_target = (p @ target_norm.T).squeeze()
        pull = (1.0 - sim_target) * 2.5

        # The Repulsion: How close are we to the hardest synonyms?
        sims_i = (p @ interlopers.T).squeeze(0)
        hard, _ = torch.topk(sims_i, 20)
        push = hard.mean() * 1.5

        loss = pull + push
        loss.backward()
        opt.step()

    return F.normalize(probe.unsqueeze(0), dim=1) * magnitude


# =========================================================
# TABLE RENDERING
# =========================================================


def render_comparison_table(v_orig, v_isolated, target_id, title, f):

    emit(f"\n--- {title} ---", f)

    vectors_unit = F.normalize(torch.cat([v_orig, v_isolated], dim=0), dim=1)
    sims = vectors_unit @ norm_dictionary.T

    labels = ["Original Token", "Isolated (Sharpened) Token"]
    col_width = 32

    header = f"{'Rank':<4} |"
    for label in labels:
        header += f" {label:<{col_width}} |"

    emit(header[:-2], f)
    emit("-" * len(header), f)

    vals, idxs = torch.topk(sims, CFG.display_k, dim=1)

    for r in range(CFG.display_k):
        row = f"{r+1:<4} |"
        for i in range(len(labels)):
            tid = idxs[i, r].item()
            token = clean_token(tokenizer.decode([tid]))
            marker = tid == target_id
            row += " " + format_cell(token, vals[i, r].item(), col_width, marker) + " |"
        emit(row[:-2], f)


# =========================================================
# EXPERIMENT LOOP
# =========================================================

pd.set_option("display.float_format", "{:.4f}".format)
pd.set_option("display.width", 1000)

# Tokens selected for having dense, highly populated synonym halos
TEST_TOKENS = [" water", " king", " beautiful", " paradox", " computer"]

with open(CFG.log_file, "w", encoding="utf-8") as f:

    emit("🌌 DISCRETE TOKEN ISOLATION LAB 🌌", f)
    emit("Mode: Eroding semantic halos while anchoring to the root geometry.\n", f)

    for word in TEST_TOKENS:

        emit("\n" + "=" * 100, f)
        emit(f"🔬 ISOLATING TOKEN: '{word}'", f)
        emit("=" * 100, f)

        # Encode
        t_ids = tokenizer.encode(word, add_special_tokens=False)
        if len(t_ids) > 1:
            emit(f"Skipping '{word}' (fragmented into multiple tokens).", f)
            continue

        target_id = t_ids[0]
        v_orig = raw_embeddings[target_id].unsqueeze(0)
        magnitude = torch.norm(v_orig, p=2, dim=1).item()

        # Forge
        with torch.enable_grad():
            v_isolated = isolate_discrete_token(target_id, norm_dictionary, magnitude)

        # Report Geometry
        orig_unit = F.normalize(v_orig, dim=1)
        iso_unit = F.normalize(v_isolated, dim=1)

        sim_to_self = (orig_unit @ iso_unit.T).item()

        emit(f"\n[Geometry Diagnostics]", f)
        emit(f"Target Token:       {target_id}", f)
        emit(f"Original Magnitude: {magnitude:.4f}", f)
        emit(f"Isolated Magnitude: {torch.norm(v_isolated, p=2).item():.4f}", f)
        emit(
            f"Anchor Fidelity:    {sim_to_self:.4f} (Cosine similarity between Original and Isolated)",
            f,
        )

        # Tables
        render_comparison_table(
            v_orig,
            v_isolated,
            target_id,
            "Nearest Neighbor Shift (Cosine Similarity)",
            f,
        )

    emit("\nIsolation sequence complete.", f)

print(f"\nAutopsy logged to → {CFG.log_file}")
