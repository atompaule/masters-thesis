import math
from dataclasses import dataclass

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.latent_embedding_experiments.algorithms.config import CFG
from src.latent_embedding_experiments.algorithms.soft_thinking import soft_thinking
from src.latent_embedding_experiments.algorithms.solver import geometric_solver

# =========================================================
# CONFIG
# =========================================================


@dataclass
class ExperimentConfig:
    model_id: str = "meta-llama/Llama-3.1-8B-Instruct"
    display_k: int = 20
    log_file: str = (
        "src/latent_embedding_experiments/logs/llama_8b_latent_comparison_concept_descriptions.txt"
    )


EXP_CFG = ExperimentConfig()


# =========================================================
# UTIL
# =========================================================


def emit(text, f=None):
    print(text)
    if f:
        f.write(text + "\n")


def clean_token(token: str, max_len=12):
    token = token.replace("\n", "\\n").replace("\r", "\\r")
    return token[:max_len] + ".." if len(token) > max_len else token


def format_cell(token, value, width=28, marker=False):
    prefix = "*" if marker else " "
    content = f"{prefix}'{token}' ({value:.3f})"
    return f"{content:<{width}}"


# =========================================================
# MODEL
# =========================================================

print(f"Loading model: {EXP_CFG.model_id}")
tokenizer = AutoTokenizer.from_pretrained(EXP_CFG.model_id)
model = AutoModelForCausalLM.from_pretrained(
    EXP_CFG.model_id, device_map="auto", torch_dtype=torch.bfloat16
)

device = model.device

vocab_embs = model.get_input_embeddings().weight.detach().to(torch.float32)
vocab_embs_norm = F.normalize(vocab_embs, dim=1)


# =========================================================
# VECTOR FORGING
# =========================================================


def build_fake_logits(target_ids, logit_values, vocab_size):
    """
    Build a full-vocabulary logit vector from handcrafted target logits.
    All non-target tokens get a very low logit so top-p will select only targets.
    """
    full_logits = torch.full((vocab_size,), -100.0, device=device)
    for tid, lv in zip(target_ids, logit_values):
        full_logits[tid] = lv
    return full_logits


def build_vectors(target_ids, logit_values):
    """
    Build all embedding variants from handcrafted target tokens and logits.
    Uses the same functions as the autoregressive comparison script.
    """
    vocab_size = vocab_embs.size(0)
    full_logits = build_fake_logits(target_ids, logit_values, vocab_size)

    # Soft thinking (top-p, probability-weighted sum)
    v_soft = soft_thinking(full_logits, vocab_embs)

    # Geometric solver (top-p, ranking-preserving)
    with torch.enable_grad():
        v_solver = geometric_solver(full_logits, vocab_embs)

    # CoLaR-style centroid (unweighted, scaled by 1/sqrt(k))
    raw_embs = vocab_embs[target_ids]
    v_colar = raw_embs.sum(dim=0, keepdim=True) / math.sqrt(len(target_ids))

    # Simple centroid (unweighted mean)
    v_centroid = raw_embs.mean(dim=0, keepdim=True)

    return v_soft, v_solver, v_colar, v_centroid


# =========================================================
# TABLE RENDERING
# =========================================================


def render_similarity_table(vectors, labels, target_ids_set, title, f):
    emit(f"\n--- {title} — * = target token ---", f)

    vectors_unit = F.normalize(vectors, dim=1)
    sims = vectors_unit @ vocab_embs_norm.T

    col_width = 30

    header = f"{'Rank':<4} |"
    for label in labels:
        header += f" {label:<{col_width}} |"
    emit(header[:-2], f)
    emit("-" * len(header), f)

    vals, idxs = torch.topk(sims, EXP_CFG.display_k, dim=1)

    for r in range(EXP_CFG.display_k):
        row = f"{r+1:<4} |"
        for i in range(len(labels)):
            tid = idxs[i, r].item()
            token = clean_token(tokenizer.decode([tid]))
            marker = tid in target_ids_set
            row += " " + format_cell(token, vals[i, r].item(), col_width, marker) + " |"
        emit(row[:-2], f)


def render_dot_table(vectors, labels, target_ids_set, title, f):
    emit(f"\n--- {title} — * = target token ---", f)

    vecs = vectors.to(torch.float32)
    emb = vocab_embs.to(torch.float32)
    dots = vecs @ emb.T

    col_width = 30

    header = f"{'Rank':<4} |"
    for label in labels:
        header += f" {label:<{col_width}} |"
    emit(header[:-2], f)
    emit("-" * len(header), f)

    vals, idxs = torch.topk(dots, EXP_CFG.display_k, dim=1)

    for r in range(EXP_CFG.display_k):
        row = f"{r+1:<4} |"
        for i in range(len(labels)):
            tid = idxs[i, r].item()
            token = clean_token(tokenizer.decode([tid]))
            marker = tid in target_ids_set
            row += " " + format_cell(token, vals[i, r].item(), col_width, marker) + " |"
        emit(row[:-2], f)


# =========================================================
# GENERATION
# =========================================================


def splice_and_evaluate(prefix, suffix, vec):
    prefix_ids = tokenizer(prefix, return_tensors="pt").input_ids.to(device)
    suffix_ids = tokenizer(
        suffix, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(device)

    prefix_emb = model.get_input_embeddings()(prefix_ids)
    suffix_emb = model.get_input_embeddings()(suffix_ids)

    vec = vec.unsqueeze(0).to(model.dtype)
    inputs = torch.cat([prefix_emb, vec, suffix_emb], dim=1)

    with torch.no_grad():
        out = model.generate(
            inputs_embeds=inputs,
            max_new_tokens=200,
            temperature=0.01,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(out.sequences[0], skip_special_tokens=True)

    logits = out.scores[0][0].float()
    probs = F.softmax(logits, dim=-1)
    top_p, top_i = torch.topk(probs, 5)

    lines = []
    for i, (tid, p) in enumerate(zip(top_i, top_p)):
        token = clean_token(tokenizer.decode([tid]))
        lines.append(f"      {i+1}. {token:<12} | {p.item()*100:>5.2f}%")

    return text.strip(), lines


# =========================================================
# EXPERIMENTS
# =========================================================

pd.set_option("display.float_format", "{:.4f}".format)
pd.set_option("display.width", 1000)

experiments = [
    {
        "name": (
            "Real top-10 (8B): after '...consciousness and artificial intelligence is' "
            "(llama_8b_latent_comparison_sequence_discrete_top1.txt)"
        ),
        "words": [
            " ability",
            " way",
            " presence",
            " nature",
            " human",
            " capacity",
            " former",
            " subjective",
            " source",
            " complexity",
        ],
        "logits": [
            17.0,
            16.076,
            15.957,
            15.894,
            15.623,
            15.536,
            15.442,
            15.402,
            15.268,
            15.106,
        ],
        "prefix": "Instruction: Vividly describe the concept represented here.\n\nConcept: ",
        "suffix": "\n\nDescription: ",
    },
    {
        "name": "The Sapphire Summit",
        "words": [" mountain", " galaxy", " snow", " peak", " wind"],
        "logits": [16.0, 15.5, 15.0, 14.5, 14.0],
        "prefix": "Instruction: Vividly describe the concept represented here.\n\nConcept: ",
        "suffix": "\n\nDescription: ",
    },
    {
        "name": "City on top — excitement, inspiration, and related uplift",
        "words": [" city", " excitement", " inspiration", " bright", " hope"],
        "logits": [16.0, 15.5, 15.0, 14.5, 14.0],
        "prefix": "Instruction: Vividly describe the concept represented here.\n\nConcept: ",
        "suffix": "\n\nDescription: ",
    },
    {
        "name": "City on top — dark, depressed, and related low-affect tone",
        "words": [" city", " dark", " depressed", " bleak", " sad"],
        "logits": [16.0, 15.5, 15.0, 14.5, 14.0],
        "prefix": "Instruction: Vividly describe the concept represented here.\n\nConcept: ",
        "suffix": "\n\nDescription: ",
    },
    {
        "name": "The Overgrown Relic",
        "words": [" forest", " magic", " ancient", " green", " magic"],
        "logits": [16.0, 15.5, 15.0, 14.5, 14.0],
        "prefix": "Instruction: Vividly describe the concept represented here.\n\nConcept: ",
        "suffix": "\n\nDescription: ",
    },
    {
        "name": "The Sommelier's Blind Taste Test",
        "words": [" coffee", " vanilla", " oat", " smoke"],
        "logits": [15.5, 15.5, 15.0, 14.5],
        "prefix": "Instruction: Vividly describe the concept represented here.\n\nConcept: ",
        "suffix": "\n\nDescription: ",
    },
]


with open(EXP_CFG.log_file, "w", encoding="utf-8") as f:

    emit("LATENT CONCEPT DESCRIPTION EXPERIMENT", f)
    emit(
        f"Temp: {CFG.temperature} | Top-p: {CFG.top_p} | "
        f"k range: [{CFG.min_k}, {CFG.max_k}] | "
        f"Solver steps: {CFG.solver_steps}",
        f,
    )
    emit("", f)

    for exp in experiments:

        emit("\n" + "=" * 120, f)
        emit(f"EXPERIMENT: {exp['name']}", f)
        emit("=" * 120, f)

        # --- Tokenize words ---
        target_ids = []
        clean_words = []

        for w in exp["words"]:
            t = tokenizer.encode(w, add_special_tokens=False)
            if len(t) == 1:
                target_ids.append(t[0])
                clean_words.append(w.strip())
            else:
                t2 = tokenizer.encode(w.strip(), add_special_tokens=False)
                if len(t2) == 1:
                    target_ids.append(t2[0])
                    clean_words.append(w.strip())
                else:
                    raise ValueError(f"Token fragmentation: {w}")

        logit_values = torch.tensor(exp["logits"], device=device)
        target_ids_set = set(target_ids)

        # --- Compute probabilities for display ---
        full_logits = build_fake_logits(target_ids, logit_values, vocab_embs.size(0))
        full_probs = F.softmax(full_logits, dim=-1)
        full_probs_scaled = F.softmax(full_logits / CFG.temperature, dim=-1)
        target_probs_raw = full_probs[target_ids]
        target_probs_scaled = full_probs_scaled[target_ids]

        # --- Input anatomy ---
        input_raw_embs = vocab_embs[target_ids]
        input_mags = torch.norm(input_raw_embs, p=2, dim=1)

        emit(f"\n--- INPUT TOKENS (k={len(target_ids)}) ---", f)
        header_in = f"  {'ID':<6} | {'Token':<12} | {'Logit':<6} | {'Raw P':<8} | {'Scaled P (T={CFG.temperature})':<20} | {'Magnitude':<9}"
        emit(header_in, f)
        emit("  " + "-" * (len(header_in) - 2), f)

        for tid, w, logit, p_raw, p_scaled, mag in zip(
            target_ids,
            clean_words,
            logit_values,
            target_probs_raw,
            target_probs_scaled,
            input_mags,
        ):
            emit(
                f"  {tid:<6} | {w:<12} | {logit.item():>5.1f} | {p_raw.item()*100:>7.2f}% | {p_scaled.item()*100:>19.2f}% | {mag.item():>7.3f}",
                f,
            )

        # --- Pre-forge cross-similarity ---
        input_unit = F.normalize(input_raw_embs, p=2, dim=1)
        cross_sim = input_unit @ input_unit.T
        df_inputs = pd.DataFrame(
            cross_sim.cpu().numpy(), index=clean_words, columns=clean_words
        )

        emit(f"\n--- INPUT CROSS-SIMILARITY ---", f)
        emit(df_inputs.to_string(), f)

        # --- Build vectors ---
        v_soft, v_solver, v_colar, v_centroid = build_vectors(
            target_ids,
            logit_values,
        )

        vectors = torch.cat([v_soft, v_solver, v_colar, v_centroid], dim=0)
        labels = ["Soft", "Solver", "CoLaR", "Centroid"]

        # --- Cross-similarity of output vectors ---
        vectors_unit = F.normalize(vectors, dim=1)
        mags = vectors.norm(p=2, dim=1).cpu().numpy()

        sim_matrix = (vectors_unit @ vectors_unit.T).cpu().numpy()
        df_vecs = pd.DataFrame(sim_matrix, index=labels, columns=labels)
        emit(f"\n--- OUTPUT CROSS-SIMILARITY ---", f)
        for label, mag in zip(labels, mags):
            emit(f"  {label}: |v| = {mag:.3f}", f)
        emit(df_vecs.to_string(), f)

        # --- Tables ---
        render_similarity_table(
            vectors, labels, target_ids_set, "NEAREST NEIGHBORS (COSINE)", f
        )
        render_dot_table(
            vectors, labels, target_ids_set, "NEAREST NEIGHBORS (DOT PRODUCT)", f
        )

        # --- Generation ---
        emit("\n--- CONCEPT DESCRIPTIONS ---", f)

        for name, vec in zip(labels, [v_soft, v_solver, v_colar, v_centroid]):
            text, lines = splice_and_evaluate(exp["prefix"], exp["suffix"], vec)

            emit(f"\n  {name:>8} → '{text}'", f)
            for line in lines:
                emit(line, f)

    emit("\nDone.", f)

print(f"\nSaved → {EXP_CFG.log_file}")
