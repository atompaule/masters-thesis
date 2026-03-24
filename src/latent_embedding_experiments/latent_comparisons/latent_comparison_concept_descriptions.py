import math
from dataclasses import dataclass

import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.latent_embedding_experiments.algorithms.solver import fast_geometric_solver

# =========================================================
# CONFIG
# =========================================================


@dataclass
class Config:
    model_id: str = "meta-llama/Llama-3.1-8B-Instruct"
    temperature: float = 0.6  # Standard thermodynamic pressure
    k: int = 5
    display_k: int = 10
    log_file: str = (
        "src/latent_embedding_experiments/logs/llama_8b_latent_comparison_concept_descriptions.txt"
    )

    solver_steps: int = 300
    lr: float = 0.05
    danger_topk: int = 2000

    # --- ENTROPY BOUNDS ---
    entropy_min: float = 1.80
    entropy_max: float = 2.10
    tau_min: float = 0.05
    tau_max: float = 10.0
    tau_bisection_steps: int = 40


CFG = Config()


# =========================================================
# THERMODYNAMICS (ENTROPY-BOUNDING)
# =========================================================


def entropy_from_probs(probs: torch.Tensor) -> torch.Tensor:
    probs = probs.clamp_min(1e-12)
    return -(probs * probs.log()).sum()


def topk_probs_from_logits_with_tau(logits: torch.Tensor, tau: float) -> torch.Tensor:
    return F.softmax(logits / tau, dim=-1)


def solve_tau_for_entropy_range(
    logits: torch.Tensor,
    entropy_min: float,
    entropy_max: float,
    tau_min: float = 0.05,
    tau_max: float = 10.0,
    steps: int = 40,
):
    with torch.no_grad():
        probs_lo = topk_probs_from_logits_with_tau(logits, tau_min)
        probs_hi = topk_probs_from_logits_with_tau(logits, tau_max)
        H_lo = entropy_from_probs(probs_lo).item()
        H_hi = entropy_from_probs(probs_hi).item()

        if H_lo >= entropy_min and H_lo <= entropy_max:
            return tau_min, probs_lo, H_lo
        if H_hi >= entropy_min and H_hi <= entropy_max:
            return tau_max, probs_hi, H_hi

        if H_lo > entropy_max:
            return tau_min, probs_lo, H_lo

        if H_hi < entropy_min:
            return tau_max, probs_hi, H_hi

        target_H = 0.5 * (entropy_min + entropy_max)

        lo = tau_min
        hi = tau_max
        best_tau = None
        best_probs = None
        best_H = None
        best_err = float("inf")

        for _ in range(steps):
            mid = 0.5 * (lo + hi)
            probs_mid = topk_probs_from_logits_with_tau(logits, mid)
            H_mid = entropy_from_probs(probs_mid).item()
            err = abs(H_mid - target_H)

            if err < best_err:
                best_err = err
                best_tau = mid
                best_probs = probs_mid
                best_H = H_mid

            if H_mid < target_H:
                lo = mid
            else:
                hi = mid

        return best_tau, best_probs, best_H


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

print(f"Awakening model: {CFG.model_id}")
tokenizer = AutoTokenizer.from_pretrained(CFG.model_id)
model = AutoModelForCausalLM.from_pretrained(
    CFG.model_id, device_map="auto", torch_dtype=torch.bfloat16
)

device = model.device

raw_embeddings = model.get_input_embeddings().weight.detach().to(torch.float32)
norm_dictionary = F.normalize(raw_embeddings, dim=1)


# =========================================================
# VECTOR FORGING
# =========================================================


def build_vectors(target_ids, adj_probs, entropy_adj_probs, isolated_logits):
    raw = raw_embeddings[target_ids]

    v_colar = raw.sum(dim=0, keepdim=True) / math.sqrt(len(target_ids))
    v_soft = (raw * adj_probs.unsqueeze(1)).sum(dim=0, keepdim=True)
    v_soft_entropy = (raw * entropy_adj_probs.unsqueeze(1)).sum(dim=0, keepdim=True)

    avg_target_mag = torch.norm(raw, p=2, dim=1).mean().item()

    target_embs_norm = norm_dictionary[target_ids]

    # T=1 softmax over the candidate list; solver applies p^(1/T) rescaling internally.
    solver_pool_probs = F.softmax(isolated_logits, dim=0)

    with torch.enable_grad():
        v_solver = fast_geometric_solver(
            target_norm=target_embs_norm,
            target_ids=target_ids,
            dict_norm=norm_dictionary,
            magnitude=avg_target_mag,
            pool_logits=solver_pool_probs,
            temperature=CFG.temperature,
        )

    return v_colar, v_soft, v_soft_entropy, v_solver


# =========================================================
# TABLE RENDERING
# =========================================================


def render_similarity_table(vectors, labels, target_ids, title, f):

    emit(f"\n--- {title} ---", f)

    vectors_unit = F.normalize(vectors, dim=1)
    sims = vectors_unit @ norm_dictionary.T

    col_width = 30

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
            marker = tid in target_ids
            row += " " + format_cell(token, vals[i, r].item(), col_width, marker) + " |"
        emit(row[:-2], f)


def render_dot_table(vectors, labels, target_ids, title, f):

    emit(f"\n--- {title} ---", f)

    vecs = vectors.to(torch.float32)
    emb = raw_embeddings.to(torch.float32)

    dots = vecs @ emb.T

    col_width = 30

    header = f"{'Rank':<4} |"
    for label in labels:
        header += f" {label:<{col_width}} |"

    emit(header[:-2], f)
    emit("-" * len(header), f)

    vals, idxs = torch.topk(dots, CFG.display_k, dim=1)

    for r in range(CFG.display_k):
        row = f"{r+1:<4} |"
        for i in range(len(labels)):
            tid = idxs[i, r].item()
            token = clean_token(tokenizer.decode([tid]))
            marker = tid in target_ids
            row += " " + format_cell(token, vals[i, r].item(), col_width, marker) + " |"
        emit(row[:-2], f)


# =========================================================
# GENERATION (EKG)
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
# EXPERIMENT LOOP
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
        # Raw % from log (T=1): 13.60, 7.75, 6.84, 6.42, 4.70, 4.15, 3.66, 3.44, 2.68, 2.09.
        # Logits = log(p) shifted so argmax logit is 17.0 → same relative softmax at T=1 as the run.
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
        "words": [
            " city",
            " excitement",
            " inspiration",
            " bright",
            " hope",
        ],
        "logits": [16.0, 15.5, 15.0, 14.5, 14.0],
        "prefix": "Instruction: Vividly describe the concept represented here.\n\nConcept: ",
        "suffix": "\n\nDescription: ",
    },
    {
        "name": "City on top — dark, depressed, and related low-affect tone",
        "words": [
            " city",
            " dark",
            " depressed",
            " bleak",
            " sad",
        ],
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
        "logits": [
            15.5,
            15.5,
            15.0,
            14.5,
        ],  # Coffee is the base, pepper is the heavy secondary
        "prefix": "Instruction: Vividly describe the concept represented here.\n\nConcept: ",
        "suffix": "\n\nDescription: ",
    },
]


with open(CFG.log_file, "w", encoding="utf-8") as f:

    emit("🌌 LATENT INTERROGATION LAB 🌌", f)
    emit(
        f"Base Temp: {CFG.temperature} | Entropy Bounds: [{CFG.entropy_min:.2f}, {CFG.entropy_max:.2f}]",
        f,
    )

    for exp in experiments:

        emit("\n" + "=" * 120, f)
        emit(f"🧪 {exp['name']}", f)
        emit("=" * 120, f)

        # --- TOKEN FIXING ---
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

        logits = torch.tensor(exp["logits"], device=device)

        # Applying softmax over the temperature-scaled isolated logits perfectly mathematically
        # mimics taking the top-K slice of a full-vocabulary distribution and renormalizing it.
        scaled_logits = logits / CFG.temperature
        adj_probs = F.softmax(scaled_logits, dim=0)
        adj_entropy = entropy_from_probs(adj_probs).item()

        # Seek the bound
        tau_entropy, entropy_adj_probs, bounded_entropy = solve_tau_for_entropy_range(
            logits=logits,
            entropy_min=CFG.entropy_min,
            entropy_max=CFG.entropy_max,
            tau_min=CFG.tau_min,
            tau_max=CFG.tau_max,
            steps=CFG.tau_bisection_steps,
        )

        input_raw_embs = raw_embeddings[target_ids]
        input_mags = torch.norm(input_raw_embs, p=2, dim=1)

        # --- THE RAW INGREDIENTS DOSSIER ---
        emit("\n--- THE CHEMICAL BASELINE (Input Anatomy) ---", f)
        emit(f"  Standard Temp (T={CFG.temperature}) Entropy: {adj_entropy:.4f}", f)
        emit(
            f"  Bounded Temp  (T={tau_entropy:.4f}) Entropy: {bounded_entropy:.4f}\n", f
        )

        header_in = f"  {'ID':<6} | {'Token':<12} | {'Logit':<6} | {'Soft P':<8} | {'Ent P':<8} | {'Magnitude':<9}"
        emit(header_in, f)
        emit("  " + "-" * (len(header_in) - 2), f)

        for i, (tid, w, logit, p_soft, p_ent, mag) in enumerate(
            zip(
                target_ids,
                clean_words,
                logits,
                adj_probs,
                entropy_adj_probs,
                input_mags,
            )
        ):
            emit(
                f"  {tid:<6} | {w:<12} | {logit.item():>5.1f} | {p_soft.item()*100:>7.2f}% | {p_ent.item()*100:>7.2f}% | {mag.item():>7.3f}",
                f,
            )

        # --- PRE-FORGE CROSS-SIMILARITY ---
        input_unit = F.normalize(input_raw_embs, p=2, dim=1)
        cross_sim = input_unit @ input_unit.T
        df_inputs = pd.DataFrame(
            cross_sim.cpu().numpy(), index=clean_words, columns=clean_words
        )

        emit(f"\n--- PRE-FORGE CROSS-SIMILARITY (Internal Gravity) ---", f)
        emit(df_inputs.to_string(), f)
        emit("\n", f)

        # --- BUILD ---
        v_colar, v_soft, v_soft_entropy, v_solver = build_vectors(
            target_ids, adj_probs, entropy_adj_probs, logits
        )

        vectors = torch.cat([v_colar, v_soft, v_soft_entropy, v_solver], dim=0)
        labels = ["CoLaR", "Soft", "Soft-Ent", "Solver"]

        # --- TABLES ---
        render_similarity_table(
            vectors, labels, target_ids, "Cosine Similarity (Geometry)", f
        )

        render_dot_table(vectors, labels, target_ids, "Dot Product (Magnitude Bias)", f)

        # --- GENERATION ---
        emit("\n--- Interrogation Probe ---", f)

        for name, vec in zip(labels, [v_colar, v_soft, v_soft_entropy, v_solver]):
            text, lines = splice_and_evaluate(exp["prefix"], exp["suffix"], vec)

            emit(f"\n{name:>8} → '{text}'", f)
            for l in lines:
                emit(l, f)

    emit("\nDone.", f)

print(f"\nSaved → {CFG.log_file}")
