import math
from dataclasses import dataclass

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
    temperature: float = 3.0
    k: int = 5
    display_k: int = 10
    log_file: str = "src/latent_embedding_experiments/logs/llama_8b_latent_comparison_concept_descriptions.txt"

    solver_steps: int = 150
    lr: float = 0.05
    danger_topk: int = 1000


CFG = Config()


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
# SOLVER
# =========================================================

def fast_geometric_solver(target_norm, target_ids, dict_norm, magnitude, adj_probs):
    k = len(target_ids)

    base = F.normalize(target_norm.sum(dim=0, keepdim=True), dim=1)

    with torch.no_grad():
        sims = base @ dict_norm.T
        _, idxs = torch.topk(sims.squeeze(0), CFG.danger_topk)

        mask = ~torch.isin(idxs, torch.tensor(target_ids, device=device))
        interlopers = dict_norm[idxs[mask]]

    probe = torch.nn.Parameter(base.squeeze(0))
    opt = optim.Adam([probe], lr=CFG.lr)

    for _ in range(CFG.solver_steps):
        opt.zero_grad()

        p = F.normalize(probe.unsqueeze(0), dim=1)
        sims_t = (p @ target_norm.T).squeeze(0)

        pull = -torch.sum(sims_t * adj_probs) * 2.0

        diffs = sims_t[1:] - sims_t[:-1]
        gaps = adj_probs[:-1] - adj_probs[1:]
        rank_loss = torch.sum(F.relu(diffs + gaps * 0.1))

        sims_i = (p @ interlopers.T).squeeze(0)
        hard, _ = torch.topk(sims_i, k)
        push = F.relu(hard.mean() - sims_t.min() + 0.05) * 1.5

        loss = pull + 3.0 * rank_loss + push
        loss.backward()
        opt.step()

    return F.normalize(probe.unsqueeze(0), dim=1) * magnitude


# =========================================================
# VECTOR FORGING
# =========================================================

def build_vectors(target_ids, adj_probs):
    raw = raw_embeddings[target_ids]
    norm = norm_dictionary[target_ids]

    magnitude = torch.norm(raw, dim=1).mean().item()

    v_colar = raw.sum(dim=0, keepdim=True) / math.sqrt(len(target_ids))
    v_soft = (raw * adj_probs.unsqueeze(1)).sum(dim=0, keepdim=True)

    with torch.enable_grad():
        v_solver = fast_geometric_solver(
            norm, target_ids, norm_dictionary, magnitude, adj_probs
        )

    return v_colar, v_soft, v_solver


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

    vecs = vectors.to(torch.bfloat16)
    emb = raw_embeddings.to(torch.bfloat16)

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
    suffix_ids = tokenizer(suffix, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

    prefix_emb = model.get_input_embeddings()(prefix_ids)
    suffix_emb = model.get_input_embeddings()(suffix_ids)

    vec = vec.unsqueeze(0).to(model.dtype)

    inputs = torch.cat([prefix_emb, vec, suffix_emb], dim=1)

    with torch.no_grad():
        out = model.generate(
            inputs_embeds=inputs,
            max_new_tokens=5,
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

experiments = [
    {
        "name": "The Luxury vs. Greed Halo",
        "words": [" rich", " gold", " fine", " cost", " envy"],
        "logits": [16.0, 14.0, 12.0, 9.0, 7.0],
        "prefix": "Instruction: List the 5 most accurate descriptors for this social concept.\n\nConcept: ",
        "suffix": "\n\nTop 5:\n1.",
        "expected": "rich, gold, fine, cost, envy",
    },
]


with open(CFG.log_file, "w", encoding="utf-8") as f:

    emit("🌌 IN VIVO VECTOR LAB 🌌", f)

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
        probs = F.softmax(logits / CFG.temperature, dim=0)

        emit("\nInput Distribution:", f)
        for i, (w, p) in enumerate(zip(clean_words, probs)):
            emit(f"  {i+1}. {w:<10} | {p.item()*100:>5.2f}%", f)

        # --- BUILD ---
        v_colar, v_soft, v_solver = build_vectors(target_ids, probs)

        vectors = torch.cat([v_colar, v_soft, v_solver], dim=0)
        labels = ["CoLaR", "Soft", "Solver"]

        # --- TABLES ---
        render_similarity_table(
            vectors, labels, target_ids, "Cosine Similarity (Geometry)", f
        )

        render_dot_table(
            vectors, labels, target_ids, "Dot Product (Magnitude Bias)", f
        )

        # --- GENERATION ---
        emit("\n--- Generation Probe ---", f)

        for name, vec in zip(labels, [v_colar, v_soft, v_solver]):
            text, lines = splice_and_evaluate(exp["prefix"], exp["suffix"], vec)

            emit(f"\n{name:>6} → '{text}'", f)
            for l in lines:
                emit(l, f)

    emit("\nDone.", f)

print(f"\nSaved → {CFG.log_file}")