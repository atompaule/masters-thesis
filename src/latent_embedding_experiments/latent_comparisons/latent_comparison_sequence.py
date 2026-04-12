"""
latent_comparison_sequence.py

Autoregressive generation using different embedding strategies.
"""

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.latent_embedding_experiments.algorithms.config import CFG
from src.latent_embedding_experiments.algorithms.discrete_sharpened import (
    discrete_sharpened,
    discrete_sharpened_dot_rescaled,
)
from src.latent_embedding_experiments.algorithms.dylar import dylar
from src.latent_embedding_experiments.algorithms.latent_head import load_latent_head
from src.latent_embedding_experiments.algorithms.soft_thinking import (
    soft_thinking,
    soft_thinking_normalized,
)
from src.latent_embedding_experiments.algorithms.soft_thinking_sharpened import (
    soft_thinking_sharpened_aggregate,
    soft_thinking_sharpened_per_token,
)
from src.latent_embedding_experiments.algorithms.solver import geometric_solver
from src.latent_embedding_experiments.algorithms.utils import emit, select_targets

# --- CONFIGURATION ---
PROMPT = "The fundamental difference between humans and artificial intelligence is"
STEPS = 50
DISPLAY_K_MIN = 20
DISPLAY_K_DIST = 20

N_INTERLOPERS = 10
TARGET_SIM = 0.93

DYLAR_K = 2
DYLAR_ENTROPY_THRESHOLD = 0.1  # set None to disable dynamic switch

NEXT_STEP_EMBEDDING = "soft_thinking"

LOG_FILE = (
    "src/latent_embedding_experiments/logs/"
    f"llama_8b_latent_comparison_sequence_{NEXT_STEP_EMBEDDING}_{TARGET_SIM}.txt"
)

LATENT_HEAD_CHECKPOINT = "/work/utsch/masters-thesis/latent_embedding_experiments/latent_head/latent_head_mlp_2h.pt"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_latent_comparison_sequence(
    next_step_embedding: str,
    latent_head_checkpoint: str,
    n_interlopers: int,
    target_sim: float,
) -> None:
    print(f"Loading model: {CFG.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(CFG.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        CFG.model_id, device_map="auto", torch_dtype=torch.bfloat16
    )
    model.eval()
    device = next(model.parameters()).device

    # --- Embedding matrix ---
    vocab_embs = model.get_input_embeddings().weight.detach().to(torch.float32)
    vocab_embs_norm = F.normalize(vocab_embs, p=2, dim=1)
    vocab_embs_bf16 = vocab_embs.to(torch.bfloat16)

    embed_layer = model.get_input_embeddings()
    prompt_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(device)
    inputs_embeds = embed_layer(prompt_ids).to(model.dtype)

    # --- LatentHead ---
    hidden_dim = model.config.hidden_size
    latent_head = load_latent_head(latent_head_checkpoint, hidden_dim, device)

    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.width", 1000)

    step_targets: list[list[tuple[str, float, float]]] = []
    # Per-step cosine similarity to discrete top-1, accumulated across steps
    discrete_sims_per_approach: dict[str, list[float]] = {}

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        emit("LLAMA 8B LATENT COMPARISON SEQUENCE LOG", f)
        emit(
            f"Steps: {STEPS} | Temp: {CFG.temperature} | "
            f"Min-p: {CFG.min_p} | min_k: {CFG.min_k}",
            f,
        )
        emit(f"Next-step input: {next_step_embedding}", f)
        emit(f"LatentHead: loaded from {latent_head_checkpoint}", f)
        if next_step_embedding in (
            "clean_soft",
            "clean_soft_aggregate",
            "discrete_cleaned",
            "discrete_cleaned_dot_rescaled",
        ):
            emit(f"n_interlopers={n_interlopers} | target_sim={target_sim}", f)
        emit("", f)

        context_so_far = PROMPT
        greedy_continuation_ids: list[int] = []

        with torch.no_grad():
            for step in range(STEPS):
                emit(f"\n{'='*140}", f)
                emit(f"STEP {step + 1} | Context: '...{context_so_far[-60:]}'", f)
                emit(f"{'='*140}", f)

                # --- Forward pass ---
                outputs = model(inputs_embeds=inputs_embeds, output_hidden_states=True)
                logits = outputs.logits[0, -1, :].to(torch.float32)
                greedy_id = int(logits.argmax(dim=-1).item())
                greedy_word = tokenizer.decode([greedy_id])
                top1_magnitude = vocab_embs[greedy_id].norm(p=2)

                # --- Full distribution for display ---
                full_probs = F.softmax(logits, dim=-1)
                full_probs_scaled = F.softmax(logits / CFG.temperature, dim=-1)

                # --- Min-p target selection ---
                target_logits, target_ids = select_targets(logits)
                k = len(target_ids)

                target_probs = full_probs[target_ids]
                target_probs_scaled = full_probs_scaled[target_ids]

                target_words = tokenizer.batch_decode(target_ids)
                target_ids_set = set(target_ids.tolist())

                step_target_info = []
                for word, p_raw, p_scaled in zip(
                    target_words, target_probs, target_probs_scaled
                ):
                    clean = word.replace("\n", "\\n").replace("\r", "\\r").strip()
                    step_target_info.append((clean, p_raw.item(), p_scaled.item()))
                step_targets.append(step_target_info)


                # --- Global ranks for display ---
                sorted_idx = torch.argsort(logits, descending=True)
                global_ranks = torch.empty_like(sorted_idx)
                global_ranks[sorted_idx] = torch.arange(
                    1, len(logits) + 1, device=device
                )

                # --- Log probability distribution ---
                top_display_logits, top_display_ids = torch.topk(logits, DISPLAY_K_DIST)
                top_display_probs = full_probs[top_display_ids]
                top_display_probs_scaled = full_probs_scaled[top_display_ids]
                top_display_words = tokenizer.batch_decode(top_display_ids)

                emit(
                    f"--- LOGIT DISTRIBUTION (top {DISPLAY_K_DIST}) — "
                    f"* = min-p target (min_p={CFG.min_p}, k={k}) ---",
                    f,
                )
                cumulative_raw = 0.0
                cumulative_scaled = 0.0
                for rank, (word, p_raw, p_scaled, tid) in enumerate(
                    zip(
                        top_display_words,
                        top_display_probs,
                        top_display_probs_scaled,
                        top_display_ids,
                    )
                ):
                    cumulative_raw += p_raw.item()
                    cumulative_scaled += p_scaled.item()
                    clean = word.replace("\n", "\\n").replace("\r", "\\r")
                    marker = "*" if tid.item() in target_ids_set else " "
                    mag = vocab_embs[tid.item()].norm(p=2).item()
                    emit(
                        f"  {marker} Rank {rank+1:2d} | Raw: {p_raw*100:6.2f}% | "
                        f"Scaled (T={CFG.temperature}): {p_scaled*100:6.2f}% | "
                        f"Cumul (raw): {cumulative_raw*100:6.2f}% | "
                        f"Cumul (scaled): {cumulative_scaled*100:6.2f}% | "
                        f"|emb|: {mag:.3f} | '{clean}'",
                        f,
                    )

                # --- Shared quantities ---
                pool_embs = vocab_embs[target_ids]  # [k, d]
                target_norms = pool_embs.norm(p=2, dim=1)  # [k]
                target_magnitude = torch.sum(target_probs_scaled * target_norms)

                last_hidden = (
                    outputs.hidden_states[-1][0, -1, :].unsqueeze(0).to(torch.float32)
                )

                # --- Vector synthesis (only compute what CFG.approaches requests) ---
                approaches = set(CFG.approaches)

                # Always compute discrete top-1 — needed as the greedy next step
                # and as the cosine similarity reference for the summary table
                v_discrete = pool_embs[0:1]

                def need(key: str) -> bool:
                    return key in approaches or key == next_step_embedding

                v_discrete_cleaned = (
                    discrete_sharpened(
                        top1_id=greedy_id,
                        vocab_embs=vocab_embs,
                        vocab_embs_norm=vocab_embs_norm,
                        target_magnitude=top1_magnitude,
                        n_interlopers=n_interlopers,
                        target_sim=target_sim,
                    )
                    if need("discrete_cleaned")
                    else None
                )

                v_discrete_cleaned_dr = (
                    discrete_sharpened_dot_rescaled(
                        top1_id=greedy_id,
                        vocab_embs=vocab_embs,
                        vocab_embs_norm=vocab_embs_norm,
                        target_magnitude=top1_magnitude, # won't matter
                        n_interlopers=n_interlopers,
                        target_sim=target_sim,
                    )
                    if need("discrete_cleaned_dot_rescaled")
                    else None
                )

                v_soft = (
                    soft_thinking(logits, vocab_embs)
                    if need("soft_thinking")
                    or need("soft_thinking_normalized")
                    or need("clean_soft_aggregate")
                    else None
                )

                v_soft_normalized = (
                    soft_thinking_normalized(logits, vocab_embs_norm, target_magnitude)
                    if need("soft_thinking_normalized") or need("clean_soft_aggregate")
                    else None
                )

                if need("dylar") or need("dylar_dynamic"):
                    last_hidden_f32 = last_hidden.squeeze(0)  # [d], float32
                    v_dylar = dylar(
                        logits=logits,
                        hidden=last_hidden_f32,
                        vocab_embs=vocab_embs,
                        K=DYLAR_K,
                        entropy_threshold=None,  # always compute for logging
                    )
                    # magnitude-normalise to match other approaches
                    dylar_norm = v_dylar.norm(p=2, dim=1, keepdim=True).clamp_min(1e-8)
                    v_dylar = (target_magnitude * v_dylar) / dylar_norm

                    # dynamic variant: falls back to discrete when entropy < threshold
                    if need("dylar_dynamic"):
                        v_dylar_dynamic = dylar(
                            logits=logits,
                            hidden=last_hidden_f32,
                            vocab_embs=vocab_embs,
                            K=DYLAR_K,
                            entropy_threshold=DYLAR_ENTROPY_THRESHOLD,
                        )
                        if v_dylar_dynamic is None:
                            # dynamic switch chose explicit decoding — use discrete top-1
                            v_dylar_dynamic = v_discrete.clone()
                        else:
                            dd_norm = v_dylar_dynamic.norm(
                                p=2, dim=1, keepdim=True
                            ).clamp_min(1e-8)
                            v_dylar_dynamic = (
                                target_magnitude * v_dylar_dynamic
                            ) / dd_norm
                    else:
                        v_dylar_dynamic = None
                else:
                    v_dylar = None
                    v_dylar_dynamic = None

                v_clean_soft = (
                    soft_thinking_sharpened_per_token(
                        vocab_embs=vocab_embs,
                        vocab_embs_norm=vocab_embs_norm,
                        target_ids=target_ids,
                        target_probs_scaled=target_probs_scaled,
                        target_magnitude=target_magnitude,
                        n_interlopers=n_interlopers,
                        target_sim=target_sim,
                    )
                    if need("clean_soft")
                    else None
                )

                v_clean_soft_agg = (
                    soft_thinking_sharpened_aggregate(
                        v_soft=v_soft_normalized.squeeze(0),
                        vocab_embs=vocab_embs,
                        vocab_embs_norm=vocab_embs_norm,
                        target_ids=target_ids,
                        target_probs_scaled=target_probs_scaled,
                        target_magnitude=target_magnitude,
                        n_interlopers=n_interlopers,
                        target_sim=target_sim,
                    )
                    if need("clean_soft_aggregate")
                    else None
                )

                if need("latent_head"):
                    with torch.enable_grad():
                        v_latent_head = latent_head(last_hidden.detach())
                    v_latent_head = v_latent_head.detach().float()
                    lh_norm = v_latent_head.norm(p=2, dim=1, keepdim=True).clamp_min(
                        1e-8
                    )
                    v_latent_head = (target_magnitude * v_latent_head) / lh_norm
                else:
                    v_latent_head = None

                if need("solver"):
                    with torch.enable_grad():
                        v_solver = geometric_solver(logits, vocab_embs)
                else:
                    v_solver = None

                v_centroid = (
                    pool_embs.mean(dim=0, keepdim=True) if need("centroid") else None
                )

                if need("coconut"):
                    v_coconut = last_hidden.clone()
                    norm = v_coconut.norm(p=2, dim=1, keepdim=True).clamp_min(1e-8)
                    v_coconut = (target_magnitude * v_coconut) / norm
                else:
                    v_coconut = None

                # --- Select next-step vector ---
                vectors = {
                    "discrete_top1": v_discrete,
                    "discrete_cleaned": v_discrete_cleaned,
                    "discrete_cleaned_dot_rescaled": v_discrete_cleaned_dr,
                    "soft_thinking": v_soft,
                    "soft_thinking_normalized": v_soft_normalized,
                    "dylar": v_dylar,
                    "dylar_dynamic": v_dylar_dynamic,
                    "clean_soft": v_clean_soft,
                    "clean_soft_aggregate": v_clean_soft_agg,
                    "latent_head": v_latent_head,
                    "solver": v_solver,
                    "centroid": v_centroid,
                    "coconut": v_coconut,
                }

                next_vec = vectors[next_step_embedding].detach().float()
                next_vec_bf16 = next_vec.to(device=device, dtype=model.dtype)
                inputs_embeds = torch.cat(
                    [inputs_embeds, next_vec_bf16.unsqueeze(0)], dim=1
                )

                # Resolve the token actually fed forward for logging
                if next_step_embedding in (
                    "discrete_top1",
                    "discrete_cleaned",
                    "discrete_cleaned_dot_rescaled",
                ):
                    fed_id = greedy_id
                else:
                    next_vec_unit = F.normalize(next_vec, p=2, dim=1)
                    fed_id = int(
                        (next_vec_unit @ vocab_embs_norm.T).squeeze(0).argmax().item()
                    )

                fed_word = tokenizer.decode([fed_id])
                greedy_continuation_ids.append(fed_id)  # was greedy_id
                context_so_far += fed_word  # was greedy_word

                clean_fed = fed_word.replace("\n", "\\n").replace("\r", "\\r")
                clean_greedy = greedy_word.replace("\n", "\\n").replace("\r", "\\r")
                emit(
                    f"--- NEXT INPUT: {next_step_embedding} | "
                    f"greedy token: '{clean_greedy}' | fed token: '{clean_fed}' ---",
                    f,
                )

                # --- Cross-similarity matrix (only active approaches) ---
                approach_display = {
                    "discrete_top1": ("Discrete", v_discrete),
                    "discrete_cleaned": ("DiscCleaned", v_discrete_cleaned),
                    "discrete_cleaned_dot_rescaled": (
                        "DiscCleanedDR",
                        v_discrete_cleaned_dr,
                    ),
                    "soft_thinking": ("Soft", v_soft),
                    "soft_thinking_normalized": ("SoftNorm", v_soft_normalized),
                    "dylar": ("DyLaR", v_dylar),
                    "dylar_dynamic": ("DyLaRDyn", v_dylar_dynamic),
                    "clean_soft": ("CleanSoft", v_clean_soft),
                    "clean_soft_aggregate": ("CleanSoftAgg", v_clean_soft_agg),
                    "latent_head": ("LatentHead", v_latent_head),
                    "solver": ("Solver", v_solver),
                    "centroid": ("Centroid", v_centroid),
                    "coconut": ("Coconut", v_coconut),
                }

                active = [
                    k for k in CFG.approaches if approach_display[k][1] is not None
                ]
                labels = [approach_display[k][0] for k in active]
                all_vecs_list = [approach_display[k][1] for k in active]

                # --- Track per-step cosine similarity to discrete top-1 ---
                v_discrete_unit = F.normalize(v_discrete, p=2, dim=1)  # [1, d]
                for key in active:
                    label, vec = approach_display[key]
                    vec_unit = F.normalize(vec, p=2, dim=1)
                    sim = (vec_unit @ v_discrete_unit.T).squeeze().item()
                    discrete_sims_per_approach.setdefault(label, []).append(sim)

                all_vecs = torch.cat(all_vecs_list, dim=0)
                all_vecs_unit = F.normalize(all_vecs, p=2, dim=1)
                mags = all_vecs.norm(p=2, dim=1).cpu().numpy()

                sim_matrix = (all_vecs_unit @ all_vecs_unit.T).cpu().numpy()
                df = pd.DataFrame(sim_matrix, index=labels, columns=labels)
                emit(f"\n--- CROSS-SIMILARITY ---", f)
                emit(df.to_string(), f)

                # --- Nearest neighbor tables ---
                display_k = max(DISPLAY_K_MIN, 2 * k)
                col_width = 36

                # Table 1: Cosine similarity
                emit(
                    f"\n--- TOP {display_k} NEAREST NEIGHBORS (COSINE) — "
                    f"* = min-p target (min_p={CFG.min_p}, k={k}) ---",
                    f,
                )
                cos_sims = all_vecs_unit @ vocab_embs_norm.T
                top_vals_cos, top_idx_cos = torch.topk(cos_sims, display_k, dim=1)

                header = f"{'Rank':<4} |"
                for label in labels:
                    header += f" {label:<{col_width}} |"
                emit(header[:-2], f)
                emit("-" * len(header[:-2]), f)

                for rank in range(display_k):
                    row = f"{rank+1:<4} |"
                    for i in range(len(labels)):
                        tid = top_idx_cos[i, rank].item()
                        w = (
                            tokenizer.decode([tid])
                            .replace("\n", "\\n")
                            .replace("\r", "\\r")
                        )
                        if len(w) > 10:
                            w = w[:8] + ".."
                        gr = global_ranks[tid].item()
                        marker = "*" if tid in target_ids_set else " "
                        val = top_vals_cos[i, rank].item()
                        cell = f"{gr:<5} {marker}'{w}' ({val:.3f})"
                        row += f" {cell:<{col_width}} |"
                    emit(row[:-2], f)

                # Table 2: Dot product
                emit(
                    f"\n--- TOP {display_k} TOKENS (DOT PRODUCT) — "
                    f"* = min-p target (min_p={CFG.min_p}, k={k}) ---",
                    f,
                )
                all_vecs_bf16 = all_vecs.to(torch.bfloat16)
                dot_prods = all_vecs_bf16 @ vocab_embs_bf16.T
                top_vals_dot, top_idx_dot = torch.topk(dot_prods, display_k, dim=1)

                header = f"{'Rank':<4} |"
                for label, mag in zip(labels, mags):
                    header += f" {label} (|v|={mag:.1f})".ljust(col_width + 1) + " |"
                emit(header[:-2], f)
                emit("-" * len(header[:-2]), f)

                for rank in range(display_k):
                    row = f"{rank+1:<4} |"
                    for i in range(len(labels)):
                        tid = top_idx_dot[i, rank].item()
                        w = (
                            tokenizer.decode([tid])
                            .replace("\n", "\\n")
                            .replace("\r", "\\r")
                        )
                        if len(w) > 10:
                            w = w[:8] + ".."
                        gr = global_ranks[tid].item()
                        marker = "*" if tid in target_ids_set else " "
                        val = top_vals_dot[i, rank].item()
                        cell = f"{gr:<5} {marker}'{w}' ({val:>7.3f})"
                        row += f" {cell:<{col_width}} |"
                    emit(row[:-2], f)

        # --- Final summaries ---
        greedy_full_ids = prompt_ids[0].tolist() + greedy_continuation_ids
        greedy_text = tokenizer.decode(greedy_full_ids, skip_special_tokens=True)
        greedy_cont = tokenizer.decode(
            greedy_continuation_ids, skip_special_tokens=True
        )

        esc = lambda s: s.replace("\n", "\\n").replace("\r", "\\r")
        emit("", f)
        emit("=" * 140, f)
        emit("GREEDY SEQUENCE (independent of --next-step-embedding)", f)
        emit(
            f"Continuation ({len(greedy_continuation_ids)} tokens): {esc(greedy_cont)}",
            f,
        )
        emit(f"Full: {esc(greedy_text)}", f)
        emit("=" * 140, f)

        # --- Target token summary table ---
        emit("", f)
        emit("=" * 140, f)
        emit(
            f"TARGET TOKEN SUMMARY (min_p={CFG.min_p}) — "
            f"raw% / scaled% (T={CFG.temperature})",
            f,
        )
        emit("=" * 140, f)

        max_targets = max(len(st) for st in step_targets)
        step_col_width = 28

        header = f"{'':>4} |"
        for s in range(STEPS):
            header += f" {'Step ' + str(s+1):^{step_col_width}} |"
        emit(header[:-2], f)
        emit("-" * len(header[:-2]), f)

        for rank in range(max_targets):
            row = f"{'#' + str(rank+1):>4} |"
            for s in range(STEPS):
                targets = step_targets[s]
                if rank < len(targets):
                    token, p_raw, p_scaled = targets[rank]
                    if len(token) > 10:
                        token = token[:8] + ".."
                    cell = f"'{token}' {p_raw*100:.1f}/{p_scaled*100:.1f}"
                else:
                    cell = ""
                row += f" {cell:^{step_col_width}} |"
            emit(row[:-2], f)

        emit("=" * 140, f)

        # --- Average cosine similarity to discrete top-1 per approach ---
        emit("", f)
        emit("=" * 140, f)
        emit("AVERAGE COSINE SIMILARITY TO DISCRETE TOP-1 PER APPROACH", f)
        emit("=" * 140, f)
        col_w = 24
        header = f"{'Approach':<{col_w}} | {'Mean':>8} | {'Min':>8} | {'Max':>8} | {'Std':>8}"
        emit(header, f)
        emit("-" * len(header), f)
        for label, sims in discrete_sims_per_approach.items():
            t = torch.tensor(sims)
            emit(
                f"{label:<{col_w}} | {t.mean().item():>8.4f} | "
                f"{t.min().item():>8.4f} | {t.max().item():>8.4f} | "
                f"{t.std().item():>8.4f}",
                f,
            )
        emit("=" * 140, f)

        emit(f"\nLog saved to {LOG_FILE}", f)


if __name__ == "__main__":
    run_latent_comparison_sequence(
        next_step_embedding=NEXT_STEP_EMBEDDING,
        latent_head_checkpoint=LATENT_HEAD_CHECKPOINT,
        n_interlopers=N_INTERLOPERS,
        target_sim=TARGET_SIM,
    )
