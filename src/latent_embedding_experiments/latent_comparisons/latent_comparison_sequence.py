import argparse

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.latent_embedding_experiments.algorithms.config import CFG
from src.latent_embedding_experiments.algorithms.soft_thinking import soft_thinking
from src.latent_embedding_experiments.algorithms.solver import geometric_solver
from src.latent_embedding_experiments.algorithms.utils import select_targets

# --- CONFIGURATION ---
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
PROMPT = "The fundamental difference between human consciousness and artificial intelligence is"
STEPS = 50
DISPLAY_K_MIN = 20  # minimum rows in nearest neighbor tables
DISPLAY_K_DIST = 20  # rows shown in the logit distribution table

# Next-step embedding mode
NEXT_STEP_EMBEDDING_DEFAULT = "solver"
NEXT_STEP_LABELS = {
    "discrete_top1": "Top-1 discrete token embedding",
    "soft_thinking": "Soft thinking (probability-weighted sum over min-p tokens)",
    "solver": "Geometric solver over min-p tokens",
    "centroid": "Unweighted centroid of min-p token embeddings",
    "coconut": "Last-layer hidden state (continuous)",
}

LOG_FILE = f"src/latent_embedding_experiments/logs/llama_8b_latent_comparison_sequence_{NEXT_STEP_EMBEDDING_DEFAULT}.txt"


def emit(text, file_handle=None):
    print(text)
    if file_handle:
        file_handle.write(text + "\n")


def parse_args():
    p = argparse.ArgumentParser(
        description="Latent comparison: autoregressive generation with different embedding strategies."
    )
    p.add_argument(
        "--next-step-embedding",
        default=NEXT_STEP_EMBEDDING_DEFAULT,
        choices=list(NEXT_STEP_LABELS.keys()),
        help=f"Embedding fed as next input (default: {NEXT_STEP_EMBEDDING_DEFAULT}).",
    )
    return p.parse_args()


def run_latent_comparison_sequence(next_step_embedding: str) -> None:
    print(f"Loading model: {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16
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

    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.width", 1000)

    # Collect per-step target info for the summary table
    step_targets: list[list[tuple[str, float, float]]] = (
        []
    )  # per step: list of (token, raw_p, scaled_p)

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        emit("LLAMA 8B LATENT COMPARISON SEQUENCE LOG", f)
        emit(
            f"Steps: {STEPS} | Temp: {CFG.temperature} | "
            f"Min-p: {CFG.min_p} | min_k: {CFG.min_k}",
            f,
        )
        emit(
            f"Next-step input: {next_step_embedding} — {NEXT_STEP_LABELS[next_step_embedding]}",
            f,
        )
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
                greedy_continuation_ids.append(greedy_id)

                # --- Full distribution for display ---
                full_probs = F.softmax(logits, dim=-1)  # [V]
                full_probs_scaled = F.softmax(logits / CFG.temperature, dim=-1)  # [V]

                # --- Min-p target selection (shared across all methods) ---
                target_logits, target_ids = select_targets(logits)
                k = len(target_ids)

                # Original probabilities from the full distribution (not renormalized)
                target_probs = full_probs[target_ids]  # [k] — raw probs
                target_probs_scaled = full_probs_scaled[
                    target_ids
                ]  # [k] — temperature-scaled probs

                target_words = tokenizer.batch_decode(target_ids)
                target_ids_set = set(target_ids.tolist())

                # Collect for summary table
                step_target_info = []
                for word, p_raw, p_scaled in zip(
                    target_words, target_probs, target_probs_scaled
                ):
                    clean = word.replace("\n", "\\n").replace("\r", "\\r").strip()
                    step_target_info.append((clean, p_raw.item(), p_scaled.item()))
                step_targets.append(step_target_info)

                greedy_word = tokenizer.decode([greedy_id])
                context_so_far += greedy_word

                # --- Global ranks for display ---
                sorted_idx = torch.argsort(logits, descending=True)
                global_ranks = torch.empty_like(sorted_idx)
                global_ranks[sorted_idx] = torch.arange(
                    1, len(logits) + 1, device=device
                )

                # --- Log probability distribution (always show top DISPLAY_K_DIST) ---
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
                    emit(
                        f"  {marker} Rank {rank+1:2d} | Raw: {p_raw*100:6.2f}% | "
                        f"Scaled (T={CFG.temperature}): {p_scaled*100:6.2f}% | "
                        f"Cumul (raw): {cumulative_raw*100:6.2f}% | "
                        f"Cumul (scaled): {cumulative_scaled*100:6.2f}% | '{clean}'",
                        f,
                    )

                # --- Vector synthesis ---
                pool_embs = vocab_embs[target_ids]

                # Discrete top-1
                v_discrete = pool_embs[0:1]

                # Soft thinking (min-p)
                v_soft = soft_thinking(logits, vocab_embs)

                # Geometric solver (min-p)
                with torch.enable_grad():
                    v_solver = geometric_solver(logits, vocab_embs)

                # Centroid (unweighted mean over min-p tokens)
                v_centroid = pool_embs.mean(dim=0, keepdim=True)

                # Coconut (last hidden state)
                v_coconut = (
                    outputs.hidden_states[-1][0, -1, :].unsqueeze(0).to(torch.float32)
                )

                # --- Select next-step vector ---
                vectors = {
                    "discrete_top1": v_discrete,
                    "soft_thinking": v_soft,
                    "solver": v_solver,
                    "centroid": v_centroid,
                    "coconut": v_coconut,
                }
                next_vec = vectors[next_step_embedding].detach().float()
                next_vec_bf16 = next_vec.to(device=device, dtype=model.dtype)
                inputs_embeds = torch.cat(
                    [inputs_embeds, next_vec_bf16.unsqueeze(0)], dim=1
                )

                clean_greedy = greedy_word.replace("\n", "\\n").replace("\r", "\\r")
                emit(
                    f"--- NEXT INPUT: {next_step_embedding} | greedy token: '{clean_greedy}' ---",
                    f,
                )

                # --- Cross-similarity matrix ---
                labels = [
                    "Discrete",
                    f"Soft (k={k})",
                    f"Solver (k={k})",
                    f"Centroid (k={k})",
                    "Coconut",
                ]
                all_vecs = torch.cat(
                    [v_discrete, v_soft, v_solver, v_centroid, v_coconut], dim=0
                )
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

        # =================================================================
        # TARGET TOKEN SUMMARY TABLE
        # One column per step, rows = target tokens with probabilities
        # =================================================================
        emit("", f)
        emit("=" * 140, f)
        emit(
            f"TARGET TOKEN SUMMARY (min_p={CFG.min_p}) — "
            f"raw% / scaled% (T={CFG.temperature})",
            f,
        )
        emit("=" * 140, f)

        # Find the max number of targets across all steps (= number of rows)
        max_targets = max(len(st) for st in step_targets)

        # Column width per step
        step_col_width = 28

        # Header: step numbers
        header = f"{'':>4} |"
        for s in range(STEPS):
            header += f" {'Step ' + str(s+1):^{step_col_width}} |"
        emit(header[:-2], f)
        emit("-" * len(header[:-2]), f)

        # One row per target rank
        for rank in range(max_targets):
            row = f"{'#' + str(rank+1):>4} |"
            for s in range(STEPS):
                targets = step_targets[s]
                if rank < len(targets):
                    token, p_raw, p_scaled = targets[rank]
                    # Truncate long tokens
                    if len(token) > 10:
                        token = token[:8] + ".."
                    cell = f"'{token}' {p_raw*100:.1f}/{p_scaled*100:.1f}"
                else:
                    cell = ""
                row += f" {cell:^{step_col_width}} |"
            emit(row[:-2], f)

        emit("=" * 140, f)
        emit(f"\nLog saved to {LOG_FILE}", f)


if __name__ == "__main__":
    args = parse_args()
    run_latent_comparison_sequence(args.next_step_embedding)
