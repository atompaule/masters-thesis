import argparse

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.latent_embedding_experiments.algorithms.solver import fast_geometric_solver

# --- CONFIGURATION ---
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
PROMPT = "The fundamental difference between human consciousness and artificial intelligence is"
STEPS = 10
K = 10
DISPLAY_K = K * 2
TEMPERATURE = 0.6

# Candidate token pool: "topk" (fixed K) or "topp" (cumulative mass on scaled probs).
CANDIDATE_POOL_DEFAULT = "topk"
TOP_P_DEFAULT = 0.95
TOP_P_MAX_TOKENS_DEFAULT = K  # optional hard cap for top-p pool size

# Next position fed to the model after each step (concatenated as inputs_embeds).
# discrete_top1 matches the original behavior (embedding of greedy top-1 token).
NEXT_STEP_EMBEDDING_DEFAULT = "discrete_top1"
NEXT_STEP_LABELS = {
    "discrete_top1": "Top-1 discrete token embedding",
    "topk_soft": "Soft mix: renormalized scaled probs over the candidate pool",
    "topk_soft_entropy": "Entropy-bounded tau softmax on pool logits (same pool)",
    "solver": "Geometric solver vector over the candidate pool",
    "centroid": "Unweighted mean of candidate pool token embeddings",
    "coconut": "Last-layer hidden state at current position (continuous)",
}

LOG_FILE = f"src/latent_embedding_experiments/logs/llama_8b_latent_comparison_sequence_{NEXT_STEP_EMBEDDING_DEFAULT}_{CANDIDATE_POOL_DEFAULT}.txt"

# --- NEW: entropy-bounded soft-thinking config ---
ENTROPY_MIN = 1.80
ENTROPY_MAX = 2.10
TAU_MIN = 0.05
TAU_MAX = 10.0
TAU_BISECTION_STEPS = 40


def emit(text, file_handle=None):
    print(text)
    if file_handle:
        file_handle.write(text + "\n")


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
    """
    Finds a temperature tau on the top-k logits so that the resulting entropy
    lies inside [entropy_min, entropy_max], if possible.

    Entropy of softmax(logits / tau) increases monotonically with tau.
    """
    with torch.no_grad():
        probs_lo = topk_probs_from_logits_with_tau(logits, tau_min)
        probs_hi = topk_probs_from_logits_with_tau(logits, tau_max)
        H_lo = entropy_from_probs(probs_lo).item()
        H_hi = entropy_from_probs(probs_hi).item()

        # If already below / above reachable range, clamp to nearest endpoint
        if H_lo >= entropy_min and H_lo <= entropy_max:
            return tau_min, probs_lo, H_lo
        if H_hi >= entropy_min and H_hi <= entropy_max:
            return tau_max, probs_hi, H_hi

        # If even the sharpest is too high-entropy, use sharpest
        if H_lo > entropy_max:
            return tau_min, probs_lo, H_lo

        # If even the flattest is too low-entropy, use flattest
        if H_hi < entropy_min:
            return tau_max, probs_hi, H_hi

        # Choose a target entropy inside the band.
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


def select_candidate_pool(
    scaled_probs: torch.Tensor,
    raw_probs: torch.Tensor,
    mode: str,
    k: int,
    top_p: float,
    max_tokens: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build the candidate set used for soft-mix / solver / centroid.

    scaled_probs: full-vocab distribution (here: softmax(logits / TEMPERATURE)).
    raw_probs: full-vocab softmax(logits) at T=1, for logging.

    top-p mode (``topp``): sort by scaled_probs descending; take the shortest
    prefix whose cumulative sum is still <= ``top_p``; always keep at least one
    token. If ``max_tokens`` is set, keep at most that many highest-probability
    tokens (may then cover < ``top_p`` mass).
    """
    if mode == "topk":
        kk = min(k, scaled_probs.shape[-1])
        pool_scaled, pool_ids = torch.topk(scaled_probs, kk)
    elif mode == "topp":
        sorted_p, sorted_idx = torch.sort(scaled_probs, descending=True)
        cum = torch.cumsum(sorted_p, dim=-1)
        n = int((cum <= top_p + 1e-8).sum().item())
        if n < 1:
            n = 1
        if max_tokens is not None:
            n = min(n, max_tokens)
        pool_ids = sorted_idx[:n]
        pool_scaled = sorted_p[:n]
    else:
        raise ValueError(f"Unknown candidate pool mode {mode!r}; use 'topk' or 'topp'.")
    pool_raw = raw_probs[pool_ids]
    return pool_ids, pool_scaled, pool_raw


def pick_next_step_vector(
    mode: str,
    v_baseline: torch.Tensor,
    v_soft: torch.Tensor,
    v_soft_entropy: torch.Tensor,
    v_solver: torch.Tensor,
    v_centroid: torch.Tensor,
    v_coconut: torch.Tensor,
) -> torch.Tensor:
    """Returns [1, hidden_size] float32 on the same device as the inputs."""
    choices = {
        "discrete_top1": v_baseline,
        "topk_soft": v_soft,
        "topk_soft_entropy": v_soft_entropy,
        "solver": v_solver,
        "centroid": v_centroid,
        "coconut": v_coconut,
    }
    if mode not in choices:
        raise ValueError(
            f"Unknown next-step embedding mode {mode!r}; "
            f"expected one of {list(choices)}"
        )
    return choices[mode].detach().float()


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Latent comparison sequence: optionally feed a synthetic embedding "
            "as the next input instead of the top-1 token embedding."
        )
    )
    p.add_argument(
        "--next-step-embedding",
        default=NEXT_STEP_EMBEDDING_DEFAULT,
        choices=list(NEXT_STEP_LABELS.keys()),
        help=(
            "Vector concatenated to inputs_embeds for the following forward pass "
            f"(default: {NEXT_STEP_EMBEDDING_DEFAULT})."
        ),
    )
    p.add_argument(
        "--candidate-pool",
        choices=("topk", "topp"),
        default=CANDIDATE_POOL_DEFAULT,
        help=(
            "How to choose the candidate token set: fixed top-k by scaled prob, "
            "or top-p cumulative mass on scaled probs (see --top-p)."
        ),
    )
    p.add_argument(
        "-k",
        "--k",
        type=int,
        default=None,
        metavar="K",
        help=f"Candidate pool size when --candidate-pool=topk (default: {K}).",
    )
    p.add_argument(
        "--top-p",
        type=float,
        default=TOP_P_DEFAULT,
        metavar="P",
        help=(
            "When --candidate-pool=topp: include tokens in descending scaled-prob "
            "order while running cumulative mass is <= P (always >= 1 token)."
        ),
    )
    p.add_argument(
        "--top-p-max-tokens",
        type=int,
        default=TOP_P_MAX_TOKENS_DEFAULT,
        metavar="N",
        help="Optional max pool size for top-p (truncates highest-prob prefix).",
    )
    return p.parse_args()


def run_latent_comparison_sequence(
    next_step_embedding: str,
    *,
    candidate_pool: str = CANDIDATE_POOL_DEFAULT,
    k: int | None = None,
    top_p: float = TOP_P_DEFAULT,
    top_p_max_tokens: int | None = TOP_P_MAX_TOKENS_DEFAULT,
) -> None:
    print(f"Loading model: {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16
    )
    model.eval()
    device = next(model.parameters()).device

    # --- EMBEDDING DICTIONARY ---
    raw_embeddings = model.get_input_embeddings().weight.detach().to(torch.float32)
    norm_dictionary = F.normalize(raw_embeddings, p=2, dim=1)

    # Create a bfloat16 version of raw embeddings for dot-product evaluation
    raw_embeddings_bf16 = raw_embeddings.to(torch.bfloat16)

    embed_layer = model.get_input_embeddings()
    prompt_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(device)
    inputs_embeds = embed_layer(prompt_ids).to(model.dtype)

    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.width", 1000)

    k_eff = K if k is None else k

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        emit("LLAMA 8B LATENT COMPARISON SEQUENCE LOG", f)
        emit(
            "Mode: Natural Autoregressive Generation with Synthetic Vector Evaluation.",
            f,
        )
        if candidate_pool == "topk":
            emit(
                f"Steps: {STEPS} | Candidate pool: top-{k_eff} (scaled softmax) | "
                f"Table display base: {DISPLAY_K} rows (grows with pool size) | "
                f"Temp: {TEMPERATURE}",
                f,
            )
        else:
            cap_msg = (
                f" | max {top_p_max_tokens} tokens/step"
                if top_p_max_tokens is not None
                else ""
            )
            emit(
                f"Steps: {STEPS} | Candidate pool: top-p (cumulative scaled mass <= {top_p}, "
                f"min 1 token){cap_msg} | Table display scales with pool | Temp: {TEMPERATURE}",
                f,
            )
        emit(
            f"Next-step model input: {next_step_embedding} — {NEXT_STEP_LABELS[next_step_embedding]}",
            f,
        )
        emit(
            "(Context suffix in each step uses greedy top-1 decode for readability; "
            "the actual next-position tensor fed to the model is as above.)",
            f,
        )
        emit(
            f"Entropy-bounded soft-thinking target range: [{ENTROPY_MIN:.3f}, {ENTROPY_MAX:.3f}]",
            f,
        )
        emit("", f)

        context_so_far = PROMPT
        greedy_continuation_ids: list[int] = []

        with torch.no_grad():
            for step in range(STEPS):
                emit(f"\n{'='*160}", f)
                emit(f"STEP {step + 1} | Context: '...{context_so_far[-60:]}'", f)
                emit(f"{'='*160}", f)

                # 1. Forward Pass (full sequence as embeddings)
                outputs = model(inputs_embeds=inputs_embeds, output_hidden_states=True)

                # 2. Extract Logits
                next_token_logits = outputs.logits[0, -1, :].to(torch.float32)
                greedy_continuation_ids.append(
                    int(next_token_logits.argmax(dim=-1).item())
                )

                # --- TEMPERATURE SCALING ON THE FULL UNIVERSE ---
                scaled_next_token_logits = next_token_logits / TEMPERATURE

                # --- GLOBAL RANK CALCULATION ---
                sorted_idx = torch.argsort(scaled_next_token_logits, descending=True)
                global_ranks = torch.empty_like(sorted_idx)
                global_ranks[sorted_idx] = torch.arange(
                    1, len(scaled_next_token_logits) + 1, device=device
                )

                # --- FULL VOCABULARY PROBABILITIES ---
                raw_vocab_probs = F.softmax(next_token_logits, dim=-1)
                full_vocab_probs = F.softmax(scaled_next_token_logits, dim=-1)

                pool_ids, pool_scaled_probs, pool_raw_probs = select_candidate_pool(
                    full_vocab_probs,
                    raw_vocab_probs,
                    candidate_pool,
                    k_eff,
                    top_p,
                    top_p_max_tokens,
                )
                n_pool = int(pool_ids.shape[0])
                pool_mass = float(pool_scaled_probs.sum().item())
                display_k = min(
                    full_vocab_probs.shape[-1],
                    max(DISPLAY_K, 2 * n_pool, 10),
                )

                pool_words = tokenizer.batch_decode(pool_ids)
                target_ids_list = pool_ids.tolist()

                # Pool logits for entropy-bounded softmax; T=1 masses for the solver
                pool_logits = next_token_logits[pool_ids]
                solver_pool_probs = pool_raw_probs / pool_raw_probs.sum()

                greedy_word = pool_words[0]
                context_so_far += greedy_word

                if candidate_pool == "topp":
                    emit(
                        f"--- POOL (top-p) | n={n_pool} | cumulative scaled mass in pool: {pool_mass:.4f} (cap p={top_p}) ---",
                        f,
                    )

                # --- RENORMALIZE PROBS OVER POOL (same as former top-k block) ---
                adj_probs = pool_scaled_probs / pool_scaled_probs.sum(dim=-1)
                adj_entropy = entropy_from_probs(adj_probs).item()

                # --- ENTROPY-BOUNDED TEMPERATURE-ADJUSTED SOFT THINKING ON POOL LOGITS ---
                tau_entropy, entropy_adj_probs, bounded_entropy = (
                    solve_tau_for_entropy_range(
                        logits=pool_logits,
                        entropy_min=ENTROPY_MIN,
                        entropy_max=ENTROPY_MAX,
                        tau_min=TAU_MIN,
                        tau_max=TAU_MAX,
                        steps=TAU_BISECTION_STEPS,
                    )
                )

                emit(f"--- PREDICTED LOGIT DISTRIBUTION ---", f)
                emit(
                    f"Standard pool ({n_pool} tok) renormalized entropy: {adj_entropy:.4f}",
                    f,
                )
                emit(
                    f"Entropy-bounded pool ({n_pool} tok) temperature: tau={tau_entropy:.4f} | entropy={bounded_entropy:.4f}",
                    f,
                )
                for rank, (word, raw_p, adj_p, ent_p) in enumerate(
                    zip(pool_words, pool_raw_probs, adj_probs, entropy_adj_probs)
                ):
                    clean_word = word.replace("\n", "\\n").replace("\r", "\\r")
                    emit(
                        f"  Rank {rank+1:2d} | Raw: {raw_p*100:5.2f}% | "
                        f"Scaled (Sum=1): {adj_p*100:5.2f}% | "
                        f"Entropy-Adj: {ent_p*100:5.2f}% | '{clean_word}'",
                        f,
                    )

                # --- VECTOR SYNTHESIS ---
                pool_raw_embs = raw_embeddings[pool_ids]
                avg_target_mag = torch.norm(pool_raw_embs, p=2, dim=1).mean().item()

                # 0. Baseline (Top-1 Token)
                v_baseline = pool_raw_embs[0:1]

                # 1. Soft mix over pool
                v_soft = torch.sum(
                    pool_raw_embs * adj_probs.unsqueeze(1), dim=0, keepdim=True
                )

                # 1b. Entropy-bounded temperature-adjusted soft thinking on pool
                v_soft_entropy = torch.sum(
                    pool_raw_embs * entropy_adj_probs.unsqueeze(1), dim=0, keepdim=True
                )

                # 2. Geometric Solver
                target_embs_norm = norm_dictionary[pool_ids]
                with torch.enable_grad():
                    v_solver = fast_geometric_solver(
                        target_embs_norm,
                        pool_ids,
                        norm_dictionary,
                        avg_target_mag,
                        solver_pool_probs,
                        temperature=TEMPERATURE,
                    )

                # 3. Centroid (unweighted mean over pool)
                v_centroid = torch.mean(pool_raw_embs, dim=0, keepdim=True)

                # 4. Coconut Vector (Continuous Hidden State)
                v_coconut = (
                    outputs.hidden_states[-1][0, -1, :].unsqueeze(0).to(torch.float32)
                )

                next_vec = pick_next_step_vector(
                    next_step_embedding,
                    v_baseline,
                    v_soft,
                    v_soft_entropy,
                    v_solver,
                    v_centroid,
                    v_coconut,
                )
                next_vec_bf16 = next_vec.to(device=device, dtype=model.dtype)
                inputs_embeds = torch.cat(
                    [inputs_embeds, next_vec_bf16.unsqueeze(0)], dim=1
                )

                clean_greedy = greedy_word.replace("\n", "\\n").replace("\r", "\\r")
                emit(
                    f"--- NEXT-POSITION INPUT --- key={next_step_embedding} | "
                    f"{NEXT_STEP_LABELS[next_step_embedding]} | "
                    f"greedy context token: '{clean_greedy}'",
                    f,
                )

                # --- COMPARATIVE ANALYSIS ---
                vectors_unnormalized = torch.cat(
                    [
                        v_baseline,
                        v_soft,
                        v_soft_entropy,
                        v_solver,
                        v_centroid,
                        v_coconut,
                    ],
                    dim=0,
                )
                vectors_unit = F.normalize(vectors_unnormalized, p=2, dim=1)
                mags = torch.norm(vectors_unnormalized, p=2, dim=1).cpu().numpy()

                labels = [
                    "Top-1 Token",
                    f"Soft (n={n_pool})",
                    f"Soft-Entropy (n={n_pool})",
                    f"Solver (n={n_pool})",
                    f"Centroid (n={n_pool})",
                    "Coconut",
                ]

                sim_matrix = torch.matmul(vectors_unit, vectors_unit.T)
                df = pd.DataFrame(
                    sim_matrix.cpu().numpy(), index=labels, columns=labels
                )

                emit(f"\n--- CROSS-SIMILARITY MATRIX ---", f)
                emit(df.to_string(), f)

                col_width = 38

                # =====================================================================
                # TABLE 1: COSINE SIMILARITY
                # =====================================================================
                emit(
                    f"\n--- [TABLE 1] TOP {display_k} NEAREST NEIGHBORS (COSINE SIMILARITY) ---",
                    f,
                )
                cos_sims_all = torch.matmul(vectors_unit, norm_dictionary.T)
                top_k_vals_cos, top_k_idx_cos = torch.topk(
                    cos_sims_all, display_k, dim=1
                )

                header_cos = f"{'Rank':<4} |"
                for label in labels:
                    header_title = f"{label} (Ang)"
                    header_cos += f" {header_title:<{col_width}} |"
                emit(header_cos[:-2], f)
                emit("-" * len(header_cos[:-2]), f)

                for rank in range(display_k):
                    row_str = f"{rank+1:<4} |"
                    for i in range(len(labels)):
                        token_id = top_k_idx_cos[i, rank].item()
                        w = tokenizer.decode([token_id])
                        cw = w.replace("\n", "\\n").replace("\r", "\\r")
                        if len(cw) > 10:
                            cw = cw[:8] + ".."

                        orig_rank = global_ranks[token_id].item()
                        marker = "*" if token_id in target_ids_list else " "
                        val = top_k_vals_cos[i, rank].item()

                        cell = f"{orig_rank:<6} {marker}'{cw}' ({val:.3f})"
                        row_str += f" {cell:<{col_width}} |"
                    emit(row_str[:-2], f)

                # =====================================================================
                # TABLE 2: UNNORMALIZED DOT PRODUCT
                # =====================================================================
                emit(
                    f"\n--- [TABLE 2] TOP {display_k} TOKENS BY MAGNITUDE (UNNORMALIZED DOT PRODUCT) ---",
                    f,
                )

                vectors_unnormalized_bf16 = vectors_unnormalized.to(torch.bfloat16)
                dot_prods_all = torch.matmul(
                    vectors_unnormalized_bf16, raw_embeddings_bf16.T
                )
                top_k_vals_dot, top_k_idx_dot = torch.topk(
                    dot_prods_all, display_k, dim=1
                )

                header_dot = f"{'Rank':<4} |"
                for label, mag in zip(labels, mags):
                    header_title = f"{label} (Mag: {mag:.1f})"
                    header_dot += f" {header_title:<{col_width}} |"
                emit(header_dot[:-2], f)
                emit("-" * len(header_dot[:-2]), f)

                for rank in range(display_k):
                    row_str = f"{rank+1:<4} |"
                    for i in range(len(labels)):
                        token_id = top_k_idx_dot[i, rank].item()
                        w = tokenizer.decode([token_id])
                        cw = w.replace("\n", "\\n").replace("\r", "\\r")
                        if len(cw) > 10:
                            cw = cw[:8] + ".."

                        orig_rank = global_ranks[token_id].item()
                        marker = "*" if token_id in target_ids_list else " "
                        val = top_k_vals_dot[i, rank].item()

                        cell = f"{orig_rank:<6} {marker}'{cw}' ({val:>7.3f})"
                        row_str += f" {cell:<{col_width}} |"
                    emit(row_str[:-2], f)

        greedy_prompt_plus_cont_ids = prompt_ids[0].tolist() + greedy_continuation_ids
        greedy_full_text = tokenizer.decode(
            greedy_prompt_plus_cont_ids, skip_special_tokens=True
        )
        greedy_cont_only = tokenizer.decode(
            greedy_continuation_ids, skip_special_tokens=True
        )

        def _escape_for_log(s: str) -> str:
            return s.replace("\n", "\\n").replace("\r", "\\r")

        emit("", f)
        emit("=" * 160, f)
        emit(
            "GREEDY SEQUENCE (DOES NOT DEPEND ON --next-step-embedding)",
            f,
        )
        emit(
            "At each step, token = argmax(next_token_logits), i.e. the most likely token "
            "under softmax(logits) at temperature 1 (raw logits, not TEMPERATURE-scaled).",
            f,
        )
        emit(
            f"Note: per-step 'greedy context token' in the log uses rank-1 under "
            f"softmax(logits/{TEMPERATURE}), which can differ when TEMPERATURE != 1.",
            f,
        )
        emit(
            f"Continuation ({len(greedy_continuation_ids)} tokens): "
            f"{_escape_for_log(greedy_cont_only)}",
            f,
        )
        emit(
            f"Full text (prompt + continuation): {_escape_for_log(greedy_full_text)}",
            f,
        )
        emit("=" * 160, f)

        emit(f"\nProcessing complete. Log file saved to {LOG_FILE}.", f)


if __name__ == "__main__":
    _args = parse_args()
    run_latent_comparison_sequence(
        _args.next_step_embedding,
        candidate_pool=_args.candidate_pool,
        k=_args.k,
        top_p=_args.top_p,
        top_p_max_tokens=_args.top_p_max_tokens,
    )
