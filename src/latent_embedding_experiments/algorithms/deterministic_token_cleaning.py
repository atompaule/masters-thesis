"""
token_cleaning_diagnostic.py

Compares deterministic token cleaning strategies on individual token embeddings
from LLaMA 3.1 8B's embedding matrix.

Interlopers are derived automatically: the top-N nearest neighbors of the target
token in the embedding matrix (by cosine similarity, excluding the token itself).
No hardcoded interloper lists are needed.

Cleaning methods
----------------
1. raw              — unmodified target embedding (baseline)
2. gs_deflation     — Gram-Schmidt sequential deflation against each interloper
3. subspace_proj    — project onto orthogonal complement of interloper subspace
4. slerp_repulsion  — normalize(e + alpha * sum(e - e_int))
5. mean_repulsion   — normalize(e + alpha * (e - mean(interlopers)))

Usage
-----
    python token_cleaning_diagnostic.py
    python token_cleaning_diagnostic.py --token ability --n-interlopers 10 --alpha 0.5
    python token_cleaning_diagnostic.py --token possible --display-k 30
"""

import argparse

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
DISPLAY_K_DEFAULT = 25
ALPHA_DEFAULT = 0.3      # repulsion strength for slerp / mean methods
BETA_DEFAULT = 0.5           # pull-back strength for subspace_proj_boosted
N_INTERLOPERS_DEFAULT = 10  # how many nearest neighbors to treat as interlopers

LOG_FILE = "src/latent_embedding_experiments/logs/token_cleaning_diagnostic.txt"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def emit(text: str, fh=None) -> None:
    print(text)
    if fh:
        fh.write(text + "\n")


def token_to_id(token_str: str, tokenizer) -> int:
    """Return the single token id for a plain string (no BOS, no spaces)."""
    ids = tokenizer.encode(token_str, add_special_tokens=False)
    if len(ids) != 1:
        # Try with a leading space (common for continuation tokens in LLaMA)
        ids_space = tokenizer.encode(" " + token_str, add_special_tokens=False)
        if len(ids_space) == 1:
            return ids_space[0]
        raise ValueError(
            f"'{token_str}' encodes to {len(ids)} tokens {ids}; "
            "choose a single-token word."
        )
    return ids[0]


def derive_interlopers(
    target_id: int,
    vocab_embs_norm: torch.Tensor,
    n: int,
) -> list[int]:
    """Return the ids of the top-n nearest neighbors of target_id by cosine similarity.

    The target itself is excluded. These are the embedding-space interlopers —
    tokens that are geometrically closest to the target regardless of semantics.
    """
    target_norm = vocab_embs_norm[target_id]  # [d]
    cos_sims = vocab_embs_norm @ target_norm  # [V]
    cos_sims[target_id] = -1.0  # exclude self
    top_ids = torch.topk(cos_sims, n).indices.tolist()
    return top_ids




# ---------------------------------------------------------------------------
# Cleaning methods
# ---------------------------------------------------------------------------

def clean_raw(e: torch.Tensor, interlopers: torch.Tensor, **_) -> torch.Tensor:
    """Baseline: no cleaning."""
    return e.clone()


def clean_gs_deflation(
    e: torch.Tensor, interlopers: torch.Tensor, **_
) -> torch.Tensor:
    """Gram-Schmidt sequential deflation: subtract each interloper's component."""
    v = e.clone()
    for ei in interlopers:
        ei_hat = F.normalize(ei.unsqueeze(0), dim=1).squeeze(0)
        v = v - (v @ ei_hat) * ei_hat
    return F.normalize(v.unsqueeze(0), dim=1).squeeze(0)


def clean_subspace_proj(
    e: torch.Tensor, interlopers: torch.Tensor, **_
) -> torch.Tensor:
    """Project onto orthogonal complement of the full interloper subspace via QR."""
    A = interlopers.T  # [d, n_int]
    Q, _ = torch.linalg.qr(A)  # Q: [d, r], r ≤ n_int
    projection = Q @ (Q.T @ e)
    v = e - projection
    return F.normalize(v.unsqueeze(0), dim=1).squeeze(0)


def clean_subspace_proj_boosted(
    e: torch.Tensor, interlopers: torch.Tensor, beta: float = BETA_DEFAULT, **_
) -> torch.Tensor:
    """Subspace projection + pull back toward the original token.

    After projecting out the interloper subspace, add beta * e to pull the
    result back toward the original token direction. beta=0 reduces to plain
    subspace_proj. Higher beta increases self-similarity at the cost of
    partially reintroducing interloper similarity.

    v = normalize((e - projection) + beta * e)
    """
    A = interlopers.T
    Q, _ = torch.linalg.qr(A)
    projection = Q @ (Q.T @ e)
    v = (e - projection) + beta * e
    return F.normalize(v.unsqueeze(0), dim=1).squeeze(0)


def clean_slerp_repulsion(
    e: torch.Tensor, interlopers: torch.Tensor, alpha: float = ALPHA_DEFAULT, **_
) -> torch.Tensor:
    """Amplify the direction away from each interloper individually then sum.

    v = normalize(e + alpha * sum_j(e - e_j))
      = normalize((1 + n*alpha)*e - alpha * sum_j(e_j))
    """
    diff_sum = torch.stack([e - ei for ei in interlopers], dim=0).sum(dim=0)
    v = e + alpha * diff_sum
    return F.normalize(v.unsqueeze(0), dim=1).squeeze(0)

METHODS = {
    "raw": clean_raw,
    "gs_deflation": clean_gs_deflation,
    "subspace_proj": clean_subspace_proj,
    "subspace_proj_boosted": clean_subspace_proj_boosted,
    "slerp_repulsion": clean_slerp_repulsion,
}

METHOD_LABELS = {
    "raw": "Raw (baseline)",
    "gs_deflation": "Gram-Schmidt deflation",
    "subspace_proj": "Subspace projection",
    "subspace_proj_boosted": f"Subspace proj boosted (β={BETA_DEFAULT})",
    "slerp_repulsion": f"Slerp repulsion (α={ALPHA_DEFAULT})",
}

# ---------------------------------------------------------------------------
# Table printing
# ---------------------------------------------------------------------------

def print_nn_table(
    cleaned_vecs: dict[str, torch.Tensor],
    vocab_embs_norm: torch.Tensor,
    tokenizer,
    interloper_ids: set[int],
    target_id: int,
    display_k: int,
    fh=None,
) -> None:
    """Print a nearest-neighbor cosine similarity table across all cleaning methods."""
    method_names = list(cleaned_vecs.keys())
    labels = [METHOD_LABELS[m] for m in method_names]

    # Stack all vectors and compute cosine sims to full vocab
    vecs = torch.stack([cleaned_vecs[m] for m in method_names], dim=0)  # [M, d]
    vecs_norm = F.normalize(vecs, p=2, dim=1)
    cos_sims = vecs_norm @ vocab_embs_norm.T  # [M, V]
    top_vals, top_idx = torch.topk(cos_sims, display_k, dim=1)

    col_width = 42
    header = f"{'Rank':<4} |"
    for label in labels:
        header += f" {label:<{col_width}} |"
    header = header[:-2]

    emit(header, fh)
    emit("-" * len(header), fh)

    for rank in range(display_k):
        row = f"{rank+1:<4} |"
        for i, method in enumerate(method_names):
            tid = top_idx[i, rank].item()
            val = top_vals[i, rank].item()
            word = (
                tokenizer.decode([tid])
                .replace("\n", "\\n")
                .replace("\r", "\\r")
            )
            if len(word) > 14:
                word = word[:12] + ".."
            marker = "◆" if tid == target_id else ("✗" if tid in interloper_ids else " ")
            cell = f"{marker} '{word}' ({val:.4f})"
            row += f" {cell:<{col_width}} |"
        emit(row[:-2], fh)


def print_interloper_sim_summary(
    cleaned_vecs: dict[str, torch.Tensor],
    interloper_ids: list[int],
    vocab_embs_norm: torch.Tensor,
    tokenizer,
    target_id: int,
    fh=None,
) -> None:
    """Print a summary table: similarity to target and each interloper per method."""
    method_names = list(cleaned_vecs.keys())
    vecs = torch.stack([cleaned_vecs[m] for m in method_names], dim=0)  # [M, d]
    vecs_norm = F.normalize(vecs, p=2, dim=1)

    check_ids = [target_id] + interloper_ids
    check_embs = vocab_embs_norm[check_ids]  # [1 + n_int, d]
    sims = (vecs_norm @ check_embs.T).cpu()  # [M, 1 + n_int]

    check_words = [tokenizer.decode([tid]) for tid in check_ids]
    col_w = 16

    header = f"{'Method':<22} |"
    for w in check_words:
        w_clean = w.strip().replace("\n", "\\n")
        if len(w_clean) > col_w - 2:
            w_clean = w_clean[: col_w - 4] + ".."
        header += f" {w_clean:^{col_w}} |"
    header = header[:-2]

    emit(header, fh)
    emit("-" * len(header), fh)

    for i, method in enumerate(method_names):
        row = f"{method:<22} |"
        for j in range(len(check_ids)):
            val = sims[i, j].item()
            # Highlight: target col = + sign, interloper cols show delta vs raw
            if j == 0:
                cell = f"{val:+.4f}"
            else:
                delta = val - sims[0, j].item()  # delta vs raw
                cell = f"{val:.4f} ({delta:+.4f})"
            row += f" {cell:^{col_w}} |"
        emit(row[:-2], fh)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Deterministic token cleaning diagnostic.")
    p.add_argument(
        "--token",
        default="ability",
        help="Target token to clean (any single-token word).",
    )
    p.add_argument(
        "--n-interlopers",
        type=int,
        default=N_INTERLOPERS_DEFAULT,
        help="Number of nearest neighbors to use as interlopers (default: 10).",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=ALPHA_DEFAULT,
        help="Repulsion strength for slerp_repulsion and mean_repulsion.",
    )
    p.add_argument(
        "--beta",
        type=float,
        default=BETA_DEFAULT,
        help="Pull-back strength for subspace_proj_boosted (default: 1.0).",
    )
    p.add_argument(
        "--display-k",
        type=int,
        default=DISPLAY_K_DEFAULT,
        help="Number of nearest neighbors to display.",
    )
    return p.parse_args()


def run(token_str: str, n_interlopers: int, alpha: float, beta: float, display_k: int) -> None:
    print(f"Loading model: {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16
    )
    model.eval()

    vocab_embs: torch.Tensor = (
        model.get_input_embeddings().weight.detach().to(torch.float32)
    )
    vocab_embs_norm = F.normalize(vocab_embs, p=2, dim=1)

    # Resolve target token
    target_id = token_to_id(token_str, tokenizer)
    target_emb = vocab_embs[target_id]  # [d]
    target_word = tokenizer.decode([target_id])

    # Derive interlopers automatically from the embedding matrix
    print(f"Deriving top-{n_interlopers} nearest neighbors as interlopers ...")
    interloper_ids = derive_interlopers(target_id, vocab_embs_norm, n=n_interlopers)
    interloper_ids_set = set(interloper_ids)
    interloper_embs = vocab_embs[interloper_ids]  # [n, d]
    interloper_words = [tokenizer.decode([i]) for i in interloper_ids]

    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.width", 1200)

    with open(LOG_FILE, "w", encoding="utf-8") as fh:
        emit("=" * 120, fh)
        emit("TOKEN CLEANING DIAGNOSTIC", fh)
        emit(f"Model        : {MODEL_ID}", fh)
        emit(f"Target token : '{target_word}' (id={target_id})", fh)
        emit(f"Interlopers  : top-{n_interlopers} nearest neighbors (by cosine sim)", fh)
        emit(f"             : {list(zip(interloper_words, interloper_ids))}", fh)
        emit(f"Alpha        : {alpha}  (for slerp_repulsion / mean_repulsion)", fh)
        emit(f"Beta         : {beta}  (for subspace_proj_boosted)", fh)
        emit(f"Display k    : {display_k}", fh)
        emit("=" * 120, fh)

        # --- Apply all cleaning methods ---
        cleaned_vecs: dict[str, torch.Tensor] = {}
        for method_name, fn in METHODS.items():
            cleaned_vecs[method_name] = fn(
                target_emb,
                interloper_embs,
                alpha=alpha,
                beta=beta,
            )
        # Update labels with actual runtime values
        for method_name in METHODS:
            if "repulsion" in method_name:
                METHOD_LABELS[method_name] = METHOD_LABELS[method_name].replace(
                    str(ALPHA_DEFAULT), str(alpha)
                )
            if "boosted" in method_name:
                METHOD_LABELS[method_name] = METHOD_LABELS[method_name].replace(
                    str(BETA_DEFAULT), str(beta)
                )

        # --- Cross-similarity between cleaned vectors ---
        labels = [METHOD_LABELS[m] for m in METHODS]
        vecs = torch.stack(list(cleaned_vecs.values()), dim=0)
        vecs_norm = F.normalize(vecs, p=2, dim=1)
        sim_matrix = (vecs_norm @ vecs_norm.T).cpu().numpy()
        df = pd.DataFrame(sim_matrix, index=list(METHODS.keys()), columns=list(METHODS.keys()))
        emit("\n--- CROSS-SIMILARITY BETWEEN CLEANED VECTORS ---", fh)
        emit(df.to_string(), fh)

        # --- Interloper similarity summary ---
        emit("\n--- SIMILARITY TO TARGET AND INTERLOPERS (delta vs raw in parentheses) ---", fh)
        emit(
            f"  ◆ = target token '{target_word}'  |  columns: target, then each interloper",
            fh,
        )
        print_interloper_sim_summary(
            cleaned_vecs,
            interloper_ids,
            vocab_embs_norm,
            tokenizer,
            target_id,
            fh,
        )

        # --- Full nearest-neighbor table ---
        emit(
            f"\n--- TOP {display_k} NEAREST NEIGHBORS (COSINE) ---", fh
        )
        emit(
            f"  ◆ = target token  |  ✗ = known interloper  |  space = other token",
            fh,
        )
        print_nn_table(
            cleaned_vecs,
            vocab_embs_norm,
            tokenizer,
            interloper_ids_set,
            target_id,
            display_k,
            fh,
        )

        emit(f"\nLog saved to {LOG_FILE}", fh)


if __name__ == "__main__":
    args = parse_args()
    run(
        token_str=args.token,
        n_interlopers=args.n_interlopers,
        alpha=args.alpha,
        beta=args.beta,
        display_k=args.display_k,
    )