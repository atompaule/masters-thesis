"""
token_cleaning_diagnostic.py

Compares subspace_proj_slerp cleaning across a sweep of (n_interlopers, target_sim)
combinations, alongside the raw baseline. All combinations are shown side-by-side
in the nearest-neighbor and interloper similarity tables.

target_sim is the desired cosine similarity between the cleaned token and the
original token. The slerp interpolation finds the exact point on the unit sphere
that achieves this similarity while maximally removing interloper signal.

Usage
-----
    python token_cleaning_diagnostic.py
    python token_cleaning_diagnostic.py --token ability
    python token_cleaning_diagnostic.py --token possible \\
        --n-interlopers-list 5 10 20 --target-sim-list 0.7 0.8 0.9 0.95
"""

import argparse
import itertools

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
DISPLAY_K_DEFAULT = 25

# Swept values — all (n_interlopers, target_sim) combinations shown as columns
N_INTERLOPERS = [3, 10, 20]
TARGET_SIM_SWEEP = [0.90, 0.97]

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
    """Return the ids of the top-n nearest neighbors of target_id (excluding self)."""
    target_norm = vocab_embs_norm[target_id]
    cos_sims = vocab_embs_norm @ target_norm
    cos_sims[target_id] = -1.0
    return torch.topk(cos_sims, n).indices.tolist()


# ---------------------------------------------------------------------------
# Cleaning methods
# ---------------------------------------------------------------------------

def clean_raw(e: torch.Tensor, **_) -> torch.Tensor:
    """Baseline: no cleaning, returned as normalized unit vector."""
    return F.normalize(e.unsqueeze(0), dim=1).squeeze(0)


def clean_subspace_proj_slerp(
    e: torch.Tensor,
    interlopers: torch.Tensor,
    target_sim: float = 0.90,
    **_,
) -> torch.Tensor:
    """Project out interloper subspace, then slerp back toward the original token.

    After projection, slerp interpolates on the unit sphere between the pure
    projection result (e_tilde) and the original token (e_hat). The parameter
    target_sim directly specifies the desired cosine similarity to the original
    token — no arbitrary beta, geometrically exact.

    alpha=0 → pure projection result (zero interloper sim, lowest self-sim).
    alpha=1 → original token (full self-sim, full interloper sim restored).

    The alpha that achieves target_sim is solved analytically:
        alpha = 1 - arccos(target_sim) / phi
    where phi = arccos(e_tilde · e_hat) is the angle between the two endpoints.
    """
    A = interlopers.T
    Q, _ = torch.linalg.qr(A)
    projection = Q @ (Q.T @ e)
    e_tilde = e - projection                                          # raw residual [d]

    e_hat   = F.normalize(e.unsqueeze(0), dim=1).squeeze(0)          # original, normalized
    e_t_hat = F.normalize(e_tilde.unsqueeze(0), dim=1).squeeze(0)    # projection result, normalized

    cos_phi = (e_t_hat @ e_hat).clamp(-1.0, 1.0)
    phi = torch.arccos(cos_phi)                                       # angle between endpoints

    if phi < 1e-6:        # already nearly identical — projection changed nothing
        return e_t_hat

    # Solve for alpha such that slerp(alpha) has cosine similarity = target_sim to e_hat
    alpha = 1.0 - torch.arccos(torch.tensor(target_sim, dtype=phi.dtype)).item() / phi.item()
    alpha = max(0.0, min(1.0, alpha))   # clamp: if target_sim > cos_phi, use pure projection

    v = torch.sin((1.0 - alpha) * phi) * e_t_hat + torch.sin(alpha * phi) * e_hat
    return F.normalize(v.unsqueeze(0), dim=1).squeeze(0)


# ---------------------------------------------------------------------------
# Table printing
# ---------------------------------------------------------------------------

def print_nn_table(
    cleaned_vecs: dict[str, torch.Tensor],
    method_labels: dict[str, str],
    vocab_embs_norm: torch.Tensor,
    tokenizer,
    interloper_ids_union: set[int],
    target_id: int,
    display_k: int,
    fh=None,
) -> None:
    method_names = list(cleaned_vecs.keys())
    labels = [method_labels[m] for m in method_names]

    vecs = torch.stack([cleaned_vecs[m] for m in method_names], dim=0)  # [M, d]
    vecs_norm = F.normalize(vecs, p=2, dim=1)
    cos_sims = vecs_norm @ vocab_embs_norm.T                             # [M, V]
    top_vals, top_idx = torch.topk(cos_sims, display_k, dim=1)

    col_width = 38
    header = f"{'Rank':<4} |"
    for label in labels:
        header += f" {label:<{col_width}} |"
    header = header[:-2]

    emit(header, fh)
    emit("-" * len(header), fh)

    for rank in range(display_k):
        row = f"{rank+1:<4} |"
        for i in range(len(method_names)):
            tid = top_idx[i, rank].item()
            val = top_vals[i, rank].item()
            word = tokenizer.decode([tid]).replace("\n", "\\n").replace("\r", "\\r")
            if len(word) > 14:
                word = word[:12] + ".."
            marker = "◆" if tid == target_id else ("✗" if tid in interloper_ids_union else " ")
            cell = f"{marker} '{word}' ({val:.4f})"
            row += f" {cell:<{col_width}} |"
        emit(row[:-2], fh)


def print_interloper_sim_summary(
    cleaned_vecs: dict[str, torch.Tensor],
    method_labels: dict[str, str],
    interloper_ids: list[int],
    vocab_embs_norm: torch.Tensor,
    tokenizer,
    target_id: int,
    fh=None,
) -> None:
    method_names = list(cleaned_vecs.keys())
    vecs = torch.stack([cleaned_vecs[m] for m in method_names], dim=0)  # [M, d]
    vecs_norm = F.normalize(vecs, p=2, dim=1)

    check_ids = [target_id] + interloper_ids
    check_embs = vocab_embs_norm[check_ids]
    sims = (vecs_norm @ check_embs.T).cpu()  # [M, 1+n]

    check_words = [tokenizer.decode([tid]) for tid in check_ids]
    col_w = 18

    header = f"{'Method':<20} |"
    for w in check_words:
        w_clean = w.strip().replace("\n", "\\n")
        if len(w_clean) > col_w - 2:
            w_clean = w_clean[:col_w - 4] + ".."
        header += f" {w_clean:^{col_w}} |"
    header = header[:-2]

    emit(header, fh)
    emit("-" * len(header), fh)

    for i, method in enumerate(method_names):
        row = f"{method_labels[method]:<20} |"
        for j in range(len(check_ids)):
            val = sims[i, j].item()
            if j == 0:
                cell = f"{val:+.4f}"
            else:
                delta = val - sims[0, j].item()   # delta vs raw
                cell = f"{val:.4f} ({delta:+.4f})"
            row += f" {cell:^{col_w}} |"
        emit(row[:-2], fh)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Token cleaning parameter sweep.")
    p.add_argument("--token", default=" ability",
                   help="Target token to clean (any single-token word).")
    p.add_argument("--n-interlopers-list", type=int, nargs="+",
                   default=N_INTERLOPERS_SWEEP,
                   help="List of n_interlopers values to sweep.")
    p.add_argument("--target-sim-list", type=float, nargs="+",
                   default=TARGET_SIM_SWEEP,
                   help="List of target self-similarity values to sweep (e.g. 0.7 0.8 0.9).")
    p.add_argument("--display-k", type=int, default=DISPLAY_K_DEFAULT,
                   help="Number of nearest neighbors to display.")
    return p.parse_args()


def run(
    token_str: str,
    n_interlopers_list: list[int],
    target_sim_list: list[float],
    display_k: int,
) -> None:
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

    target_id = token_to_id(token_str, tokenizer)
    target_emb = vocab_embs[target_id]
    target_word = tokenizer.decode([target_id])

    # Derive interloper sets — one per n, plus union for NN table markers
    max_n = max(n_interlopers_list)
    print(f"Deriving interloper sets (max n={max_n}) ...")
    interloper_ids_cache: dict[int, list[int]] = {}
    for n in n_interlopers_list:
        interloper_ids_cache[n] = derive_interlopers(target_id, vocab_embs_norm, n=n)
    interloper_ids_max = interloper_ids_cache[max_n]
    interloper_ids_union: set[int] = set(interloper_ids_max)

    # Method labels
    method_labels: dict[str, str] = {"raw": "Raw (baseline)"}
    for n, s in itertools.product(n_interlopers_list, target_sim_list):
        method_labels[f"n{n}_s{s}"] = f"n_int={n:2d}  s*={s}"

    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.width", 1400)

    with open(LOG_FILE, "w", encoding="utf-8") as fh:
        emit("=" * 120, fh)
        emit("TOKEN CLEANING PARAMETER SWEEP", fh)
        emit(f"Model          : {MODEL_ID}", fh)
        emit(f"Target token   : '{target_word}' (id={target_id})", fh)
        emit(f"n_interlopers  : {n_interlopers_list}", fh)
        emit(f"target_sim     : {target_sim_list}", fh)
        emit(f"Columns        : raw + {len(method_labels)-1} (n × target_sim) combinations", fh)
        emit("=" * 120, fh)

        # --- Apply all methods ---
        cleaned_vecs: dict[str, torch.Tensor] = {}
        cleaned_vecs["raw"] = clean_raw(target_emb)

        for n, s in itertools.product(n_interlopers_list, target_sim_list):
            key = f"n{n}_s{s}"
            int_embs = vocab_embs[interloper_ids_cache[n]]
            cleaned_vecs[key] = clean_subspace_proj_slerp(target_emb, int_embs, target_sim=s)

        # --- Cross-similarity matrix ---
        vecs = torch.stack(list(cleaned_vecs.values()), dim=0)
        vecs_norm = F.normalize(vecs, p=2, dim=1)
        sim_matrix = (vecs_norm @ vecs_norm.T).cpu().numpy()
        keys = list(cleaned_vecs.keys())
        df = pd.DataFrame(sim_matrix, index=keys, columns=keys)
        emit("\n--- CROSS-SIMILARITY BETWEEN CLEANED VECTORS ---", fh)
        emit(df.to_string(), fh)

        # --- Interloper similarity summary (max-n interlopers as columns) ---
        emit("\n--- SIMILARITY TO TARGET AND INTERLOPERS (max-n set; delta vs raw) ---", fh)
        emit(f"  ◆ = target '{target_word}'  |  interloper columns = top-{max_n} neighbors", fh)
        print_interloper_sim_summary(
            cleaned_vecs,
            method_labels,
            interloper_ids_max,
            vocab_embs_norm,
            tokenizer,
            target_id,
            fh,
        )

        # --- Nearest-neighbor table ---
        emit(f"\n--- TOP {display_k} NEAREST NEIGHBORS (COSINE) ---", fh)
        emit("  ◆ = target token  |  ✗ = interloper (any n)  |  space = other", fh)
        print_nn_table(
            cleaned_vecs,
            method_labels,
            vocab_embs_norm,
            tokenizer,
            interloper_ids_union,
            target_id,
            display_k,
            fh,
        )

        emit(f"\nLog saved to {LOG_FILE}", fh)


if __name__ == "__main__":
    args = parse_args()
    run(
        token_str=args.token,
        n_interlopers_list=args.n_interlopers_list,
        target_sim_list=args.target_sim_list,
        display_k=args.display_k,
    )