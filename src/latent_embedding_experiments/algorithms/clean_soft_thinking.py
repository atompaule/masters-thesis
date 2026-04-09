"""
hyperparam_search_clean_soft_thinking.py

Searches for the best (n_interlopers, target_sim) hyperparameters for
clean soft thinking (subspace_proj_slerp per target token) by evaluating
the latent_head_loss on batches from the arcee-ai/LLama-405B-Logits dataset,
using the LLaMA 3.1 405B embedding matrix.

The "latent" fed to latent_head_loss is the clean soft thinking vector — a
[B, L, d] embedding constructed deterministically from the logit distribution
and evaluated against the same loss terms used to train the latent head. This
gives a principled, task-aligned score for each (n_interlopers, target_sim) pair.

Usage
-----
    python hyperparam_search_clean_soft_thinking.py
    python hyperparam_search_clean_soft_thinking.py \\
        --n-interlopers-list 5 10 20 \\
        --target-sim-list 0.7 0.8 0.9 0.95 \\
        --n-batches 50 --batch-size 4
"""

import argparse
import itertools
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from src.latent_embedding_experiments.algorithms.config import CFG
from src.latent_embedding_experiments.algorithms.deterministic_token_cleaning import (
    clean_subspace_proj_slerp,
    derive_interlopers,
)
from src.latent_embedding_experiments.algorithms.solver import latent_head_loss
from src.latent_embedding_experiments.algorithms.utils import select_targets

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EMBEDDINGS_MODEL_ID = "meta-llama/Llama-3.1-405B-Instruct"
DATASET_ID = "arcee-ai/LLama-405B-Logits"

N_INTERLOPERS_SWEEP = [10]
TARGET_SIM_SWEEP = [0.95]

N_BATCHES_DEFAULT = 1
BATCH_SIZE_DEFAULT = 1
LOG_FILE = Path(
    "src/latent_embedding_experiments/logs/hyperparam_search_clean_soft_thinking.txt"
)


# ---------------------------------------------------------------------------
# Embedding loader
# ---------------------------------------------------------------------------


def load_embeddings(model_id: str, device: torch.device) -> dict[str, torch.Tensor]:
    """Load only the embedding matrix from a sharded safetensors model."""
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
    return {"raw": raw, "norm": F.normalize(raw, dim=1)}


# ---------------------------------------------------------------------------
# Clean soft thinking — batched
# ---------------------------------------------------------------------------


def clean_soft_thinking_batch(
    logits: torch.Tensor,  # [B, L, V]
    vocab_embs: torch.Tensor,  # [V, d]
    vocab_embs_norm: torch.Tensor,  # [V, d]
    attention_mask: torch.Tensor,  # [B, L]
    n_interlopers: int,
    target_sim: float,
) -> torch.Tensor:
    """Compute clean soft thinking vectors for a full batch.

    For each valid (b, l) position:
      - Select min-p target tokens from logits[b, l]
      - For each target token, clean it via subspace_proj_slerp against its
        own top-n interloper neighborhood
      - Aggregate cleaned (normalized) embeddings with softmax-scaled weights
      - Rescale to weighted target magnitude

    Padded positions (attention_mask == 0) get zero vectors.

    Returns: [B, L, d] float32 tensor.
    """
    B, L, V = logits.shape
    d = vocab_embs.shape[1]
    device = logits.device

    result = torch.zeros(B, L, d, dtype=torch.float32, device=device)

    for b in range(B):
        for l in range(L):
            if not attention_mask[b, l]:
                continue

            pos_logits = logits[b, l]  # [V]

            # Select targets — select_targets expects [V], returns ([k], [k])
            _, target_ids = select_targets(pos_logits)
            if len(target_ids) == 0:
                continue

            target_ids_list = target_ids.tolist()
            full_probs = F.softmax(pos_logits, dim=-1)
            target_probs_scaled = F.softmax(
                pos_logits[target_ids] / CFG.temperature, dim=-1
            )  # [k]

            # Weighted target magnitude for rescaling
            target_embs = vocab_embs[target_ids]  # [k, d]
            target_norms = target_embs.norm(p=2, dim=1)  # [k]
            target_magnitude = (target_probs_scaled * target_norms).sum()

            # Clean each target token and collect
            cleaned = []
            for tid in target_ids_list:
                e = vocab_embs[tid]  # [d]
                int_ids = derive_interlopers(tid, vocab_embs_norm, n=n_interlopers)
                int_embs = vocab_embs[int_ids]  # [n, d]
                e_clean = clean_subspace_proj_slerp(e, int_embs, target_sim=target_sim)
                cleaned.append(e_clean)  # [d], normalized

            cleaned_stack = torch.stack(cleaned, dim=0)  # [k, d]
            aggregate = (target_probs_scaled.unsqueeze(1) * cleaned_stack).sum(
                dim=0
            )  # [d]

            norm = aggregate.norm(p=2).clamp_min(1e-8)
            result[b, l] = target_magnitude * aggregate / norm

    return result  # [B, L, d]


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------


def evaluate_hyperparams(
    n_interlopers: int,
    target_sim: float,
    batches: list[dict],
    vocab_embs: torch.Tensor,
    vocab_embs_norm: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate one (n_interlopers, target_sim) pair over all batches.

    Returns averaged loss components.
    """
    accum = {
        "loss_rank": 0.0,
        "loss_margin": 0.0,
        "loss_interloper": 0.0,
        "loss_target": 0.0,
        "total_loss": 0.0,
    }

    for batch in batches:
        logits = batch["logits"].to(device)  # [B, L, V]
        attention_mask = batch["attention_mask"].to(device)  # [B, L]

        with torch.no_grad():
            latent = clean_soft_thinking_batch(
                logits=logits,
                vocab_embs=vocab_embs,
                vocab_embs_norm=vocab_embs_norm,
                attention_mask=attention_mask,
                n_interlopers=n_interlopers,
                target_sim=target_sim,
            )  # [B, L, d]

            losses = latent_head_loss(
                latent=latent,
                logits=logits,
                vocab_embs=vocab_embs,
                vocab_embs_norm=vocab_embs_norm,
                attention_mask=attention_mask,
            )

        for k in accum:
            accum[k] += losses[k].item()

    n = len(batches)
    return {k: v / n for k, v in accum.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Hyperparam search for clean soft thinking via latent_head_loss."
    )
    p.add_argument(
        "--n-interlopers-list", type=int, nargs="+", default=N_INTERLOPERS_SWEEP
    )
    p.add_argument("--target-sim-list", type=float, nargs="+", default=TARGET_SIM_SWEEP)
    p.add_argument(
        "--n-batches",
        type=int,
        default=N_BATCHES_DEFAULT,
        help="Number of dataset batches to evaluate over.",
    )
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE_DEFAULT)
    return p.parse_args()


def run(
    n_interlopers_list: list[int],
    target_sim_list: list[float],
    n_batches: int,
    batch_size: int,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load 405B embeddings ---
    print(f"Loading embeddings from {EMBEDDINGS_MODEL_ID} ...")
    embs = load_embeddings(EMBEDDINGS_MODEL_ID, device)
    vocab_embs: torch.Tensor = embs["raw"]  # [V, d]
    vocab_embs_norm: torch.Tensor = embs["norm"]  # [V, d]
    print(f"Embedding matrix: {vocab_embs.shape}")

    # --- Load dataset batches (once, shared across all hyperparam combos) ---
    print(f"Loading {n_batches} batches from {DATASET_ID} ...")
    dataset = load_dataset(DATASET_ID, split="train", streaming=True)
    batches: list[dict] = []
    buf: list = []
    for sample in dataset:
        buf.append(sample)
        if len(buf) == batch_size:
            logits = torch.stack(
                [torch.tensor(s["top_values"], dtype=torch.float32) for s in buf]
            )  # [B, L, V]
            attention_mask = torch.stack(
                [torch.tensor(s["attention_mask"], dtype=torch.long) for s in buf]
            )  # [B, L]
            batches.append({"logits": logits, "attention_mask": attention_mask})
            buf = []
            if len(batches) == n_batches:
                break
    print(f"Loaded {len(batches)} batches of size {batch_size}.")

    # --- Sweep ---
    combos = list(itertools.product(n_interlopers_list, target_sim_list))
    results: list[dict] = []

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "w", encoding="utf-8") as fh:

        def emit(text: str) -> None:
            print(text)
            fh.write(text + "\n")

        emit("=" * 100)
        emit("HYPERPARAM SEARCH — CLEAN SOFT THINKING")
        emit(f"Embeddings     : {EMBEDDINGS_MODEL_ID}")
        emit(f"Dataset        : {DATASET_ID}")
        emit(f"n_batches      : {n_batches}  |  batch_size : {batch_size}")
        emit(f"n_interlopers  : {n_interlopers_list}")
        emit(f"target_sim     : {target_sim_list}")
        emit(f"Combos         : {len(combos)}")
        emit("=" * 100)

        for i, (n_int, t_sim) in enumerate(combos):
            print(
                f"[{i+1}/{len(combos)}] n_interlopers={n_int}  target_sim={t_sim} ..."
            )
            avg = evaluate_hyperparams(
                n_interlopers=n_int,
                target_sim=t_sim,
                batches=batches,
                vocab_embs=vocab_embs,
                vocab_embs_norm=vocab_embs_norm,
                device=device,
            )
            results.append(
                {
                    "n_interlopers": n_int,
                    "target_sim": t_sim,
                    **avg,
                }
            )
            emit(
                f"  n_int={n_int:2d}  s*={t_sim:.2f} | "
                f"total={avg['total_loss']:+.4f}  "
                f"rank={avg['loss_rank']:.4f}  "
                f"margin={avg['loss_margin']:.4f}  "
                f"interloper={avg['loss_interloper']:.4f}  "
                f"target={avg['loss_target']:.4f}"
            )

        # --- Ranked summary ---
        results.sort(key=lambda r: r["total_loss"])

        emit("")
        emit("=" * 100)
        emit("RESULTS RANKED BY TOTAL LOSS (lower is better)")
        emit("=" * 100)
        header = (
            f"{'Rank':<5} {'n_int':<7} {'s*':<6} "
            f"{'total':>10} {'rank':>10} {'margin':>10} "
            f"{'interloper':>12} {'target':>10}"
        )
        emit(header)
        emit("-" * len(header))
        for rank, r in enumerate(results, 1):
            emit(
                f"{rank:<5} {r['n_interlopers']:<7} {r['target_sim']:<6.2f} "
                f"{r['total_loss']:>10.4f} {r['loss_rank']:>10.4f} "
                f"{r['loss_margin']:>10.4f} {r['loss_interloper']:>12.4f} "
                f"{r['loss_target']:>10.4f}"
            )

        best = results[0]
        emit("")
        emit(
            f"BEST: n_interlopers={best['n_interlopers']}  "
            f"target_sim={best['target_sim']:.2f}  "
            f"total_loss={best['total_loss']:+.4f}"
        )
        emit(f"\nLog saved to {LOG_FILE}")


if __name__ == "__main__":
    args = parse_args()
    run(
        n_interlopers_list=args.n_interlopers_list,
        target_sim_list=args.target_sim_list,
        n_batches=args.n_batches,
        batch_size=args.batch_size,
    )
