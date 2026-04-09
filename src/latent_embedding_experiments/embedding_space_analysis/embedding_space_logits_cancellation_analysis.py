"""
Load arcee-ai/LLama-405B-Logits, take top-k tokens per timestep, build their
embedding cosine matrix, and summarize how much those directions cancel under
soft weights (softmax over top-k logits).

"Auslöschung" (two related scalars):
  - weighted_pairwise: E_{i≠j}[1 - cos(e_i,e_j)] with i,j drawn independently
    from the softmax over top-k (pair weight w_i w_j / (1 - Σ w_k^2)).
    This equals (1 - ||Σ_i w_i ê_i||²) / (1 - Σ_i w_i²) for L2-normalized ê_i.
    Range up to ~2 when opposing directions both get mass.
  - soft_sum: 1 - ||Σ_i w_i ê_i|| (literal soft mixture; always in [0, 1] for unit
    embeddings).

Big picture: 10 equal-width bins on [0, 2] for weighted_pairwise (primary).
Values > 2 are clamped into the last bin (rare).
"""

from __future__ import annotations

import gc
import json
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoTokenizer

from src.latent_embedding_experiments.algorithms.utils import emit


@dataclass
class Config:
    dataset_id: str = "arcee-ai/LLama-405B-Logits"
    model_id: str = "meta-llama/Llama-3.1-405B-Instruct"
    top_k: int = 10
    num_bins: int = 10
    pairwise_hist_hi: float = 2.0
    max_rows: int | None = 200
    streaming: bool = True
    log_file: str = (
        "src/latent_embedding_experiments/logs/llama_405b_logits_topk_ausloeschung.txt"
    )
    example_matrices: int = 2


CFG = Config()


def load_embeddings(model_id: str, device: torch.device) -> dict[str, torch.Tensor]:
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


def iter_timesteps(
    row: dict,
    tokenizer: AutoTokenizer,
    device: torch.device,
    max_k: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    if "token_ids" not in row or "top_values" not in row:
        return []

    ids = torch.as_tensor(row["token_ids"], device=device, dtype=torch.long)
    logits = torch.as_tensor(row["top_values"], device=device, dtype=torch.float32)

    if ids.numel() == 0:
        return []

    special = torch.tensor(tokenizer.all_special_ids, device=device, dtype=torch.long)
    out: list[tuple[torch.Tensor, torch.Tensor]] = []

    def one_step(tid: torch.Tensor, log: torch.Tensor) -> None:
        if tid.dim() != 1:
            tid = tid.flatten()
            log = log.flatten()
        mask = ~torch.isin(tid, special)
        tid = tid[mask][:max_k]
        log = log[mask][:max_k]
        if tid.numel() < 2:
            return
        probs = F.softmax(log, dim=-1)
        out.append((probs, tid))

    if ids.dim() == 1:
        one_step(ids, logits)
    elif ids.dim() == 2:
        for t in range(ids.shape[0]):
            one_step(ids[t], logits[t])
    else:
        ids_flat = ids.reshape(-1, ids.shape[-1])
        logits_flat = logits.reshape(-1, logits.shape[-1])
        for t in range(ids_flat.shape[0]):
            one_step(ids_flat[t], logits_flat[t])

    return out


def metrics_for_timestep(
    probs: torch.Tensor,
    ids: torch.Tensor,
    emb_norm: torch.Tensor,
) -> tuple[torch.Tensor, float, float]:
    k = ids.shape[0]
    e = emb_norm[ids]
    cos = e @ e.T

    w = probs.view(-1, 1)
    v = (e * w).sum(dim=0)
    soft_sum_ausl = 1.0 - v.norm().item()

    sum_w_sq = (probs**2).sum()
    denom = 1.0 - sum_w_sq
    if denom.item() <= 1e-12:
        w_pair = float("nan")
    else:
        vv = (v.norm() ** 2).item()
        w_pair = (1.0 - vv) / denom.item()

    return cos, w_pair, soft_sum_ausl


def bin_pairwise(x: float, num_bins: int, hi: float) -> int:
    width = hi / num_bins
    xc = min(max(x, 0.0), hi - 1e-12)
    b = int(xc / width)
    return min(b, num_bins - 1)


def main():
    gc.collect()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(CFG.model_id)
    emb = load_embeddings(CFG.model_id, device)

    ds = load_dataset(
        CFG.dataset_id,
        split="train",
        streaming=CFG.streaming,
    )

    primary_values: list[float] = []
    soft_sum_values: list[float] = []

    counts = torch.zeros(CFG.num_bins, dtype=torch.long)
    hi = CFG.pairwise_hist_hi
    width = hi / CFG.num_bins

    row_idx = 0
    examples: list[
        tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, float, float]
    ] = []

    for row in ds:
        if CFG.max_rows is not None and row_idx >= CFG.max_rows:
            break

        steps = iter_timesteps(row, tokenizer, device, CFG.top_k)
        for probs, ids in steps:
            cos, w_pair, soft_ausl = metrics_for_timestep(probs, ids, emb["norm"])
            if not math.isfinite(w_pair):
                continue
            primary_values.append(w_pair)
            soft_sum_values.append(soft_ausl)

            counts[bin_pairwise(w_pair, CFG.num_bins, hi)] += 1

            if len(examples) < CFG.example_matrices:
                examples.append(
                    (
                        row_idx,
                        cos.detach().cpu(),
                        probs.cpu(),
                        ids.cpu(),
                        w_pair,
                        soft_ausl,
                    )
                )

        row_idx += 1
        if row_idx % 50 == 0:
            print(f"rows={row_idx} timesteps={len(primary_values)} ...")

    total = counts.sum().item()
    n_primary = len(primary_values)

    with open(CFG.log_file, "w", encoding="utf-8") as f:

        def e(s: str) -> None:
            emit(s, f)

        e("=== Logits top-k Auslöschung (405B logits dataset) ===\n")
        e(f"Dataset: {CFG.dataset_id}")
        e(f"Embeddings: {CFG.model_id} (L2-normalized rows for cosines)")
        e(
            f"top_k: {CFG.top_k}, rows scanned: {row_idx}, valid timesteps: {n_primary}\n"
        )

        e("Definitions:")
        e("  cos_ij = cosine(embed(token_i), embed(token_j)) on top-k candidates.")
        e("  w = softmax(logits | top-k), Σ w = 1.")
        e(
            "  weighted_pairwise_ausloeschung = (1 - ||Σ_i w_i ê_i||²) / (1 - Σ_i w_i²) "
            "= E_{i≠j}[1 - cos_ij],  i,j i.i.d. ~ w.  (Can exceed 1.)"
        )
        e(
            "  soft_sum_ausloeschung = 1 - ||Σ_i w_i ê_i||  "
            "(cancellation of the soft mixture; in [0,1] for unit ê_i)."
        )
        e("")

        if n_primary == 0:
            e("No valid timesteps (need ≥2 non-special tokens per step).")
            print(f"Done → {CFG.log_file}")
            return

        t = torch.tensor(primary_values)
        e(
            f"weighted_pairwise: mean={t.mean().item():.6f} std={t.std(unbiased=False).item():.6f} "
            f"min={t.min().item():.6f} max={t.max().item():.6f}"
        )
        ts = torch.tensor(soft_sum_values)
        e(
            f"soft_sum:          mean={ts.mean().item():.6f} std={ts.std(unbiased=False).item():.6f} "
            f"min={ts.min().item():.6f} max={ts.max().item():.6f}"
        )

        e(
            f"\n--- Big picture: {CFG.num_bins} groups (equal-width bins on "
            f"[0, {hi}] for weighted_pairwise) ---"
        )
        e("bin_range          count    relative_freq")
        for b in range(CFG.num_bins):
            lo = b * width
            hi_b = (b + 1) * width
            if b < CFG.num_bins - 1:
                label = f"[{lo:.2f}, {hi_b:.2f})"
            else:
                label = f"[{lo:.2f}, {hi_b:.2f}]"
            c = counts[b].item()
            freq = c / total if total else 0.0
            e(f"{label:<18}  {c:6d}    {freq:.4%}")

        e(f"\ntotal timesteps in histogram: {total}")

        if examples:
            e("\n--- Example cosine matrices (first valid timesteps) ---")
            for ex_i, (r_i, cos, probs, ids, w_pair, soft_ausl) in enumerate(examples):
                toks = tokenizer.convert_ids_to_tokens(ids.tolist())
                toks = [t.replace("Ġ", " ") for t in toks]
                e(f"\nexample #{ex_i + 1} (dataset row index {r_i})")
                e(f"weighted_pairwise={w_pair:.6f}  soft_sum={soft_ausl:.6f}")
                e("tokens: " + ", ".join(toks))
                e("probs:  " + ", ".join(f"{p:.4f}" for p in probs.tolist()))
                e(str(cos.numpy()))

    print(f"\nDone → {CFG.log_file}")


if __name__ == "__main__":
    main()
