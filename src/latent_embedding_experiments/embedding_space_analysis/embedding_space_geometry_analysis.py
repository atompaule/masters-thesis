"""
Geometry diagnostics for static token embeddings (e.g. model.embed_tokens.weight).

Includes:
  1) Average pairwise cosine similarity (common NLP "narrow cone" proxy).
  2) PC / eigenvalue spectrum ratios (variance concentration), plus top tokens aligned
     with each of the first PCs (cosine to PC axis after sample-mean centering).
  3) Partition-based global isotropy I(W) from Rajaee et al. (ACL 2021), Eq. (2),
     following Mu & Viswanath (2018) — min/max of Arora et al. partition scores
     over eigenvectors of W^T W.
  4) IsoScore ι(X) from Rudman et al. (2022), arXiv:2108.07344.
  5) Intrinsic dimension (Levina–MLE) and ID / ambient_dim (ID score as in Rudman et al.).
  6) Norm distribution: 20 equal-width bins over [min_norm, max_norm], with token
     count and 10 random example tokens per bin.
"""

from __future__ import annotations

import json
import math
import random

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoTokenizer

LOG_FILE = "src/latent_embedding_experiments/logs/llama_8b_embedding_space_geometry_analysis.txt"
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"


def log(message, file):
    print(message)
    file.write(message + "\n")


def load_embeddings(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    index_path = hf_hub_download(
        repo_id=model_id, filename="model.safetensors.index.json"
    )

    with open(index_path, "r") as f:
        index_data = json.load(f)

    tensor_name = "model.embed_tokens.weight"
    tensor_file = index_data["weight_map"][tensor_name]

    tensor_path = hf_hub_download(repo_id=model_id, filename=tensor_file)
    tensors = load_file(tensor_path)

    embeddings = tensors[tensor_name].to(torch.float32)

    return tokenizer, embeddings


def _sample_rows(embeddings: torch.Tensor, sample_size: int, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    n = embeddings.shape[0]
    k = min(sample_size, n)
    idx = torch.randperm(n, device=embeddings.device)[:k]
    return embeddings[idx]


def analyze_norms(embeddings, tokenizer, log_file):
    norms = torch.norm(embeddings, dim=1)

    log(f"Mean norm: {norms.mean().item():.4f}", log_file)
    log(f"Median norm: {norms.median().item():.4f}", log_file)
    log(f"Std norm:  {norms.std().item():.4f}", log_file)

    top_vals, top_idxs = torch.topk(norms, 15)
    log("\nTop tokens by norm:", log_file)
    for val, idx in zip(top_vals, top_idxs):
        token = tokenizer.decode([idx])
        log(f"{val.item():.4f} | '{token}'", log_file)

    low_vals, low_idxs = torch.topk(norms, 15, largest=False)
    log("\nLowest tokens by norm:", log_file)
    for val, idx in zip(low_vals, low_idxs):
        token = tokenizer.decode([idx])
        log(f"{val.item():.4f} | '{token}'", log_file)


def analyze_norm_distribution(
    embeddings: torch.Tensor,
    tokenizer,
    log_file,
    n_bins: int = 20,
    examples_per_bin: int = 10,
    seed: int = 42,
) -> None:
    """
    Divide the norm range [min, max] into n_bins equal-width bins and report:
      - bin edges
      - token count and fraction of vocab in each bin
      - examples_per_bin random tokens sampled uniformly from the bin
    """
    log(f"\n--- 6. Norm distribution ({n_bins} bins, {examples_per_bin} examples each) ---", log_file)

    rng = random.Random(seed)
    norms = torch.norm(embeddings, dim=1)  # [V]
    v_min = norms.min().item()
    v_max = norms.max().item()
    vocab_size = norms.shape[0]

    edges = torch.linspace(v_min, v_max, n_bins + 1)

    log(f"Norm range: [{v_min:.4f}, {v_max:.4f}]  |  vocab size: {vocab_size:,}", log_file)
    log("", log_file)

    col_norm  = 18   # bin range column
    col_count = 10   # count column
    col_frac  = 8    # fraction column
    header = (
        f"  {'bin range':^{col_norm}}  "
        f"{'count':>{col_count}}  "
        f"{'frac':>{col_frac}}  "
        f"examples"
    )
    log(header, log_file)
    log("  " + "─" * (len(header) + 20), log_file)

    for b in range(n_bins):
        lo = edges[b].item()
        hi = edges[b + 1].item()

        # Include right edge in the last bin so max-norm token isn't missed
        if b < n_bins - 1:
            mask = (norms >= lo) & (norms < hi)
        else:
            mask = (norms >= lo) & (norms <= hi)

        indices = mask.nonzero(as_tuple=True)[0].tolist()
        count = len(indices)
        frac = count / vocab_size

        if count == 0:
            sample_str = "(empty)"
        else:
            sample_ids = rng.sample(indices, min(examples_per_bin, count))
            tokens = [repr(tokenizer.decode([tid])) for tid in sample_ids]
            sample_str = "  ".join(tokens)

        bin_range = f"[{lo:7.3f}, {hi:7.3f})"
        log(
            f"  {bin_range:{col_norm}}  "
            f"{count:{col_count},}  "
            f"{frac:{col_frac}.4f}  "
            f"{sample_str}",
            log_file,
        )


def analyze_average_pairwise_cosine_similarity(
    embeddings: torch.Tensor, log_file, sample_size: int = 10_000, seed: int = 42
):
    """
    Mean off-diagonal cosine similarity over random token pairs (after row L2 norm).
    Higher → vectors more aligned (often called higher anisotropy in NLP).
    Rudman et al. also discuss 1 - |mean cos| as a (flawed) isotropy score.
    """
    log("\n--- 1. Average pairwise cosine similarity ---", log_file)
    normalized = F.normalize(embeddings, dim=1)
    sample = _sample_rows(normalized, sample_size, seed)
    m = sample.shape[0]
    sim_matrix = sample @ sample.T
    sim_matrix.fill_diagonal_(0.0)
    avg_sim = sim_matrix.sum().item() / (m * (m - 1))
    log(
        f"Sample size: {m}. Mean off-diagonal cosine similarity: {avg_sim:.6f}",
        log_file,
    )
    log(
        f"1 - |mean cosine| (Rudman-style comparison to isotropy scale [0,1]): "
        f"{1.0 - abs(avg_sim):.6f}",
        log_file,
    )
    if avg_sim > 0.1:
        log("Interpretation (heuristic): high alignment / narrow-cone proxy.", log_file)
    elif avg_sim > 0.01:
        log("Interpretation (heuristic): moderate alignment.", log_file)
    else:
        log(
            "Interpretation (heuristic): mean cosine near 0 (not a strong global cone).",
            log_file,
        )


def _centered_matrix(X: torch.Tensor) -> torch.Tensor:
    return X - X.mean(dim=0, keepdim=True)


def _log_pc_token_alignment(
    embeddings: torch.Tensor,
    tokenizer,
    log_file,
    sample_mean: torch.Tensor,
    Vh: torch.Tensor,
    lam: torch.Tensor,
    total_var: torch.Tensor,
    num_pcs: int,
    token_top_k: int,
    row_batch: int,
) -> None:
    """
    For each of the first num_pcs principal axes (rows of Vh from SVD(centered sample)),
    list vocabulary tokens with highest cosine similarity to that axis after centering
    every row by the same sample mean. Interprets: which token directions co-vary most
    with that PC in the sample-defined subspace.
    """
    rank = Vh.shape[0]
    k_pcs = min(num_pcs, rank)
    if k_pcs < 1 or token_top_k < 1:
        return

    log("\n--- 2b. Vocabulary tokens most cosine-aligned with sample PCs ---", log_file)
    log(
        "PC axis = row of Vh from torch.linalg.svd(centered sample). "
        "Score = cos(L2_norm(embed - sample_mean), PC_axis). "
        "Same sample_mean as in §2; full vocab scanned in row batches.",
        log_file,
    )

    device = embeddings.device
    mu = sample_mean.to(device)
    P = Vh[:k_pcs].to(device)

    vocab_size = embeddings.shape[0]
    scores_chunks: list[torch.Tensor] = []
    for start in range(0, vocab_size, row_batch):
        end = min(start + row_batch, vocab_size)
        block = embeddings[start:end]
        centered_n = F.normalize(block - mu, dim=1)
        scores_chunks.append(centered_n @ P.T)

    scores = torch.cat(scores_chunks, dim=0)

    tv = total_var.clamp_min(1e-30)
    for j in range(k_pcs):
        frac_j = (lam[j] / tv).item() if j < lam.numel() else float("nan")
        log(
            f"\nPC {j + 1} (fraction of sample variance along this axis: {frac_j:.6f})",
            log_file,
        )
        vals, idx = torch.topk(scores[:, j], min(token_top_k, vocab_size))
        for r, (val, tid) in enumerate(zip(vals, idx), start=1):
            tok = tokenizer.decode([tid.item()])
            log(f"  {r:2}. cos={val.item():.6f} | id={tid.item()} | '{tok}'", log_file)


def analyze_pc_spectrum(
    embeddings: torch.Tensor,
    tokenizer,
    log_file,
    sample_size: int = 10_000,
    seed: int = 42,
    top_k_report: tuple[int, ...] = (1, 5, 10, 50),
    pc_token_top_k: int = 15,
    num_pcs_for_token_report: int = 10,
    pc_alignment_row_batch: int = 8192,
):
    """
    Eigenvalue-based variance concentration (PCA spectrum of centered embeddings).
    λ_max / sum λ: high → dominant first PC (anisotropic variance).

    Optionally lists, for each of the first num_pcs_for_token_report axes, the
    vocabulary tokens whose (sample-mean-centered) embeddings have highest cosine
    similarity to that PC — i.e. which tokens load most on that axis relative to the
    random sample's mean.
    """
    log("\n--- 2. PC spectrum / eigenvalue ratios ---", log_file)
    X = _sample_rows(embeddings, sample_size, seed)
    n, d = X.shape
    if n < 2:
        log("Not enough points for spectrum.", log_file)
        return

    Xc = _centered_matrix(X)
    sample_mean = X.mean(dim=0)
    _, s, Vh = torch.linalg.svd(Xc, full_matrices=False)
    rank = min(n - 1, d, s.numel())
    s = s[:rank]
    lam = (s**2) / (n - 1)

    if lam.numel() == 0:
        log("Not enough points for spectrum.", log_file)
        return

    total = lam.sum().clamp_min(1e-30)
    lam_max = lam[0]
    log(f"Sample size: {n}, ambient dim: {embeddings.shape[1]}", log_file)
    log(f"Number of positive PCs (from SVD rank): {lam.numel()}", log_file)
    log(
        f"λ_max / sum(λ) (anisotropy concentration): {(lam_max / total).item():.6f}",
        log_file,
    )
    log(
        f"Participation ratio (sum λ)² / sum λ²: "
        f"{((total**2) / (lam**2).sum()).item():.4f}",
        log_file,
    )
    p = lam / total
    h = -(p * torch.log(p.clamp_min(1e-30))).sum()
    log(
        f"Effective rank exp(entropy of normalized spectrum): {torch.exp(h).item():.4f}",
        log_file,
    )
    for k in top_k_report:
        if k <= lam.numel():
            frac = lam[:k].sum() / total
            log(f"Variance in top-{k} PCs: {frac.item():.6f}", log_file)

    if pc_token_top_k > 0 and num_pcs_for_token_report > 0:
        _log_pc_token_alignment(
            embeddings,
            tokenizer,
            log_file,
            sample_mean,
            Vh,
            lam,
            total,
            num_pcs_for_token_report,
            pc_token_top_k,
            pc_alignment_row_batch,
        )


def _partition_F_columnwise(
    W: torch.Tensor, unit_cols: torch.Tensor, row_chunk: int = 8192
) -> torch.Tensor:
    """
    F(u) = sum_i exp(u^T w_i) for each column u of unit_cols (d x r), W is (N x d).
    Returns tensor of shape (r,) with F values.
    """
    device = W.device
    dtype = W.dtype
    r = unit_cols.shape[1]
    acc = torch.zeros(r, device=device, dtype=dtype)
    n = W.shape[0]
    for start in range(0, n, row_chunk):
        end = min(start + row_chunk, n)
        batch = W[start:end] @ unit_cols  # (chunk, r)
        m = batch.max(dim=0).values
        acc += torch.exp(batch - m.unsqueeze(0)).sum(dim=0) * torch.exp(m)
    return acc


def analyze_rajaee_partition_isotropy(
    embeddings: torch.Tensor,
    log_file,
    svd_sample_size: int = 8_192,
    eval_sample_size: int | None = None,
    max_directions: int = 256,
    seed: int = 42,
    row_chunk: int = 8192,
):
    """
    Rajaee et al. (ACL 2021), Eq. (2): I(W) = min_{u in U} F(u) / max_{u in U} F(u),
    with F the Arora partition score and U eigenvectors of W^T W (Mu & Viswanath 2018).

    Full-vocab W^T W is infeasible for huge D; we estimate principal directions from a
    row subsample of W and evaluate F on the same (or larger) row subsample for scale.
    """
    log("\n--- 3. Global partition isotropy I(W) (Rajaee et al. 2021) ---", log_file)
    N, d = embeddings.shape
    eval_n = eval_sample_size if eval_sample_size is not None else min(N, 65_536)
    W_svd = _sample_rows(embeddings, svd_sample_size, seed)
    W_eval = _sample_rows(embeddings, eval_n, seed + 1)

    _, _, Vh = torch.linalg.svd(W_svd, full_matrices=False)
    V = Vh.T[:, : min(max_directions, Vh.shape[0])]
    unit_cols = F.normalize(V, dim=0)

    F_vals = _partition_F_columnwise(W_eval, unit_cols, row_chunk=row_chunk)
    pos = F_vals[F_vals > 0]
    if pos.numel() == 0:
        log("Partition scores degenerate; skip I(W).", log_file)
        return
    i_w = (pos.min() / pos.max()).item()
    log(
        f"SVD rows: {W_svd.shape[0]}, eval rows: {W_eval.shape[0]}, "
        f"|U| (eigvecs used): {unit_cols.shape[1]}",
        log_file,
    )
    log(
        f"I(W) = min F / max F over U: {i_w:.6f} (1 = isotropic, 0 = extreme)", log_file
    )
    log(f"log I(W): {math.log(i_w + 1e-30):.6f}", log_file)


def isoscore_rudman(X: torch.Tensor) -> float:
    """
    IsoScore ι(X) from Rudman et al. (2022), Eq. (4.1). X: (m, d) point cloud, not necessarily centered.
    """
    m, d = X.shape
    if m < 2 or d < 2:
        return float("nan")
    Xc = _centered_matrix(X)
    _, s, Vh = torch.linalg.svd(Xc, full_matrices=False)
    rank = min(m - 1, d, s.numel())
    lam = torch.zeros(d, device=X.device, dtype=X.dtype)
    lam[:rank] = (s[:rank] ** 2) / (m - 1)
    if lam.sum() <= 0:
        return 0.0
    sigma_hat = math.sqrt(d) * lam / lam.norm()
    one = torch.ones_like(sigma_hat)
    numer = (sigma_hat - one).norm().item()
    denom = math.sqrt(2 * (d - math.sqrt(d)))
    delta = numer / denom
    inner = d - (delta**2) * (d - math.sqrt(d))
    iota = ((inner**2) - d) / (d * (d - 1))
    return max(0.0, min(1.0, float(iota)))


def analyze_isoscore(
    embeddings: torch.Tensor, log_file, sample_size: int = 10_000, seed: int = 42
):
    log("\n--- 4. IsoScore (Rudman et al. 2022) ---", log_file)
    X = _sample_rows(embeddings, sample_size, seed)
    score = isoscore_rudman(X)
    log(f"Sample size: {X.shape[0]}, ambient dim d={embeddings.shape[1]}", log_file)
    log(f"IsoScore ι(X) in [0, 1]: {score:.6f} (1 = isotropic utilization)", log_file)


def intrinsic_dimension_levina_mle(
    X: torch.Tensor, k: int = 20, seed: int = 0
) -> tuple[float, float]:
    """
    Levina & Bickel (2004) MLE intrinsic dimension from k-NN distances.
    Returns (mean local estimate, fraction of finite estimates used).
    """
    n, d = X.shape
    if n < k + 1:
        return float("nan"), 0.0
    torch.manual_seed(seed)
    # Subsample if n very large to cap O(n^2) memory
    max_n = 8000
    if n > max_n:
        idx = torch.randperm(n, device=X.device)[:max_n]
        X = X[idx]
        n = max_n
    dist = torch.cdist(X, X)
    dist.fill_diagonal_(float("inf"))
    knn_d, _ = dist.topk(k, largest=False)
    knn_d = knn_d.clamp_min(1e-12)
    rk = knn_d[:, k - 1]
    log_rk = torch.log(rk)
    log_rj = torch.log(knn_d[:, : k - 1])
    denom = (k - 1) * log_rk.unsqueeze(1) - log_rj
    denom = denom.sum(dim=1)
    local_m = (k - 1) / denom.clamp_min(1e-12)
    finite = torch.isfinite(local_m) & (local_m > 0) & (local_m < 1e4)
    if not finite.any():
        return float("nan"), 0.0
    m_hat = local_m[finite].mean().item()
    frac = finite.float().mean().item()
    return m_hat, frac


def analyze_intrinsic_dimension(
    embeddings: torch.Tensor,
    log_file,
    sample_size: int = 10_000,
    knn_k: int = 20,
    seed: int = 42,
):
    """
    Intrinsic dimensionality (Levina–MLE) and ID / D (ID score, Rudman et al. naming).
    Low ID relative to D suggests variance lies on a low-dimensional manifold.
    """
    log("\n--- 5. Intrinsic dimension (Levina–MLE) ---", log_file)
    X = _sample_rows(embeddings, sample_size, seed)
    m_hat, frac = intrinsic_dimension_levina_mle(X, k=knn_k, seed=seed + 7)
    d = embeddings.shape[1]
    log(
        f"Sample size: {X.shape[0]}, k={knn_k}, valid local estimates: {frac:.2%}",
        log_file,
    )
    if math.isnan(m_hat):
        log("ID estimate failed.", log_file)
        return
    log(f"Intrinsic dimension estimate (mean local MLE): {m_hat:.4f}", log_file)
    log(f"ID / ambient_dim (normalized ID score): {m_hat / d:.8f}", log_file)


def analyze_topology(normalized, tokenizer, log_file, batch_size=2048):
    vocab_size = normalized.shape[0]

    max_neighbor_sim = torch.zeros(vocab_size, device=normalized.device)
    density_counts = torch.zeros(vocab_size, device=normalized.device)

    for i in range(0, vocab_size, batch_size):
        end = min(i + batch_size, vocab_size)
        batch = normalized[i:end]

        sims = batch @ normalized.T

        for j in range(end - i):
            sims[j, i + j] = -1.0

        max_sims, _ = torch.max(sims, dim=1)
        max_neighbor_sim[i:end] = max_sims

        density = (sims > 0.85).sum(dim=1)
        density_counts[i:end] = density.float()

    low_vals, low_idxs = torch.topk(max_neighbor_sim, 15, largest=False)
    log("\nMost isolated tokens:", log_file)
    for val, idx in zip(low_vals, low_idxs):
        token = tokenizer.decode([idx.item()])
        log(f"{val.item():.4f} | '{token}'", log_file)

    high_vals, high_idxs = torch.topk(density_counts, 15)
    log("\nMost dense tokens:", log_file)
    for val, idx in zip(high_vals, high_idxs):
        token = tokenizer.decode([idx.item()])
        log(f"{int(val.item())} neighbors | '{token}'", log_file)


def main():
    tokenizer, embeddings = load_embeddings(MODEL_ID)

    vocab_size, dim = embeddings.shape

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        log(f"Vocab size: {vocab_size}", f)
        log(f"Embedding dimension: {dim}", f)

        analyze_norms(embeddings, tokenizer, f)
        analyze_norm_distribution(embeddings, tokenizer, f)
        analyze_average_pairwise_cosine_similarity(embeddings, f)
        analyze_pc_spectrum(embeddings, tokenizer, f)
        analyze_rajaee_partition_isotropy(embeddings, f)
        analyze_isoscore(embeddings, f)
        analyze_intrinsic_dimension(embeddings, f)

    print(f"Done. Results saved to {LOG_FILE}")


if __name__ == "__main__":
    main()