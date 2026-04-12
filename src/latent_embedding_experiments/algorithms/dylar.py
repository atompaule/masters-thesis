import torch
import torch.nn.functional as F

from src.latent_embedding_experiments.algorithms.config import CFG
from src.latent_embedding_experiments.algorithms.utils import select_targets


def _soft_project(
    vec: torch.Tensor,       # [d] — any vector in embedding space
    vocab_embs: torch.Tensor,  # [V, d]
) -> torch.Tensor:
    """
    Soft-project a vector onto the embedding space via full-vocabulary softmax.

    Unlike select_targets-based soft thinking, DyLaR's residual projections
    operate on arbitrary residual vectors (not logits from a model forward pass),
    so we cannot use select_targets here — residuals have no natural top-p nucleus.
    We use full-vocab softmax with the same temperature as soft thinking.

    Args:
        vec:        [d] vector to project (either h_t or a residual r_k)
        vocab_embs: [V, d] token embedding matrix (unnormalized)

    Returns: [d] soft projection — probability-weighted sum of token embeddings.
    """
    # Treat the vector as logits via dot product with embedding matrix (tied-weights convention)
    logits = vocab_embs @ vec  # [V]
    probs = F.softmax(logits / CFG.temperature, dim=-1)  # [V]
    return probs @ vocab_embs  # [d]


def dylar(
    logits: torch.Tensor,    # [V] — raw logits from LLM
    hidden: torch.Tensor,    # [d] — last hidden state h_t
    vocab_embs: torch.Tensor,  # [V, d] — token embedding matrix (unnormalized)
    K: int = 2,
    lam: float = 2.0,
    entropy_threshold: float | None = None,
) -> torch.Tensor:
    """
    DyLaR: Dynamic Latent Reasoning via Semantic Residual Refinement.

    Implements the SRR module from Lv et al. (AAAI 2026):
      1. Initial soft projection c_0 from model logits (top-p nucleus, matching
         soft_thinking style for the first step).
      2. Residual refinement: r_k = r_{k-1} - c_{k-1}, then c_k = soft_project(r_k).
      3. Semantic fusion: c_hat = c_0 + sum_k beta_k * c_k / (1 + exp(-lam * ||c_k||))
         with beta_k = 3 * exp(-k) and the sigmoid-like norm gate.

    NOTE on tied vs untied embeddings: the residual subtraction r_1 = h_t - c_0
    is geometrically meaningful only when the LM head weight W equals the input
    embedding matrix E (tied embeddings). For untied models (e.g. LLaMA 3.1),
    h_t and c_0 live in nominally different spaces; results are empirically useful
    but not principled. See paper Table 1 footnote and our thesis Section X.

    If entropy_threshold is provided, the dynamic switch policy is applied:
    when entropy of the initial distribution is below the threshold, the function
    returns None to signal that explicit (discrete) decoding should be used instead.

    Args:
        logits:            [V] raw logits from the LLM (for the initial projection)
        hidden:            [d] last hidden state h_t (for residual computation)
        vocab_embs:        [V, d] token embedding matrix
        K:                 number of residual refinement iterations (default 2)
        lam:               sigmoid gate sharpness (default 2.0, from paper)
        entropy_threshold: if not None, enables dynamic switching policy

    Returns:
        [1, d] SRR embedding, or None if dynamic switch selects explicit decoding.
    """
    # ── Step 1: initial soft projection (nucleus, matching soft_thinking) ──────
    target_logits, target_ids = select_targets(logits)
    probs_0 = F.softmax(target_logits / CFG.temperature, dim=-1)  # [k]
    target_embs = vocab_embs[target_ids]                           # [k, d]
    c0 = (probs_0.unsqueeze(1) * target_embs).sum(dim=0)          # [d]

    # ── Dynamic switch policy (optional) ──────────────────────────────────────
    if entropy_threshold is not None:
        # Entropy of the full initial distribution (not just nucleus)
        full_probs = F.softmax(logits / CFG.temperature, dim=-1)
        entropy = -(full_probs * full_probs.clamp(min=1e-12).log()).sum()
        if entropy.item() < entropy_threshold:
            return None  # caller should fall back to discrete decoding

    # ── Step 2: residual refinement ───────────────────────────────────────────
    # r_1 = h_t - c_0  (NOTE: meaningful only for tied embeddings)
    residual = hidden - c0  # [d]

    cs = []  # will hold c_1 ... c_K
    for _ in range(K):
        ck = _soft_project(residual, vocab_embs)  # [d]
        cs.append(ck)
        residual = residual - ck

    # ── Step 3: semantic fusion ───────────────────────────────────────────────
    # c_hat = c_0 + sum_{k=1}^{K} beta_k * c_k / (1 + exp(-lam * ||c_k||))
    result = c0.clone()
    for k, ck in enumerate(cs, start=1):
        beta_k = 3.0 * torch.exp(torch.tensor(-float(k), device=ck.device))
        norm_ck = ck.norm()
        gate = 1.0 / (1.0 + torch.exp(-lam * norm_ck))  # scalar
        result = result + beta_k * gate * ck

    return result.unsqueeze(0)  # [1, d]


def dylar_normalized(
    logits: torch.Tensor,    # [V]
    hidden: torch.Tensor,    # [d]
    vocab_embs_norm: torch.Tensor,  # [V, d] — normalized embeddings
    target_magnitude: torch.Tensor,
    K: int = 2,
    lam: float = 2.0,
    entropy_threshold: float | None = None,
) -> torch.Tensor:
    """
    DyLaR in direction space, followed by magnitude injection.

    Mirrors soft_thinking_normalized: runs the full SRR pipeline on normalized
    embeddings, then renormalizes and injects target_magnitude.

    Returns:
        [1, d] tensor, or None if dynamic switch selects explicit decoding.
    """
    # Use unnormalized vocab_embs for the dot-product logits inside _soft_project,
    # but since we only have norm embeddings here, the projection is approximate.
    # In practice, callers should pass both; this variant keeps the same signature
    # as soft_thinking_normalized for drop-in compatibility.
    result = dylar(
        logits=logits,
        hidden=hidden,
        vocab_embs=vocab_embs_norm,
        K=K,
        lam=lam,
        entropy_threshold=entropy_threshold,
    )

    if result is None:
        return None

    direction = F.normalize(result, dim=1)  # [1, d]
    return target_magnitude * direction     # [1, d]