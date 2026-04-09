import torch
import torch.nn.functional as F

from src.latent_embedding_experiments.algorithms.token_sharpening import (
    clean_subspace_proj_slerp,
)


def soft_thinking_sharpened_per_token(
    vocab_embs: torch.Tensor,
    vocab_embs_norm: torch.Tensor,
    target_ids: torch.Tensor,
    target_probs_scaled: torch.Tensor,
    target_magnitude: torch.Tensor,
    n_interlopers: int,
    target_sim: float,
) -> torch.Tensor:
    target_ids_set = set(target_ids.tolist())

    cleaned = []
    for tid in target_ids.tolist():
        e = vocab_embs_norm[tid]  # [d]

        cos_sims = vocab_embs_norm @ vocab_embs_norm[tid]
        for t in target_ids_set:
            cos_sims[t] = -1.0

        int_ids = torch.topk(cos_sims, n_interlopers).indices.tolist()
        int_embs = vocab_embs_norm[int_ids]

        e_clean = clean_subspace_proj_slerp(e, int_embs, target_sim=0.97)
        cleaned.append(e_clean)

    cleaned_stack = torch.stack(cleaned, dim=0)  # [k, d]
    weights = target_probs_scaled.to(cleaned_stack.dtype)
    aggregate = (weights.unsqueeze(1) * cleaned_stack).sum(dim=0)

    norm = aggregate.norm(p=2).clamp_min(1e-8)
    return (target_magnitude * aggregate / norm).unsqueeze(0)


def soft_thinking_sharpened_aggregate(
    v_soft: torch.Tensor,
    vocab_embs: torch.Tensor,
    vocab_embs_norm: torch.Tensor,
    target_ids: torch.Tensor,
    target_probs_scaled: torch.Tensor,
    target_magnitude: torch.Tensor,
    n_interlopers: int,
    target_sim: float,
) -> torch.Tensor:
    target_ids_set = set(target_ids.tolist())

    v_norm = F.normalize(v_soft.unsqueeze(0), dim=1).squeeze(0)

    cos_sims = vocab_embs_norm @ v_norm
    for tid in target_ids_set:
        cos_sims[tid] = -1.0

    int_ids = torch.topk(cos_sims, n_interlopers).indices.tolist()
    int_embs_norm = vocab_embs_norm[int_ids]  # [n, d]

    A = int_embs_norm.T
    Q, _ = torch.linalg.qr(A)

    v_tilde = v_norm - Q @ (Q.T @ v_norm)
    v_tilde = F.normalize(v_tilde.unsqueeze(0), dim=1).squeeze(0)

    target_centroid = F.normalize(
        (target_probs_scaled.unsqueeze(1) * vocab_embs_norm[target_ids]).sum(dim=0),
        dim=0,
    )

    cos_phi = (v_tilde @ target_centroid).clamp(-1, 1)
    phi = torch.arccos(cos_phi)

    if phi < 1e-6:
        v_out = v_tilde
    else:
        alpha = (
            1.0
            - torch.arccos(torch.tensor(target_sim, dtype=phi.dtype)).item()
            / phi.item()
        )
        alpha = max(0.0, min(1.0, alpha))
        v_out = (
            torch.sin((1 - alpha) * phi) * v_tilde
            + torch.sin(alpha * phi) * target_centroid
        )

    return (target_magnitude * v_out).unsqueeze(0)
