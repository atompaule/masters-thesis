import torch

from src.latent_embedding_experiments.algorithms.token_sharpening import (
    clean_subspace_proj_slerp,
)


def discrete_sharpened(
    top1_id: int,
    vocab_embs: torch.Tensor,
    vocab_embs_norm: torch.Tensor,
    target_magnitude: torch.Tensor,
    n_interlopers: int,
    target_sim: float,
) -> torch.Tensor:
    """Clean the greedy top-1 token embedding via subspace projection + slerp.

    Derives the top-n nearest neighbors of the token from the embedding matrix
    (excluding the token itself), projects out their subspace, then slerps back
    to target_sim cosine similarity with the original token.

    Returns a [1, d] tensor rescaled to target_magnitude.
    """
    e = vocab_embs_norm[top1_id]  # [d], unit norm

    cos_sims = vocab_embs_norm @ e
    cos_sims[top1_id] = -1.0
    int_ids = torch.topk(cos_sims, n_interlopers).indices.tolist()
    int_embs = vocab_embs_norm[int_ids]  # [n, d]

    e_clean = clean_subspace_proj_slerp(e, int_embs, target_sim=target_sim)  # [d]

    norm = e_clean.norm(p=2).clamp_min(1e-8)
    return (target_magnitude * e_clean / norm).unsqueeze(0)  # [1, d]


def discrete_sharpened_dot_rescaled(
    top1_id: int,
    vocab_embs: torch.Tensor,
    vocab_embs_norm: torch.Tensor,
    target_magnitude: torch.Tensor,
    n_interlopers: int,
    target_sim: float,
) -> torch.Tensor:
    """Clean via subspace proj + slerp, then rescale so dot product to the
    original token equals the original token's self-dot-product (||e||^2).

    After cleaning we have a unit-norm vector e_clean with cosine similarity
    target_sim to e. We want:

        dot(s * e_clean, e) = dot(e, e) = ||e||^2

    Solving for s:

        s = ||e||^2 / dot(e_clean, e)
          = ||e||^2 / (target_sim * ||e_clean|| * ||e||)
          = ||e|| / target_sim          (since ||e_clean|| = 1)

    This ensures the model sees the same dot-product signal toward the target
    token as it would from the original embedding, while still benefiting from
    the reduced interloper similarity in direction.

    Returns a [1, d] tensor.
    """
    e = vocab_embs[top1_id]  # [d], unnormalized
    e_norm = vocab_embs_norm[top1_id]  # [d], unit norm

    cos_sims = vocab_embs_norm @ e_norm
    cos_sims[top1_id] = -1.0
    int_ids = torch.topk(cos_sims, n_interlopers).indices.tolist()
    int_embs = vocab_embs_norm[int_ids]  # [n, d]

    e_clean = clean_subspace_proj_slerp(
        e_norm, int_embs, target_sim=target_sim
    )  # [d], unit norm

    # Rescale so dot(result, e) == ||e||^2
    # s = ||e|| / target_sim
    e_mag = e.norm(p=2).clamp_min(1e-8)
    s = e_mag / max(target_sim, 1e-8)
    return (s * e_clean).unsqueeze(0)  # [1, d]
