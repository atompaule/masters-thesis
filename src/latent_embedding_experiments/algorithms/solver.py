from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torch.optim as optim


@dataclass
class Config:
    solver_steps: int = 300
    lr: float = 0.05

    # Number of interlopers to consider in loss 3
    num_interlopers: int = 500

    # Loss 1: Target ranking loss
    # Pure ordering — fires when any pair (i, j) with p_i > p_j has s_j > s_i
    ranking_weight: float = 3.0

    # Loss 2: Target margin loss
    # Adjacent pairs only — demands similarity gap proportional to probability gap
    margin_weight: float = 10.0
    margin_scale: float = 1.0

    # Loss 3: Interloper similarity loss
    # Pushes interloper similarities below the weakest target's similarity
    interloper_weight: float = 2.0
    interloper_margin: float = 0.03

    # Loss 4: Target similarity loss
    # Pulls the embedding toward all target tokens (unweighted)
    target_similarity_weight: float = 5.0


CFG = Config()


def geometric_solver(
    target_norm: torch.Tensor,       # [k, d] — L2-normalized embeddings of top-k target tokens
    target_ids: torch.Tensor,        # [k]    — token indices of top-k targets
    dict_norm: torch.Tensor,         # [V, d] — L2-normalized full vocabulary embeddings
    magnitude: float,                # scalar — desired output norm (e.g., weighted avg of target norms)
    pool_logits: torch.Tensor,       # [k]    — logits (or probabilities) for the top-k targets
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Geometric solver that optimizes a latent embedding to satisfy four loss terms:
      1. Target ranking loss: preserve pairwise ordering of target similarities
      2. Target margin loss: enforce similarity gaps proportional to probability gaps (adjacent pairs)
      3. Interloper similarity loss: push interloper similarities below weakest target similarity
      4. Target similarity loss: pull embedding toward all target tokens

    Returns: [1, d] tensor — the optimized latent embedding scaled to desired magnitude.
    """
    k = len(target_ids)
    device = dict_norm.device

    # Initialize probe as normalized centroid of targets
    base = F.normalize(target_norm.sum(dim=0, keepdim=True), dim=1)

    # --- Precompute interlopers (tokens close to centroid but NOT in top-k targets) ---
    with torch.no_grad():
        sims_to_base = (base @ dict_norm.T).squeeze(0)  # [V]

        # Mask out target tokens
        interloper_mask = torch.ones(dict_norm.size(0), dtype=torch.bool, device=device)
        interloper_mask[target_ids] = False

        # Get top-N interlopers by similarity to centroid
        interloper_sims = sims_to_base.clone()
        interloper_sims[~interloper_mask] = -float("inf")
        _, interloper_ids = torch.topk(interloper_sims, CFG.num_interlopers)
        interloper_embs = dict_norm[interloper_ids]  # [num_interlopers, d]

    # --- Precompute probability structure for losses 1 and 2 ---
    temp_probs = F.softmax(pool_logits / temperature, dim=-1)  # [k]

    with torch.no_grad():
        # Loss 1: all-pairs mask where p_i > p_j
        # pairwise_mask[i, j] = 1 if p_i > p_j (meaning s_i should be > s_j)
        pairwise_mask = (temp_probs.unsqueeze(1) > temp_probs.unsqueeze(0)).float()  # [k, k]

        # Loss 2: adjacent margins
        # For adjacent pairs (i, i+1) in descending probability order,
        # margin_i = margin_scale * (p_i - p_{i+1})
        # Note: pool_logits / temp_probs should already be sorted descending
        adjacent_margins = CFG.margin_scale * (temp_probs[:-1] - temp_probs[1:])  # [k-1]

    # --- Optimization loop ---
    probe = torch.nn.Parameter(base.squeeze(0).clone())  # [d]
    opt = optim.Adam([probe], lr=CFG.lr)

    for _ in range(CFG.solver_steps):
        opt.zero_grad()

        # Normalize probe for cosine similarity computation
        p = F.normalize(probe.unsqueeze(0), dim=1)  # [1, d]

        # Cosine similarities to targets
        sims_target = (p @ target_norm.T).squeeze(0)  # [k]

        # Cosine similarities to interlopers
        sims_interloper = (p @ interloper_embs.T).squeeze(0)  # [num_interlopers]

        # =====================================================================
        # Loss 1: Target ranking loss
        # For all pairs (i, j) where p_i > p_j: penalize if s_j > s_i
        # L_rank = sum_{p_i > p_j} max(0, s_j - s_i)
        # =====================================================================
        sim_diffs = sims_target.unsqueeze(1) - sims_target.unsqueeze(0)  # [k, k]
        # sim_diffs[i, j] = s_i - s_j
        # We want s_i > s_j when p_i > p_j, i.e., sim_diffs[i,j] > 0
        # Loss fires when sim_diffs[i,j] < 0, i.e., max(0, -sim_diffs[i,j]) = max(0, s_j - s_i)
        ranking_violations = F.relu(-sim_diffs)  # [k, k]
        loss_rank = CFG.ranking_weight * (pairwise_mask * ranking_violations).sum()

        # =====================================================================
        # Loss 2: Target margin loss (adjacent pairs only)
        # For each adjacent pair (i, i+1) in descending probability order:
        # L_margin = sum_i max(0, m_i - (s_i - s_{i+1}))
        # where m_i = margin_scale * (p_i - p_{i+1})
        # =====================================================================
        adjacent_sim_diffs = sims_target[:-1] - sims_target[1:]  # [k-1]
        loss_margin = CFG.margin_weight * F.relu(adjacent_margins - adjacent_sim_diffs).sum()

        # =====================================================================
        # Loss 3: Interloper similarity loss
        # For each interloper l: penalize if s_l is too close to s_k (weakest target)
        # L_interloper = sum_l max(0, m_l + s_l - s_k)
        # where m_l is a fixed margin and s_k = min target similarity
        # =====================================================================
        s_k = sims_target[-1]  # similarity to weakest target (last in sorted order)
        loss_interloper = CFG.interloper_weight * F.relu(
            CFG.interloper_margin + sims_interloper - s_k
        ).sum()

        # =====================================================================
        # Loss 4: Target similarity loss
        # Pull embedding toward all target tokens
        # L_target = -lambda_target * sum_i s_i, for i in top-k
        # =====================================================================
        loss_target = -CFG.target_similarity_weight * sims_target.sum()

        # =====================================================================
        # Total loss
        # =====================================================================
        loss = loss_rank + loss_margin + loss_interloper + loss_target
        loss.backward()
        opt.step()

    # Return optimized embedding scaled to desired magnitude
    with torch.no_grad():
        result = F.normalize(probe.unsqueeze(0), dim=1) * magnitude

    return result


# =============================================================================
# Convenience function: extract inputs from a logit vector and embedding matrix
# =============================================================================
def prepare_solver_inputs(
    logits: torch.Tensor,          # [V] — raw logits from LLM
    embedding_matrix: torch.Tensor, # [V, d] — token embedding matrix (unnormalized)
    top_k: int = 10,
) -> dict:
    """
    Given raw logits and the embedding matrix, prepare all inputs needed
    for the geometric solver.

    Returns a dict with keys:
        target_norm, target_ids, dict_norm, magnitude, pool_logits
    """
    # Get top-k tokens
    pool_logits, target_ids = torch.topk(logits, top_k)

    # Normalize full embedding matrix
    dict_norm = F.normalize(embedding_matrix, dim=1)

    # Get target embeddings (unnormalized for magnitude, normalized for solver)
    target_embs = embedding_matrix[target_ids]
    target_norm = F.normalize(target_embs, dim=1)

    # Compute desired magnitude as probability-weighted average of target norms
    probs = F.softmax(pool_logits, dim=-1)
    target_norms = target_embs.norm(dim=1)
    magnitude = (probs * target_norms).sum().item()

    return {
        "target_norm": target_norm,
        "target_ids": target_ids,
        "dict_norm": dict_norm,
        "magnitude": magnitude,
        "pool_logits": pool_logits,
    }


# =============================================================================
# Loss functions for latent head training (same objectives, differentiable)
# =============================================================================
def latent_head_loss(
    z_hat: torch.Tensor,             # [d] or [1, d] — latent head output
    target_norm: torch.Tensor,       # [k, d] — L2-normalized target embeddings
    interloper_embs: torch.Tensor,   # [num_interlopers, d] — L2-normalized interloper embeddings
    temp_probs: torch.Tensor,        # [k] — temperature-scaled probabilities, sorted descending
    target_magnitude: float,         # desired output norm
    cfg: Config = CFG,
) -> dict:
    """
    Compute the four loss terms for training a latent head.
    Same objectives as the geometric solver, but applied to the latent head's output
    so gradients flow back into the head's parameters.

    Returns a dict with individual losses and total loss.
    """
    if z_hat.dim() == 1:
        z_hat = z_hat.unsqueeze(0)

    k = target_norm.size(0)

    # Normalize for cosine similarity
    z_norm = F.normalize(z_hat, dim=1)  # [1, d]

    # Cosine similarities
    sims_target = (z_norm @ target_norm.T).squeeze(0)      # [k]
    sims_interloper = (z_norm @ interloper_embs.T).squeeze(0)  # [num_interlopers]

    # --- Loss 1: Target ranking loss ---
    pairwise_mask = (temp_probs.unsqueeze(1) > temp_probs.unsqueeze(0)).float()
    sim_diffs = sims_target.unsqueeze(1) - sims_target.unsqueeze(0)
    loss_rank = cfg.ranking_weight * (pairwise_mask * F.relu(-sim_diffs)).sum()

    # --- Loss 2: Target margin loss (adjacent pairs) ---
    adjacent_margins = cfg.margin_scale * (temp_probs[:-1] - temp_probs[1:])
    adjacent_sim_diffs = sims_target[:-1] - sims_target[1:]
    loss_margin = cfg.margin_weight * F.relu(adjacent_margins - adjacent_sim_diffs).sum()

    # --- Loss 3: Interloper similarity loss ---
    s_k = sims_target[-1]  # weakest target
    loss_interloper = cfg.interloper_weight * F.relu(
        cfg.interloper_margin + sims_interloper - s_k
    ).sum()

    # --- Loss 4: Target similarity loss ---
    loss_target = -cfg.target_similarity_weight * sims_target.sum()

    # --- Norm loss (keep output magnitude plausible) ---
    loss_norm = (z_hat.norm() - target_magnitude) ** 2

    total = loss_rank + loss_margin + loss_interloper + loss_target + loss_norm

    return {
        "loss_rank": loss_rank,
        "loss_margin": loss_margin,
        "loss_interloper": loss_interloper,
        "loss_target": loss_target,
        "loss_norm": loss_norm,
        "total": total,
    }