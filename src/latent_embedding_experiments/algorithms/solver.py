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
    ranking_weight: float = 0.0#3.0

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
    target_sim_weight: float = 5.0


CFG = Config()


def geometric_solver(
    target_embs_norm: torch.Tensor,  # [k, d] — L2-normalized embeddings of top-k target tokens
    target_ids: torch.Tensor,  # [k]    — token indices of top-k targets
    vocab_embs_norm: torch.Tensor,  # [V, d] — L2-normalized full vocabulary embeddings
    target_magnitude: float,  # scalar — desired output norm (e.g., weighted avg of target norms)
    target_logits: torch.Tensor,  # [k]    — logits (or probabilities) for the top-k targets
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Geometric solver that optimizes a latent embedding to satisfy four loss terms:
      1. Target ranking loss: preserve pairwise ordering of target similarities
      2. Target margin loss: enforce similarity gaps proportional to probability gaps (adjacent pairs)
      3. Interloper similarity loss: push interloper similarities below weakest target similarity
      4. Target similarity loss: pull embedding toward all target tokens

    The solver operates entirely in normalized (unit sphere) space. The embedding is
    initialized as random noise on the unit sphere. Magnitude is only applied to the
    final output.

    Returns: [1, d] tensor — the optimized latent embedding scaled to desired magnitude.
    """
    device = vocab_embs_norm.device
    d = vocab_embs_norm.size(1)

    latent_emb_init = F.normalize(torch.randn(d, device=device), dim=0)
    latent_emb = torch.nn.Parameter(latent_emb_init)  # [d]

    opt = optim.Adam([latent_emb], lr=CFG.lr)

    target_probs = F.softmax(target_logits / temperature, dim=-1)  # [k]
    with torch.no_grad():
        # Loss 1: all-pairs mask where p_i > p_j
        pairwise_mask = (
            target_probs.unsqueeze(1) > target_probs.unsqueeze(0)
        ).float()  # [k, k]

        # Loss 2: adjacent margins
        target_probs_margins = CFG.margin_scale * (
            target_probs[:-1] - target_probs[1:]
        )  # [k-1]

    # --- Precompute interlopers (tokens close to target centroid but NOT in top-k targets) ---
    with torch.no_grad():
        centroid_emb_norm = F.normalize(
            target_embs_norm.sum(dim=0, keepdim=True), dim=1
        )
        sims_to_centroid = (centroid_emb_norm @ vocab_embs_norm.T).squeeze(0)  # [V]

        interloper_mask = torch.ones(
            vocab_embs_norm.size(0), dtype=torch.bool, device=device
        )
        interloper_mask[target_ids] = False

        # Get top-N interlopers by similarity to target centroid
        interloper_sims = sims_to_centroid.clone()
        interloper_sims[~interloper_mask] = -float("inf")
        _, interloper_ids = torch.topk(interloper_sims, CFG.num_interlopers)
        interloper_embs_norm = vocab_embs_norm[interloper_ids]  # [num_interlopers, d]

    # --- Optimization loop ---

    for _ in range(CFG.solver_steps):
        opt.zero_grad()

        # Normalize probe for cosine similarity computation
        latent_emb_norm = F.normalize(latent_emb.unsqueeze(0), dim=1)  # [1, d]

        # Cosine similarities to targets and interlopers
        target_sims = (latent_emb_norm @ target_embs_norm.T).squeeze(0)  # [k]
        interloper_sims = (latent_emb_norm @ interloper_embs_norm.T).squeeze(
            0
        )  # [num_interlopers]

        # =====================================================================
        # Loss 1: Target ranking loss
        # For all pairs (i, j) where p_i > p_j: penalize if s_j > s_i
        # L_rank = sum_{p_i > p_j} max(0, s_j - s_i)
        # =====================================================================
        target_sims_diffs = target_sims.unsqueeze(1) - target_sims.unsqueeze(
            0
        )  # [k, k]
        ranking_violations = F.relu(-target_sims_diffs)  # [k, k]
        loss_rank = CFG.ranking_weight * (pairwise_mask * ranking_violations).sum()

        # =====================================================================
        # Loss 2: Target margin loss (adjacent pairs only)
        # For each adjacent pair (i, i+1) in descending probability order:
        # L_margin = sum_i max(0, m_i - (s_i - s_{i+1}))
        # where m_i = margin_scale * (p_i - p_{i+1})
        # =====================================================================
        target_sims_margins = target_sims[:-1] - target_sims[1:]  # [k-1]
        loss_margin = (
            CFG.margin_weight * F.relu(target_probs_margins - target_sims_margins).sum()
        )

        # =====================================================================
        # Loss 3: Interloper similarity loss
        # For each interloper l: penalize if s_l is too close to s_k (weakest target)
        # L_interloper = sum_l max(0, m_l + s_l - s_k)
        # where m_l is a fixed margin and s_k = min target similarity
        # =====================================================================
        target_sim_k = target_sims[-1]  # similarity to weakest target
        loss_interloper = (
            CFG.interloper_weight
            * F.relu(CFG.interloper_margin + interloper_sims - target_sim_k).sum()
        )

        # =====================================================================
        # Loss 4: Target similarity loss
        # Pull embedding toward all target tokens
        # L_target = -lambda_target * sum_i s_i, for i in top-k
        # =====================================================================
        loss_target = -CFG.target_sim_weight * target_sims.sum()

        # =====================================================================
        # Total loss
        # =====================================================================
        loss = loss_rank + loss_margin + loss_interloper + loss_target
        loss.backward()
        opt.step()

    # Scale to desired magnitude only at the end
    with torch.no_grad():
        result = F.normalize(latent_emb.unsqueeze(0), dim=1) * target_magnitude

    return result


# =============================================================================
# Convenience function: extract inputs from a logit vector and embedding matrix
# =============================================================================
def prepare_solver_inputs(
    logits: torch.Tensor,  # [V] — raw logits from LLM
    vocab_embs: torch.Tensor,  # [V, d] — token embedding matrix (unnormalized)
    top_k: int = 10,
) -> dict:
    """
    Given raw logits and the embedding matrix, prepare all inputs needed
    for the geometric solver.

    Returns a dict with keys:
        target_norm, target_ids, dict_norm, magnitude, pool_logits
    """
    # Get top-k tokens
    target_logits, target_ids = torch.topk(logits, top_k)

    # Normalize full embedding matrix
    vocab_embs_norm = F.normalize(vocab_embs, dim=1)

    # Get target embeddings (unnormalized for magnitude, normalized for solver)
    target_embs = vocab_embs[target_ids]
    target_embs_norm = F.normalize(target_embs, dim=1)

    # Compute desired magnitude as probability-weighted average of target norms
    target_probs = F.softmax(target_logits, dim=-1)
    target_magnitudes = target_embs.norm(dim=1)
    target_magnitude = (target_probs * target_magnitudes).sum().item()

    return {
        "target_embs_norm": target_embs_norm,
        "target_ids": target_ids,
        "vocab_embs_norm": vocab_embs_norm,
        "target_magnitude": target_magnitude,
        "target_logits": target_logits,
    }


# =============================================================================
# Loss functions for latent head training (same objectives, differentiable)
# =============================================================================
def latent_head_loss(
    latent_emb: torch.Tensor,  # [d] or [1, d] — latent head output
    target_embs_norm: torch.Tensor,  # [k, d] — L2-normalized target embeddings
    interloper_embs_norm: torch.Tensor,  # [num_interlopers, d] — L2-normalized interloper embeddings
    target_probs: torch.Tensor,  # [k] — temperature-scaled probabilities, sorted descending
    target_magnitude: float,  # desired output norm
    cfg: Config = CFG,
) -> dict:
    """
    Compute the four loss terms for training a latent head.
    Same objectives as the geometric solver, but applied to the latent head's output
    so gradients flow back into the head's parameters.

    Returns a dict with individual losses and total loss.
    """
    if latent_emb.dim() == 1:
        latent_emb = latent_emb.unsqueeze(0)

    # Normalize for cosine similarity
    latent_emb_norm = F.normalize(latent_emb, dim=1)  # [1, d]

    # Cosine similarities
    target_sims = (latent_emb_norm @ target_embs_norm.T).squeeze(0)  # [k]
    interloper_sims = (latent_emb_norm @ interloper_embs_norm.T).squeeze(
        0
    )  # [num_interlopers]

    # --- Loss 1: Target ranking loss ---
    pairwise_mask = (target_probs.unsqueeze(1) > target_probs.unsqueeze(0)).float()
    target_sims_diffs = target_sims.unsqueeze(1) - target_sims.unsqueeze(0)
    loss_rank = cfg.ranking_weight * (pairwise_mask * F.relu(-target_sims_diffs)).sum()

    # --- Loss 2: Target margin loss (adjacent pairs) ---
    target_probs_margins = cfg.margin_scale * (target_probs[:-1] - target_probs[1:])
    target_sims_margins = target_sims[:-1] - target_sims[1:]
    loss_margin = (
        cfg.margin_weight * F.relu(target_probs_margins - target_sims_margins).sum()
    )

    # --- Loss 3: Interloper similarity loss ---
    target_sim_k = target_sims[-1]  # weakest target
    loss_interloper = (
        cfg.interloper_weight
        * F.relu(cfg.interloper_margin + interloper_sims - target_sim_k).sum()
    )

    # --- Loss 4: Target similarity loss ---
    loss_target = -cfg.target_sim_weight * target_sims.sum()

    # --- Norm loss (keep output magnitude plausible) ---
    loss_magnitude = (latent_emb.norm() - target_magnitude) ** 2

    loss = loss_rank + loss_margin + loss_interloper + loss_target + loss_magnitude

    return {
        "loss_rank": loss_rank,
        "loss_margin": loss_margin,
        "loss_interloper": loss_interloper,
        "loss_target": loss_target,
        "loss_magnitude": loss_magnitude,
        "loss": loss,
    }
