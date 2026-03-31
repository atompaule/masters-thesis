import torch
import torch.nn.functional as F
import torch.optim as optim

from src.latent_embedding_experiments.algorithms.config import CFG
from src.latent_embedding_experiments.algorithms.utils import select_targets


def geometric_solver(
    logits: torch.Tensor,  # [V] — raw logits from LLM
    vocab_embs: torch.Tensor,  # [V, d] — token embedding matrix (unnormalized)
    use_cosine: bool = True,  # if True, use cosine similarity; if False, use dot product
) -> torch.Tensor:
    """
    Geometric solver that optimizes a latent embedding to satisfy four loss terms:
      1. Target ranking loss: preserve pairwise ordering of target similarities
      2. Target margin loss: enforce similarity gaps proportional to probability gaps (adjacent pairs)
      3. Interloper similarity loss: push interloper similarities below weakest target similarity
      4. Target similarity loss: pull embedding toward all target tokens

    All losses are averaged by their number of elements, making them invariant
    to the dynamic k from top-p selection.

    Args:
        logits:      [V] raw logits from the LLM at a given time step
        vocab_embs:  [V, d] token embedding matrix (unnormalized)
        use_cosine:  if True, compute cosine similarity (normalize both sides);
                     if False, compute raw dot products (unnormalized vocab embeddings,
                     latent embedding optimized in full vector space)

    Returns: [1, d] tensor — the optimized latent embedding scaled to target magnitude.
    """
    device = logits.device
    d = vocab_embs.size(1)

    # --- Normalize vocabulary embeddings ---
    vocab_embs_norm = F.normalize(vocab_embs, dim=1)  # [V, d]

    # --- Select target tokens via top-p ---
    target_logits, target_ids = select_targets(logits)
    k = len(target_ids)

    # Target embeddings
    target_embs = vocab_embs[target_ids]
    target_embs_norm = F.normalize(target_embs, dim=1)  # [k, d]

    # Target magnitude (probability-weighted average of target norms)
    target_probs_raw = F.softmax(target_logits, dim=-1)
    target_magnitude = (target_probs_raw * target_embs.norm(dim=1)).sum().item()

    # --- Precompute probability structure for losses 1 and 2 ---
    target_probs = F.softmax(target_logits / CFG.temperature, dim=-1)  # [k]

    # Vocab embeddings used for similarity: normalized for cosine, raw for dot product
    vocab_embs_sim = vocab_embs_norm if use_cosine else vocab_embs
    target_embs_sim = target_embs_norm if use_cosine else target_embs

    with torch.no_grad():
        # Loss 1: all-pairs mask where p_i > p_j
        pairwise_mask = (
            target_probs.unsqueeze(1) > target_probs.unsqueeze(0)
        ).float()  # [k, k]
        n_pairs = pairwise_mask.sum().clamp(min=1)

        # Loss 2: adjacent margins
        target_probs_margins = CFG.margin_scale * (
            target_probs[:-1] - target_probs[1:]
        )  # [k-1]
        n_adjacent = max(k - 1, 1)

    # --- Precompute interlopers ---
    with torch.no_grad():
        centroid_emb_norm = F.normalize(
            target_embs_norm.sum(dim=0, keepdim=True), dim=1
        )
        sims_to_centroid = (centroid_emb_norm @ vocab_embs_norm.T).squeeze(0)  # [V]

        # Mask out target tokens
        interloper_mask = torch.ones(
            vocab_embs_norm.size(0), dtype=torch.bool, device=device
        )
        interloper_mask[target_ids] = False

        interloper_sims_masked = sims_to_centroid.clone()
        interloper_sims_masked[~interloper_mask] = -float("inf")
        n_interlopers = min(CFG.num_interlopers, interloper_mask.sum().item())
        _, interloper_ids = torch.topk(interloper_sims_masked, n_interlopers)
        interloper_embs_sim = vocab_embs_sim[interloper_ids]  # [n_interlopers, d]

    # --- Initialize probe ---
    # Start on the unit sphere
    latent_emb = torch.nn.Parameter(
        F.normalize(torch.randn(d, device=device), dim=0)
    )

    opt = optim.Adam([latent_emb], lr=CFG.lr)

    # --- Optimization loop ---
    for _ in range(CFG.solver_steps):
        opt.zero_grad()

        if use_cosine:
            latent_emb_q = F.normalize(latent_emb.unsqueeze(0), dim=1)  # [1, d]
        else:
            latent_emb_q = latent_emb.unsqueeze(0)  # [1, d], unnormalized

        # Similarities (cosine or dot product)
        target_sims = (latent_emb_q @ target_embs_sim.T).squeeze(0)       # [k]
        interloper_sims = (latent_emb_q @ interloper_embs_sim.T).squeeze(0)  # [n_interlopers]

        # =====================================================================
        # Loss 1: Target ranking loss (averaged over ordered pairs)
        # L_rank = (1/n_pairs) * sum_{p_i > p_j} max(0, s_j - s_i)
        # =====================================================================
        target_sims_diffs = target_sims.unsqueeze(1) - target_sims.unsqueeze(
            0
        )  # [k, k]
        ranking_violations = F.relu(-target_sims_diffs)
        loss_rank = (
            CFG.ranking_weight * (pairwise_mask * ranking_violations).sum() / n_pairs
        )

        # =====================================================================
        # Loss 2: Target margin loss (averaged over adjacent pairs)
        # L_margin = (1/(k-1)) * sum_i max(0, m_min + m_i - (s_i - s_{i+1}))
        # =====================================================================
        target_sims_margins = target_sims[:-1] - target_sims[1:]  # [k-1]
        loss_margin = (
            CFG.margin_weight
            * F.relu(CFG.margin_min + target_probs_margins - target_sims_margins).sum()
            / n_adjacent
        )

        # =====================================================================
        # Loss 3: Interloper similarity loss (averaged over interlopers)
        # L_interloper = (1/n_interlopers) * sum_l max(0, m_l + s_l - s_k)
        # =====================================================================
        loss_interloper = (
            CFG.interloper_weight * F.relu(interloper_sims - 0.2).sum() / n_interlopers
        )

        # =====================================================================
        # Loss 4: Target similarity loss (averaged over targets)
        # L_target = -(lambda_target / k) * sum_i s_i
        # =====================================================================
        loss_target = -CFG.target_sim_weight * target_sims.sum() / k

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
# Loss functions for latent head training (same objectives, differentiable)
# =============================================================================
def latent_head_loss(
    latent_emb: torch.Tensor,  # [d] or [1, d] — latent head output
    target_embs_norm: torch.Tensor,  # [k, d] — L2-normalized target embeddings
    interloper_embs_norm: torch.Tensor,  # [n_interlopers, d] — L2-normalized interloper embeddings
    target_probs: torch.Tensor,  # [k] — temperature-scaled probabilities, sorted descending
    target_magnitude: float,  # desired output norm
    use_cosine: bool = True,  # if True, use cosine similarity; if False, use dot product
) -> dict:
    """
    Compute the four loss terms for training a latent head.
    Same objectives as the geometric solver, but applied to the latent head's output
    so gradients flow back into the head's parameters.

    All losses are averaged by their number of elements.

    Args:
        latent_emb:            [d] or [1, d] latent head output (unnormalized)
        target_embs_norm:      [k, d] L2-normalized target token embeddings
        interloper_embs_norm:  [n_interlopers, d] L2-normalized interloper embeddings
        target_probs:          [k] temperature-scaled probabilities, sorted descending
        target_magnitude:      desired output norm (used for magnitude loss)
        use_cosine:            if True, normalize latent_emb before similarity computation
                               (cosine similarity); if False, use raw dot products against
                               the provided (already normalized) embedding matrices.
                               Note: target_embs_norm and interloper_embs_norm are always
                               passed in normalized; set use_cosine=False to use them as
                               unnormalized dot-product targets by passing unnormalized
                               matrices instead.

    Returns a dict with individual losses and total loss.
    """
    if latent_emb.dim() == 1:
        latent_emb = latent_emb.unsqueeze(0)

    k = target_embs_norm.size(0)
    n_interlopers = interloper_embs_norm.size(0)

    if use_cosine:
        latent_emb_q = F.normalize(latent_emb, dim=1)  # [1, d]
    else:
        latent_emb_q = latent_emb  # [1, d], unnormalized

    # Similarities (cosine or dot product depending on use_cosine and passed-in matrices)
    target_sims = (latent_emb_q @ target_embs_norm.T).squeeze(0)       # [k]
    interloper_sims = (latent_emb_q @ interloper_embs_norm.T).squeeze(0)  # [n_interlopers]

    # --- Loss 1: Target ranking loss (averaged over ordered pairs) ---
    pairwise_mask = (target_probs.unsqueeze(1) > target_probs.unsqueeze(0)).float()
    n_pairs = pairwise_mask.sum().clamp(min=1)
    target_sims_diffs = target_sims.unsqueeze(1) - target_sims.unsqueeze(0)
    loss_rank = (
        CFG.ranking_weight
        * (pairwise_mask * F.relu(-target_sims_diffs)).sum()
        / n_pairs
    )

    # --- Loss 2: Target margin loss (averaged over adjacent pairs) ---
    n_adjacent = max(k - 1, 1)
    target_probs_margins = CFG.margin_scale * (target_probs[:-1] - target_probs[1:])
    target_sims_margins = target_sims[:-1] - target_sims[1:]
    loss_margin = (
        CFG.margin_weight
        * F.relu(CFG.margin_min + target_probs_margins - target_sims_margins).sum()
        / n_adjacent
    )

    # --- Loss 3: Interloper similarity loss (averaged over interlopers) ---
    target_sim_k = target_sims[-1]
    loss_interloper = (
        CFG.interloper_weight
        * F.relu(CFG.interloper_margin_relative + interloper_sims - target_sim_k).sum()
        / max(n_interlopers, 1)
    )

    # --- Loss 4: Target similarity loss (averaged over targets) ---
    loss_target = -CFG.target_sim_weight * target_sims.sum() / k

    # --- Norm loss ---
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