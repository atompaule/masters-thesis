import torch
import torch.nn.functional as F
import torch.optim as optim

from src.latent_embedding_experiments.algorithms.config import CFG
from src.latent_embedding_experiments.algorithms.utils import select_targets


def geometric_solver(
    logits: torch.Tensor,  # [V] — raw logits from LLM
    vocab_embs: torch.Tensor,  # [V, d] — token embedding matrix (unnormalized)
    use_cosine: bool = False,  # if True, use cosine similarity; if False, use dot product
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

    vocab_embs_norm = F.normalize(vocab_embs, dim=1)  # [V, d]

    target_logits, target_ids = select_targets(logits)
    k = len(target_ids)

    target_embs = vocab_embs[target_ids]
    target_embs_norm = F.normalize(target_embs, dim=1)  # [k, d]

    target_probs_raw = F.softmax(target_logits, dim=-1)
    target_magnitude = (target_probs_raw * target_embs.norm(dim=1)).sum().item()

    target_probs = F.softmax(target_logits / CFG.temperature, dim=-1)  # [k]

    vocab_embs_sim = vocab_embs_norm if use_cosine else vocab_embs
    target_embs_sim = target_embs_norm if use_cosine else target_embs

    with torch.no_grad():
        pairwise_mask = (
            target_probs.unsqueeze(1) > target_probs.unsqueeze(0)
        ).float()  # [k, k]
        n_pairs = pairwise_mask.sum().clamp(min=1)

        target_probs_margins = CFG.margin_scale * (
            target_probs[:-1] - target_probs[1:]
        )  # [k-1]
        n_adjacent = max(k - 1, 1)

    with torch.no_grad():
        centroid_emb_norm = F.normalize(
            target_embs_norm.sum(dim=0, keepdim=True), dim=1
        )
        sims_to_centroid = (centroid_emb_norm @ vocab_embs_norm.T).squeeze(0)  # [V]

        interloper_mask = torch.ones(
            vocab_embs_norm.size(0), dtype=torch.bool, device=device
        )
        interloper_mask[target_ids] = False

        interloper_sims_masked = sims_to_centroid.clone()
        interloper_sims_masked[~interloper_mask] = -float("inf")
        n_interlopers = min(CFG.num_interlopers, interloper_mask.sum().item())
        _, interloper_ids = torch.topk(interloper_sims_masked, n_interlopers)
        interloper_embs_sim = vocab_embs_sim[interloper_ids]  # [n_interlopers, d]

    latent_emb = torch.nn.Parameter(F.normalize(torch.randn(d, device=device), dim=0))

    opt = optim.Adam([latent_emb], lr=CFG.lr)

    for _ in range(CFG.solver_steps):
        opt.zero_grad()

        latent_emb_q = F.normalize(latent_emb.unsqueeze(0), dim=1)  # [1, d]

        target_sims = (latent_emb_q @ target_embs_sim.T).squeeze(0)  # [k]
        interloper_sims = (latent_emb_q @ interloper_embs_sim.T).squeeze(
            0
        )  # [n_interlopers]

        # Loss 1: Target ranking loss (averaged over ordered pairs)
        # L_rank = (1/n_pairs) * sum_{p_i > p_j} max(0, s_j - s_i)
        target_sims_diffs = target_sims.unsqueeze(1) - target_sims.unsqueeze(
            0
        )  # [k, k]
        ranking_violations = F.relu(-target_sims_diffs)
        loss_rank = (
            CFG.ranking_weight * (pairwise_mask * ranking_violations).sum() / n_pairs
        )

        # Loss 2: Target margin loss (averaged over adjacent pairs)
        # L_margin = (1/(k-1)) * sum_i max(0, m_min + m_i - (s_i - s_{i+1}))
        target_sims_margins = target_sims[:-1] - target_sims[1:]  # [k-1]
        loss_margin = (
            CFG.margin_weight
            * F.relu(CFG.margin_min + target_probs_margins - target_sims_margins).sum()
            / n_adjacent
        )

        # Loss 3: Interloper similarity loss (averaged over interlopers)
        # L_interloper = (1/n_interlopers) * sum_l max(0, m_l + s_l - s_k)
        loss_interloper = (
            CFG.interloper_weight
            * F.relu(interloper_sims - CFG.interloper_margin_threshold).sum()
            / n_interlopers
        )

        # Loss 4: Target similarity loss (averaged over targets)
        # L_target = -(lambda_target / k) * sum_i s_i
        loss_target = -CFG.target_sim_weight * target_sims.sum() / k

        # Total loss
        loss = loss_rank + loss_margin + loss_target + loss_interloper
        loss.backward()
        opt.step()

    # Scale to desired magnitude only at the end
    with torch.no_grad():
        result = F.normalize(latent_emb.unsqueeze(0), dim=1) * target_magnitude

    return result


def latent_head_loss(
    latent: torch.Tensor,  # [B, L, d] — latent head output
    logits: torch.Tensor,  # [B, L, V] — raw logits from the LLM
    vocab_embs: torch.Tensor,  # [V, d] — token embedding matrix (unnormalized)
    vocab_embs_norm: torch.Tensor,  # [V, d] — token embedding matrix (normalized)
    attention_mask: torch.Tensor,  # [B, L] — 1 for real tokens, 0 for padding
    use_cosine: bool = False,
) -> dict:
    device = logits.device
    B, L, _ = latent.shape

    # ── Target selection ───────────────────────────────────────────────────────
    target_logits, target_ids, k_per_pos = select_targets(
        logits
    )  # [B, L, k], [B, L, k], [B, L]

    pad_mask = target_ids == -1  # [B, L, k] — padded slots
    valid_mask = ~pad_mask  # [B, L, k]
    target_ids = target_ids.clamp(min=0)  # [B, L, k]

    # ── Embeddings ─────────────────────────────────────────────────────────────
    vocab_embs_sim = vocab_embs_norm if use_cosine else vocab_embs

    target_embs_norm = vocab_embs_norm[target_ids]  # [B, L, k, d]
    target_embs_sim = vocab_embs_sim[target_ids]  # [B, L, k, d]
    target_embs_norm[pad_mask] = 0.0
    target_embs_sim[pad_mask] = 0.0

    # ── Target probs (temperature-scaled, padded slots → 0 via -inf) ──────────
    target_probs = F.softmax(target_logits / CFG.temperature, dim=-1)  # [B, L, k]
    target_probs = target_probs * valid_mask.float()  # zero out padding

    # ── Latent query ───────────────────────────────────────────────────────────
    latent_emb_q = F.normalize(latent, dim=-1)  # [B, L, d]

    # target_sims[b, l, i] = sim(latent[b,l], target_emb[b,l,i])
    target_sims = torch.einsum(
        "bld,blkd->blk", latent_emb_q, target_embs_sim
    )  # [B, L, k]
    target_sims = target_sims * valid_mask.float()  # zero out padded slots

    # ── Interlopers (per position) ─────────────────────────────────────────────
    with torch.no_grad():
        # Centroid of target embeddings per position [B, L, d]
        centroid = target_embs_norm.sum(dim=2)  # [B, L, d]
        centroid_norm = F.normalize(centroid, dim=-1)  # [B, L, d]

        # Similarity of each centroid to full vocab [B, L, V]
        sims_to_centroid = torch.einsum("bld,vd->blv", centroid_norm, vocab_embs_norm)

        # Mask out target ids per position
        interloper_mask = torch.ones(
            B, L, vocab_embs_norm.size(0), dtype=torch.bool, device=device
        )
        # Scatter False at target positions
        interloper_mask.scatter_(2, target_ids, False)
        # Also mask padding targets back to True (they were clamped to 0)
        interloper_mask.scatter_(2, target_ids * valid_mask.long(), False)

        sims_to_centroid[~interloper_mask] = -float("inf")
        n_interlopers = CFG.num_interlopers
        _, interloper_ids = torch.topk(
            sims_to_centroid, n_interlopers, dim=-1
        )  # [B, L, n_interlopers]

    interloper_embs_sim = vocab_embs_sim[interloper_ids]  # [B, L, n_interlopers, d]
    interloper_sims = torch.einsum(
        "bld,blnd->bln", latent_emb_q, interloper_embs_sim
    )  # [B, L, n_interlopers]

    # ── Valid position mask for loss averaging ─────────────────────────────────
    pos_valid = attention_mask.bool()  # [B, L]
    n_valid = pos_valid.float().sum().clamp(min=1)

    # ── Loss 1: Ranking ────────────────────────────────────────────────────────
    # pairwise_mask[b,l,i,j] = 1 if p_i > p_j
    pairwise_mask = (
        target_probs.unsqueeze(-1) > target_probs.unsqueeze(-2)
    ).float()  # [B, L, k, k]
    pairwise_mask = (
        pairwise_mask
        * valid_mask.unsqueeze(-1).float()
        * valid_mask.unsqueeze(-2).float()
    )

    sims_diffs = target_sims.unsqueeze(-1) - target_sims.unsqueeze(-2)  # [B, L, k, k]
    ranking_violations = F.relu(-sims_diffs) * pairwise_mask
    n_pairs = pairwise_mask.sum(dim=(-1, -2)).clamp(min=1)  # [B, L]
    loss_rank_per_pos = ranking_violations.sum(dim=(-1, -2)) / n_pairs  # [B, L]
    loss_rank = (
        CFG.ranking_weight * (loss_rank_per_pos * pos_valid.float()).sum() / n_valid
    )

    # ── Loss 2: Margin ─────────────────────────────────────────────────────────
    target_probs_margins = CFG.margin_scale * (
        target_probs[..., :-1] - target_probs[..., 1:]
    )  # [B, L, k-1]
    sims_margins = target_sims[..., :-1] - target_sims[..., 1:]  # [B, L, k-1]

    # Only include adjacent pairs where both slots are valid
    adj_valid = valid_mask[..., :-1] & valid_mask[..., 1:]  # [B, L, k-1]
    n_adjacent = adj_valid.float().sum(dim=-1).clamp(min=1)  # [B, L]

    margin_violations = F.relu(CFG.margin_min + target_probs_margins - sims_margins)
    margin_violations = margin_violations * adj_valid.float()
    loss_margin_per_pos = margin_violations.sum(dim=-1) / n_adjacent  # [B, L]
    loss_margin = (
        CFG.margin_weight * (loss_margin_per_pos * pos_valid.float()).sum() / n_valid
    )

    # ── Loss 3: Interloper ─────────────────────────────────────────────────────
    interloper_violations = F.relu(interloper_sims - CFG.interloper_margin_threshold)
    loss_interloper_per_pos = (
        interloper_violations.sum(dim=-1) / n_interlopers
    )  # [B, L]
    loss_interloper = (
        CFG.interloper_weight
        * (loss_interloper_per_pos * pos_valid.float()).sum()
        / n_valid
    )

    # ── Loss 4: Target similarity ──────────────────────────────────────────────
    k_per_pos_clamped = k_per_pos.float().clamp(min=1)  # [B, L]
    loss_target_per_pos = -target_sims.sum(dim=-1) / k_per_pos_clamped  # [B, L]
    loss_target = (
        CFG.target_sim_weight
        * (loss_target_per_pos * pos_valid.float()).sum()
        / n_valid
    )

    total_loss = loss_rank + loss_margin + loss_target + loss_interloper

    return {
        "loss_rank": loss_rank,
        "loss_margin": loss_margin,
        "loss_interloper": loss_interloper,
        "loss_target": loss_target,
        "total_loss": total_loss,
    }
