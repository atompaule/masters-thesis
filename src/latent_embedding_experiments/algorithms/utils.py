import torch
import torch.nn.functional as F

from src.latent_embedding_experiments.algorithms.config import CFG


def select_targets(
    logits: torch.Tensor,  # [V] or [B, L, V] — raw logits
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Select target tokens via min-p filtering on the full distribution:
      1. Softmax over full vocabulary (no temperature — raw probabilities)
      2. Keep all tokens with p >= min_p
      3. Keep at least min_k tokens
      4. Return sorted descending by logit

    Returns:
        target_logits: [k] — logits of selected tokens, sorted descending
        target_ids:    [k] — token indices of selected tokens
    """
    # Full probability distribution (raw, no temperature)
    probs = F.softmax(logits, dim=-1)  # [V] or [B, L, V]

    # Sort descending
    sorted_probs, sorted_ids = torch.sort(probs, descending=True, dim=-1)

    # Min-p filtering: find how many tokens pass the probability threshold
    mask = sorted_probs >= CFG.min_p  # [V] or [B, L, V]
    if logits.dim() == 1:
        k = max(int(mask.sum().item()), CFG.min_k)  # scalar

        target_ids = sorted_ids[:k]
        target_logits = logits[target_ids]

        return target_logits, target_ids
    else:
        # Per-position k: [B, L], then take global max for uniform tensor shape
        k_per_pos = mask.sum(dim=-1).clamp(min=CFG.min_k)  # [B, L]
        k_max = int(k_per_pos.max().item())

        target_ids = sorted_ids[..., :k_max].clone()  # [B, L, k_max]
        target_logits = torch.gather(logits, -1, target_ids).clone()  # [B, L, k_max])

        pos_idx = torch.arange(k_max, device=logits.device)  # [k_max]
        pad_mask = pos_idx.unsqueeze(0).unsqueeze(0) >= k_per_pos.unsqueeze(
            -1
        )  # [B, L, k]

        target_ids[pad_mask] = -1
        target_logits[pad_mask] = float("-inf")

        return target_logits, target_ids, k_per_pos
