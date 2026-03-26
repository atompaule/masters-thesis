import torch
import torch.nn.functional as F

from src.latent_embedding_experiments.algorithms.config import CFG


def select_targets(
    logits: torch.Tensor,  # [V] — raw logits
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
    # Full distribution (raw, no temperature)
    probs = F.softmax(logits, dim=-1)  # [V]

    # Sort descending
    sorted_probs, sorted_ids = torch.sort(probs, descending=True)

    # Min-p filtering: find how many tokens pass the threshold
    mask = sorted_probs >= CFG.min_p
    k = max(int(mask.sum().item()), CFG.min_k)

    target_ids = sorted_ids[:k]
    target_logits = logits[target_ids]

    return target_logits, target_ids
