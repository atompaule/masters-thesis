import torch
import torch.nn.functional as F

from src.latent_embedding_experiments.algorithms.config import CFG
from src.latent_embedding_experiments.algorithms.utils import select_targets


def soft_thinking(
    logits: torch.Tensor,  # [V] — raw logits from LLM
    vocab_embs: torch.Tensor,  # [V, d] — token embedding matrix (unnormalized)
) -> torch.Tensor:
    """
    Soft thinking: probability-weighted sum of token embeddings over the top-p nucleus.

    Selects tokens via top-p, computes softmax(logits/temperature) over the selected
    subset, and returns the weighted sum of their embeddings.

    Args:
        logits:      [V] raw logits from the LLM
        vocab_embs:  [V, d] token embedding matrix (unnormalized)

    Returns: [1, d] tensor — the soft thinking embedding.
    """
    # Select tokens via top-p
    target_logits, target_ids = select_targets(logits)

    # Probability weights over selected subset
    probs = F.softmax(target_logits / CFG.temperature, dim=-1)  # [k]

    # Weighted sum of embeddings
    target_embs = vocab_embs[target_ids]  # [k, d]
    result = (probs.unsqueeze(1) * target_embs).sum(dim=0, keepdim=True)  # [1, d]

    return result
