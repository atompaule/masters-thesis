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


def soft_thinking_normalized(
    logits: torch.Tensor,  # [V]
    vocab_embs_norm: torch.Tensor,  # [V, d] — normalized embeddings
    target_magnitude: torch.Tensor,
) -> torch.Tensor:
    """
    Soft thinking in direction space, followed by magnitude injection.

    Computes a probability-weighted average over normalized embeddings,
    then rescales the result to match the provided target magnitude.

    Args:
        logits:             [V] raw logits from the LLM
        vocab_embs_norm:    [V, d] normalized embedding matrix
        target_magnitude:   scalar tensor for final vector norm

    Returns:
        [1, d] tensor — normalized soft thinking embedding with target magnitude.
    """
    # Select tokens via top-p
    target_logits, target_ids = select_targets(logits)

    # Probability weights over selected subset
    probs = F.softmax(target_logits / CFG.temperature, dim=-1)  # [k]

    # Weighted sum in normalized space
    target_embs = vocab_embs_norm[target_ids]  # [k, d]
    direction = (probs.unsqueeze(1) * target_embs).sum(dim=0)  # [d]

    # Normalize direction (important: weighted sums drift off-sphere)
    direction = F.normalize(direction.unsqueeze(0), dim=1).squeeze(0)

    # Inject magnitude
    return (target_magnitude * direction).unsqueeze(0)
