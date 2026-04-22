import torch
import torch.nn.functional as F

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
    temperature = 0.6
    top_p = 0.95

    # Select tokens via top-p
    target_props, target_ids = select_targets(logits, temperature, top_p)

    # Weighted sum of embeddings
    target_embs = vocab_embs[target_ids]  # [k, d]
    result = (target_props.unsqueeze(1) * target_embs).sum(dim=0, keepdim=True)  # [1, d]

    return result