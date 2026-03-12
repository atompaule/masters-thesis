import torch
import torch.nn.functional as F


def forge_fluid_chimera(target_ids, dict_norm, mean, std, magnitude_target, adj_probs):
    """
    The original strict Chimera. Only splices the exact predicted targets.
    """
    target_embs = dict_norm[target_ids]
    k = len(target_ids)

    z_scores = (target_embs - mean) / std
    abs_z = torch.abs(z_scores)

    biased_z = abs_z * (adj_probs.unsqueeze(1) * k)
    competition_weights = F.softmax(biased_z, dim=0)

    weighted_contributions = target_embs * competition_weights
    fluid_spikes = torch.sum(weighted_contributions, dim=0)

    soft_centroid = torch.sum(target_embs * adj_probs.unsqueeze(1), dim=0)

    max_z, _ = torch.max(abs_z, dim=0)
    fluidity_gate = torch.sigmoid(max_z - 2.0)

    final_unit = (fluidity_gate * fluid_spikes) + ((1 - fluidity_gate) * soft_centroid)
    return F.normalize(final_unit.unsqueeze(0), p=2, dim=1) * magnitude_target


def forge_mass_equalized_chimera(
    target_ids, dict_norm, mean, std, magnitude_target, adj_probs
):
    """
    The Mass-Equalized Genetic Chimera.
    Distributes probability as a strict dimensional budget.
    """
    # target_embs shape: (K, 4096)
    target_embs = dict_norm[target_ids]

    # 1. Compute Z-scores relative to the full dictionary
    z_scores = (target_embs - mean) / std
    abs_z = torch.abs(z_scores)

    # 2. Softmax per dimension across targets
    # For every column (dimension), the weights across the K targets sum to 1.0.
    # This finds the "winners" of each specific dimension based purely on anomaly strength, not probability.
    dim_weights = F.softmax(abs_z, dim=0)

    # 3. Normalize per target to sum to their allocated probability budget
    # Sum the weights across all 4096 dimensions for each target. Shape: (K, 1)
    target_mass = torch.sum(dim_weights, dim=1, keepdim=True)

    # Scale each target's dimensional matrix so the sum across dimensions equals its adj_prob.
    # If adj_prob is 0.90, the sum of target_p[0] is exactly 0.90.
    target_p = dim_weights * (adj_probs.unsqueeze(1) / (target_mass + 1e-9))

    # 4. Create the Chimera Embedding
    # Element-wise multiplication of the budget matrix and the embeddings, then squashed into a single vector.
    chimera_raw = torch.sum(target_p * target_embs, dim=0, keepdim=True)

    # 5. Re-tether to the physical shell
    return F.normalize(chimera_raw, p=2, dim=1) * magnitude_target
