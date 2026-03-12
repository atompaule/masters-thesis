import torch
import torch.nn.functional as F
import torch.optim as optim


def fast_geometric_solver(
    target_embs_norm,
    target_ids,
    dict_norm,
    soft_centroid_norm,
    magnitude_target,
    adj_probs,
):
    k = len(target_ids)
    device = dict_norm.device

    # 1. Establish the Quarantine Zone
    with torch.no_grad():
        initial_sims = torch.matmul(soft_centroid_norm, dict_norm.T).squeeze(0)
        _, danger_zone_idxs = torch.topk(initial_sims, 2000)
        danger_mask = ~torch.isin(
            danger_zone_idxs, torch.tensor(target_ids, device=device)
        )
        local_interlopers = dict_norm[danger_zone_idxs[danger_mask]]

    # 2. Deploy the Probe
    probe = torch.nn.Parameter(soft_centroid_norm.clone().squeeze(0))
    optimizer = optim.Adam([probe], lr=0.05)

    # 3. The Gradient War
    for _ in range(150):
        optimizer.zero_grad()
        probe_n = F.normalize(probe.unsqueeze(0), p=2, dim=1)

        # Pull Loss (Anchored to Tempered Probabilities)
        t_sims = torch.matmul(probe_n, target_embs_norm.T).squeeze(0)
        pull_loss = -torch.sum(t_sims * adj_probs)

        # Push Loss (Carving the Moat)
        i_sims = torch.matmul(probe_n, local_interlopers.T).squeeze(0)
        hard_negs, _ = torch.topk(i_sims, k)
        push_loss = F.relu(hard_negs.mean() - t_sims.min() + 0.1) * 2.5

        loss = pull_loss + push_loss
        loss.backward()
        optimizer.step()

    return F.normalize(probe.unsqueeze(0), p=2, dim=1).detach() * magnitude_target
