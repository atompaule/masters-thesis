import torch
import torch.nn.functional as F
import torch.optim as optim


def fast_geometric_solver(
    target_embs_norm,
    target_ids,
    dict_norm,
    soft_centroid_norm,  # We keep this in the signature so main.py doesn't break, but we ignore it!
    magnitude_target,
    adj_probs,
):
    k = len(target_ids)
    device = dict_norm.device

    # --- THE COLAR INITIALIZATION ---
    # Instead of starting at the heavily biased soft centroid, we start at the perfect geometric middle.
    # Summing the unit vectors of the targets and normalizing gives us the exact CoLaR direction.
    colar_dir = F.normalize(
        torch.sum(target_embs_norm, dim=0, keepdim=True), p=2, dim=1
    )

    # 1. Establish the Quarantine Zone from the CoLaR center
    with torch.no_grad():
        initial_sims = torch.matmul(colar_dir, dict_norm.T).squeeze(0)
        _, danger_zone_idxs = torch.topk(initial_sims, 2000)
        danger_mask = ~torch.isin(
            danger_zone_idxs, torch.tensor(target_ids, device=device)
        )
        local_interlopers = dict_norm[danger_zone_idxs[danger_mask]]

    # 2. Deploy the Probe exactly at the CoLaR coordinate
    probe = torch.nn.Parameter(colar_dir.clone().squeeze(0))
    optimizer = optim.Adam([probe], lr=0.05)

    # 3. The Gradient War
    for _ in range(2000):
        optimizer.zero_grad()
        probe_n = F.normalize(probe.unsqueeze(0), p=2, dim=1)

        # --- A. The Gravity ---
        t_sims = torch.matmul(probe_n, target_embs_norm.T).squeeze(0)
        # Boost the raw gravitational pull to keep absolute similarities high
        pull_loss = -torch.sum(t_sims * adj_probs) * 2.0

        # --- B. The Hierarchical Ranking Loss ---
        sim_diffs = t_sims[1:] - t_sims[:-1]
        p_gaps = adj_probs[:-1] - adj_probs[1:]

        # We only need a *relative* ordinal gap to maintain rank, not a massive absolute canyon
        margins = p_gaps * 0.1
        ranking_loss = torch.sum(F.relu(sim_diffs + margins))

        # --- C. The Moat ---
        i_sims = torch.matmul(probe_n, local_interlopers.T).squeeze(0)
        hard_negs, _ = torch.topk(i_sims, k)

        # Slightly relax the moat constraint so the probe doesn't flee into the void
        push_loss = F.relu(hard_negs.mean() - t_sims.min() + 0.05) * 1.5

        # Rebalance the forces. The ranking loss still holds authority, but gravity fights back.
        loss = pull_loss + (ranking_loss * 3.0) + push_loss
        loss.backward()
        optimizer.step()

    return F.normalize(probe.unsqueeze(0), p=2, dim=1).detach() * magnitude_target
