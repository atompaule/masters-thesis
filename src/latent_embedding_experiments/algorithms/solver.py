from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torch.optim as optim


@dataclass
class Config:
    solver_steps: int = 300
    lr: float = 0.05
    danger_topk: int = 2000

    ranking_weight: float = 3.0
    
    # We keep a tiny baseline floor, but now add a dynamic scaler.
    # The margin will scale relative to the difference in probabilities.
    ranking_margin_base: float = 1e-4
    ranking_margin_scale: float = 0.5 

    pull_weight: float = 1.5
    push_weight: float = 2.2


CFG = Config()


def fast_geometric_solver(
    target_norm, target_ids, dict_norm, magnitude, pool_logits, temperature=1.0
):
    k = len(target_ids)
    device = dict_norm.device
    base = F.normalize(target_norm.sum(dim=0, keepdim=True), dim=1)

    with torch.no_grad():
        sims = base @ dict_norm.T
        _, idxs = torch.topk(sims.squeeze(0), CFG.danger_topk)
        mask = ~torch.isin(idxs, torch.tensor(target_ids, device=device))
        interlopers = dict_norm[idxs[mask]]

    probe = torch.nn.Parameter(base.squeeze(0))
    opt = optim.Adam([probe], lr=CFG.lr)

    temp_probs = F.softmax(pool_logits / temperature, dim=-1)

    # Pairwise probability structure
    with torch.no_grad():
        p_diff = temp_probs.unsqueeze(1) - temp_probs.unsqueeze(0)  # [k, k]

        # p_i > p_j => target sim_i > sim_j
        pos_mask = p_diff > 0
        
        # Here is the magic: the margin is now a topographical map of the prob dist.
        # It expands where the probability gap is vast, and shrinks where it's tight.
        dynamic_margins = CFG.ranking_margin_base + (p_diff * CFG.ranking_margin_scale)

    for _ in range(CFG.solver_steps):
        opt.zero_grad()
        p = F.normalize(probe.unsqueeze(0), dim=1)

        # 1. Anchor
        sims_t = (p @ target_norm.T).squeeze(0)
        pull = -torch.sum(sims_t * temp_probs) * CFG.pull_weight

        # 3. Pairwise similarity-difference matrix: sim_diffs[i,j] = sim_i - sim_j
        sim_diffs = sims_t.unsqueeze(1) - sims_t.unsqueeze(0)  # [k, k]

        # 3b. Elastic Ranking hinge
        pm = pos_mask.to(sim_diffs.dtype)
        ranking_loss = torch.sum(pm * F.relu(dynamic_margins - sim_diffs)) * CFG.ranking_weight

        # 4. Repulsion
        sims_i = (p @ interlopers.T).squeeze(0)
        hard, _ = torch.topk(sims_i, k)
        push = F.relu(hard.mean() - sims_t.min() + 0.05) * CFG.push_weight

        # loss = pull + topology_loss + pairwise_reg_loss + pairwise_gap_loss + push
        loss = ranking_loss + push + pull
        loss.backward()
        opt.step()

    return F.normalize(probe.unsqueeze(0), dim=1) * magnitude