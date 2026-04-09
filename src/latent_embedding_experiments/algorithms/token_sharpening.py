import torch
import torch.nn.functional as F


def clean_subspace_proj_slerp(
    e: torch.Tensor,
    interlopers: torch.Tensor,
    target_sim: float = 0.90,
    **_,
) -> torch.Tensor:
    """Project out interloper subspace, then slerp back toward the original token.

    After projection, slerp interpolates on the unit sphere between the pure
    projection result (e_tilde) and the original token (e_hat). The parameter
    target_sim directly specifies the desired cosine similarity to the original
    token — no arbitrary beta, geometrically exact.

    alpha=0 → pure projection result (zero interloper sim, lowest self-sim).
    alpha=1 → original token (full self-sim, full interloper sim restored).

    The alpha that achieves target_sim is solved analytically:
        alpha = 1 - arccos(target_sim) / phi
    where phi = arccos(e_tilde · e_hat) is the angle between the two endpoints.
    """
    A = interlopers.T
    Q, _ = torch.linalg.qr(A)
    projection = Q @ (Q.T @ e)
    e_tilde = e - projection  # raw residual [d]

    e_hat = F.normalize(e.unsqueeze(0), dim=1).squeeze(0)  # original, normalized
    e_t_hat = F.normalize(e_tilde.unsqueeze(0), dim=1).squeeze(
        0
    )  # projection result, normalized

    cos_phi = (e_t_hat @ e_hat).clamp(-1.0, 1.0)
    phi = torch.arccos(cos_phi)  # angle between endpoints

    if phi < 1e-6:  # already nearly identical — projection changed nothing
        return e_t_hat

    # Solve for alpha such that slerp(alpha) has cosine similarity = target_sim to e_hat
    alpha = (
        1.0
        - torch.arccos(torch.tensor(target_sim, dtype=phi.dtype)).item() / phi.item()
    )
    alpha = max(
        0.0, min(1.0, alpha)
    )  # clamp: if target_sim > cos_phi, use pure projection

    v = torch.sin((1.0 - alpha) * phi) * e_t_hat + torch.sin(alpha * phi) * e_hat
    return F.normalize(v.unsqueeze(0), dim=1).squeeze(0)
