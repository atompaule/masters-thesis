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
    # Compute the QR decomposition of matrix A
    # A = QR
    # Q will consist of col vectors that are all orthogonal to each other, computed one after the other
    # q_i = v_i * sum_{i=1}^{n-1} \frac{v_n^T * q_i}{|q_i|^2} * q_i
    # Q builds a basis in the same vector room that the vectors in A span
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


def noisy_target_sim(
    e: torch.Tensor,
    target_sim: float = 0.90,
    reference: torch.Tensor | None = None,
    tol: float = 1e-4,
) -> torch.Tensor:
    """Perturb a token embedding with random noise to achieve target_sim cosine similarity.

    By default, similarity is measured against e itself (i.e. the result is a
    noisy version of e at a fixed angular distance). If reference is provided,
    the result is instead slerped to have cosine similarity target_sim to
    reference, while preserving the magnitude of e.

    Args:
        e:           Input embedding [d], need not be unit norm.
        target_sim:  Desired cosine similarity to the reference vector.
        reference:   Reference vector [d] to target similarity against.
                     Defaults to e itself if None.
        tol:         Tolerance below which target_sim is treated as 1.0.

    Returns:
        Vector [d] with cosine similarity ~target_sim to reference, magnitude ||e||.
    """
    mag = e.norm(p=2).clamp_min(1e-8)
    e_hat = e / mag  # [d], unit norm

    ref_hat = (
        F.normalize(reference.unsqueeze(0), dim=1).squeeze(0)
        if reference is not None
        else e_hat
    )

    if target_sim >= 1.0 - tol:
        return mag * ref_hat

    noise = torch.randn_like(ref_hat)
    noise = noise - (noise @ ref_hat) * ref_hat
    noise = F.normalize(noise.unsqueeze(0), dim=1).squeeze(
        0
    )  # unit norm, perp to ref_hat

    theta = torch.arccos(torch.tensor(target_sim, dtype=e.dtype))
    result = torch.cos(theta) * ref_hat + torch.sin(theta) * noise

    return mag * F.normalize(result.unsqueeze(0), dim=1).squeeze(0)
