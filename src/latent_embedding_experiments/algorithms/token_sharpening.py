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

    theta = torch.arccos(torch.tensor(target_sim, dtype=e.dtype, device=e.device))
    result = torch.cos(theta) * ref_hat + torch.sin(theta) * noise

    return mag * F.normalize(result.unsqueeze(0), dim=1).squeeze(0)


def noisy_discrete_bernoulli(
    e: torch.Tensor,
    target_sim: float = 0.90,
    reference: torch.Tensor | None = None,
    keep_prob: float = 0.7,
    tol: float = 1e-4,
) -> torch.Tensor:
    """Perturb a token embedding via Bernoulli dropout noise to achieve target_sim.

    Instead of isotropic Gaussian noise, the perturbation direction is drawn
    by randomly zeroing out dimensions of the reference vector and taking the
    residual perpendicular component. This creates sparse, axis-aligned
    perturbations — structurally different from isotropic noise.

    The result is slerped to cosine similarity target_sim to the reference,
    at the original magnitude of e.

    Args:
        e:           Input embedding [d], need not be unit norm.
        target_sim:  Desired cosine similarity to the reference vector.
        reference:   Reference vector [d]. Defaults to e itself.
        keep_prob:   Probability of keeping each dimension (Bernoulli p).
                     Lower = sparser perturbation direction.
        tol:         Tolerance below which target_sim is treated as 1.0.

    Returns:
        Vector [d] with cosine similarity ~target_sim to reference, magnitude ||e||.
    """
    mag = e.norm(p=2).clamp_min(1e-8)
    e_hat = e / mag

    ref_hat = (
        F.normalize(reference.unsqueeze(0), dim=1).squeeze(0)
        if reference is not None
        else e_hat
    )

    if target_sim >= 1.0 - tol:
        return mag * ref_hat

    # Bernoulli mask: randomly zero out dimensions, then project out ref_hat component
    # to get a perpendicular noise direction. Retry if the mask produces a zero vector.
    d = e.shape[0]
    for _ in range(10):  # retries in case of degenerate mask
        mask = torch.bernoulli(torch.full((d,), keep_prob, dtype=e.dtype, device=e.device))
        noise = ref_hat * mask  # sparse version of ref_hat
        noise = noise - (noise @ ref_hat) * ref_hat  # project out ref_hat component
        noise_norm = noise.norm(p=2)
        if noise_norm > 1e-6:
            break
    else:
        # Fallback: isotropic Gaussian (degenerate mask — extremely rare in high-d)
        noise = torch.randn_like(ref_hat)
        noise = noise - (noise @ ref_hat) * ref_hat

    noise = F.normalize(noise.unsqueeze(0), dim=1).squeeze(0)

    theta = torch.arccos(torch.tensor(target_sim, dtype=e.dtype, device=e.device))
    result = torch.cos(theta) * ref_hat + torch.sin(theta) * noise

    return mag * F.normalize(result.unsqueeze(0), dim=1).squeeze(0)


def noisy_discrete_pc_deterministic(
    e: torch.Tensor,
    embedding_matrix: torch.Tensor,
    target_sim: float = 0.90,
    n_components: int = 50,
    _pc_cache: dict | None = None,
) -> torch.Tensor:
    """Blend a token embedding toward its own top-PC projection at an exact target similarity.

    The blend direction is deterministic and token-specific: it is the component
    of e that lies in the top-n_components PC subspace of the embedding matrix —
    the 'generic' directions that all token embeddings vary along most. Moving e
    toward this direction makes it less individuated and more prototypical.

    Unlike the random-noise approaches, the perturbation direction is not drawn
    randomly but is the PC shadow of e itself, making it a function of the token.

    Args:
        e:                Input token embedding [d], need not be unit norm.
        embedding_matrix: Full token embedding matrix [V, d], used for PCA.
        target_sim:       Desired cosine similarity to e after blending.
        n_components:     Number of top PCs to project onto (default 50).
        _pc_cache:        Optional dict for caching the SVD result across calls.

    Returns:
        Vector [d] with magnitude ||e||, blended toward the PC shadow of e.
    """
    mag = e.norm(p=2).clamp_min(1e-8)
    e_hat = e / mag

    if target_sim >= 1.0 - 1e-4:
        return e.clone()

    # --- PCA: compute or retrieve top principal components ---
    cache_key = ("top_pcs", n_components)
    if _pc_cache is not None and cache_key in _pc_cache:
        top_pcs = _pc_cache[cache_key]
    else:
        E = embedding_matrix.float()
        E_centered = E - E.mean(dim=0, keepdim=True)
        _, _, Vt = torch.linalg.svd(E_centered, full_matrices=False)
        top_pcs = Vt[:n_components]  # [n_components, d]
        if _pc_cache is not None:
            _pc_cache[cache_key] = top_pcs

    top_pcs_dev = top_pcs.to(dtype=e.dtype, device=e.device)

    # --- Project e onto the top-PC subspace (the "PC shadow" of this token) ---
    coords = top_pcs_dev @ e_hat        # [n_components]: how much of each PC is in e
    e_pc = top_pcs_dev.T @ coords       # [d]: e's projection onto the PC subspace

    # --- Make e_pc perpendicular to e_hat so slerp is well-defined ---
    direction = e_pc - (e_pc @ e_hat) * e_hat
    dir_norm = direction.norm(p=2)
    if dir_norm < 1e-6:
        # e already lies fully in the top-PC subspace; fall back to isotropic noise
        noise = torch.randn_like(e_hat)
        direction = noise - (noise @ e_hat) * e_hat
        direction = direction / direction.norm(p=2).clamp_min(1e-8)
    else:
        direction = direction / dir_norm

    # --- Slerp from e_hat toward the PC shadow direction by target_sim ---
    theta = torch.arccos(torch.tensor(target_sim, dtype=e.dtype, device=e.device))
    result = torch.cos(theta) * e_hat + torch.sin(theta) * direction

    return mag * F.normalize(result.unsqueeze(0), dim=1).squeeze(0)


def noisy_discrete_pc_random(
    e: torch.Tensor,
    embedding_matrix: torch.Tensor,
    target_sim: float = 0.90,
    reference: torch.Tensor | None = None,
    n_components: int = 50,
    use_top: bool = True,
    tol: float = 1e-4,
    _pc_cache: dict | None = None,
) -> torch.Tensor:
    """Perturb a token embedding with noise confined to the top (or bottom) PCs of E.

    The perturbation direction is drawn uniformly from the subspace spanned by
    the top-n_components principal components of the embedding matrix, then
    projected to be perpendicular to the reference vector.

    use_top=True  → noise in high-variance subspace (semantically loaded axes).
    use_top=False → noise in the complementary low-variance subspace (background).

    Comparing isotropic / top-PC / bottom-PC noise isolates whether perturbation
    direction relative to the embedding space's PCA structure matters.

    Args:
        e:                 Input embedding [d], need not be unit norm.
        embedding_matrix:  Full token embedding matrix [V, d], used for PCA.
        target_sim:        Desired cosine similarity to the reference vector.
        reference:         Reference vector [d]. Defaults to e itself.
        n_components:      Number of PCs to use (or exclude if use_top=False).
        use_top:           If True, noise ∈ span(top PCs). If False, noise ∈ complement.
        tol:               Tolerance below which target_sim is treated as 1.0.
        _pc_cache:         Optional dict for caching SVD results across calls.
                           Pass the same dict repeatedly to avoid recomputing.
                           Key used: ('top_pcs', n_components).

    Returns:
        Vector [d] with cosine similarity ~target_sim to reference, magnitude ||e||.

    Note on PCA cost:
        SVD on a large embedding matrix (e.g. [128k, 4096]) is expensive.
        Use _pc_cache to amortize across many calls:

            cache = {}
            for e in embeddings:
                result = noisy_discrete_pc_random(e, E, _pc_cache=cache)
    """
    mag = e.norm(p=2).clamp_min(1e-8)
    e_hat = e / mag

    ref_hat = (
        F.normalize(reference.unsqueeze(0), dim=1).squeeze(0)
        if reference is not None
        else e_hat
    )

    if target_sim >= 1.0 - tol:
        return mag * ref_hat

    # --- PCA: compute or retrieve top principal components ---
    cache_key = ('top_pcs', n_components)
    if _pc_cache is not None and cache_key in _pc_cache:
        top_pcs = _pc_cache[cache_key]  # [n_components, d]
    else:
        E = embedding_matrix.float()
        E_centered = E - E.mean(dim=0, keepdim=True)
        # Thin SVD: we only need the right singular vectors (principal components)
        # Using torch.linalg.svd with full_matrices=False for efficiency
        _, _, Vt = torch.linalg.svd(E_centered, full_matrices=False)
        top_pcs = Vt[:n_components]  # [n_components, d], rows are PC directions
        if _pc_cache is not None:
            _pc_cache[cache_key] = top_pcs

    # --- Draw noise direction from the chosen subspace ---
    if use_top:
        # Random linear combination of top PCs (uniform on their span)
        coeffs = torch.randn(n_components, dtype=e.dtype, device=e.device)
        noise = (coeffs.unsqueeze(1) * top_pcs.to(dtype=e.dtype, device=e.device)).sum(dim=0)  # [d]
    else:
        # Random isotropic vector, then project OUT the top-PC subspace
        # Leaves only the complement (bottom PCs / residual) subspace
        noise = torch.randn_like(ref_hat)
        proj_onto_top = top_pcs.to(dtype=e.dtype, device=e.device)  # [n_components, d]
        # Subtract projections onto each top PC
        coords = proj_onto_top @ noise  # [n_components]
        noise = noise - (proj_onto_top.T @ coords)  # [d]

    # Project out the ref_hat component so noise ⊥ ref_hat
    noise = noise - (noise @ ref_hat) * ref_hat
    noise_norm = noise.norm(p=2)
    if noise_norm < 1e-6:
        # Degenerate: ref_hat is already in the chosen subspace; resample isotropically
        noise = torch.randn_like(ref_hat)
        noise = noise - (noise @ ref_hat) * ref_hat
    noise = F.normalize(noise.unsqueeze(0), dim=1).squeeze(0)

    theta = torch.arccos(torch.tensor(target_sim, dtype=e.dtype, device=e.device))
    result = torch.cos(theta) * ref_hat + torch.sin(theta) * noise

    return mag * F.normalize(result.unsqueeze(0), dim=1).squeeze(0)