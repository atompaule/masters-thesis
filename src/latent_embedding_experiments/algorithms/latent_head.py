import torch
import torch.nn as nn


class LatentHead(nn.Module):
    """Two-layer MLP from hidden-state space into the embedding space."""

    def __init__(self, hidden_dim: int, intermediate_dim: int = 0):
        super().__init__()
        if intermediate_dim <= 0:
            intermediate_dim = hidden_dim * 2
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim, bias=False),
            nn.SiLU(),
            nn.Linear(intermediate_dim, intermediate_dim, bias=False),
            nn.SiLU(),
            nn.Linear(intermediate_dim, hidden_dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


def load_latent_head(
    checkpoint_path: str, hidden_dim: int, device: torch.device
) -> LatentHead:
    head = LatentHead(hidden_dim=hidden_dim)
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    head.load_state_dict(state)
    head.to(device=device, dtype=torch.float32)
    head.eval()
    print(f"[LatentHead] Loaded from {checkpoint_path}")
    return head
