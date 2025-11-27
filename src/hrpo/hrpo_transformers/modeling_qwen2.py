import torch
from torch import nn
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM, Qwen2Model

from .utils import HRPOGenerationMixin


class ThinkingResidualLambda(nn.Module):
    c = 8.0

    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.Lambda = nn.Parameter(torch.randn(config.hidden_size))

    def reset_lambda_parameters(self, r_min=0.9, r_max=0.999):
        with torch.no_grad():
            nn.init.uniform_(self.Lambda, a=r_min, b=r_max)
            self.Lambda.data.copy_(-torch.log((self.Lambda ** (-1.0 / self.c)) - 1))

    def forward(self, r_t):
        a_t = torch.exp(
            -self.c * nn.functional.softplus(-self.Lambda, beta=1, threshold=20) * r_t
        )
        return a_t


class HRPOQwen2Model(Qwen2Model):
    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.thinking_residual_gate_r = nn.Linear(
            config.hidden_size, config.hidden_size
        )
        self.thinking_residual_gate_i = nn.Linear(
            config.hidden_size, config.hidden_size
        )
        self.thinking_residual_Lambda = ThinkingResidualLambda(config)
        self.post_init()

    def thinking_residual(self, embeds, residual, eps=1e-8):
        """Computes the next input embedding as a function of the current discrete embedding and the residuals."""
        r_t = torch.sigmoid(self.thinking_residual_gate_r(embeds))
        i_t = torch.sigmoid(self.thinking_residual_gate_i(embeds))
        a_t = self.thinking_residual_Lambda(r_t)
        return a_t * embeds + torch.sqrt(1 - a_t.pow(2) + eps) * (i_t * residual), a_t


class HRPOQwen2ForCausalLM(Qwen2ForCausalLM, HRPOGenerationMixin):
    def __init__(self, config):
        super().__init__(config)
        self.model = HRPOQwen2Model(config)
        self.post_init()
