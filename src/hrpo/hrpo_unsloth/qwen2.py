from src.hrpo.hrpo_transformers.modeling_qwen2 import (  # LlamaAttention,; LlamaDecoderLayer,; LlamaForCausalLM, <-- TODO: double check those
    HRPOQwen2Model,
)
from src.hrpo.hrpo_unsloth.llama import LlamaModel_fast_forward

HRPOQwen2Model.forward = LlamaModel_fast_forward
