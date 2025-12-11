"""
Run this via:
python -m debugpy --connect n-hpc-login1:5678 --wait-for-client src/hrpo/inspect_latent_gates.py "Qwen/Qwen2.5-1.5B-Instruct" "/work/utsch/masters-thesis/experiments/Qwen2.5-1.5B-Instruct-gsm8k-group4-lora32-rmin0.99-temp0.5-hrpo-2025-12-10_16-54-49/final_model"
"""

import os
import sys

# Setup paths
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src", "external", "transformers", "src"))

import matplotlib.pyplot as plt
import numpy as np
import torch
from peft import PeftModel
from transformers import AutoTokenizer

from src.external.transformers.src.transformers.models.qwen2.modeling_qwen2 import (
    HRPOQwen2ForCausalLM,
)
from src.external.transformers.src.transformers.trainer_utils import set_seed


def inspect_latent_gates(model_path, adapter_path):
    print(f"Loading base model from {model_path}...")
    model = HRPOQwen2ForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
    )

    # Check values before loading adapter
    print("\n--- Before Adapter Load, r_min=0.99, r_max=0.999 ---")
    if hasattr(model.model, "latent_gate_a"):
        set_seed(42)
        model.model.latent_gate_a.reset_lambda_parameters(
            r_min=0.99,
            r_max=0.999,
        )
        pre_lambda_param = model.model.latent_gate_a.Lambda
        print(f"Lambda dtype: {pre_lambda_param.dtype}")
        print(f"Lambda mean: {pre_lambda_param.mean().item():.6e}")
        print(f"Lambda std: {pre_lambda_param.std().item():.6e}")
        print(f"Lambda min: {pre_lambda_param.min().item():.6e}")
        print(f"Lambda max: {pre_lambda_param.max().item():.6e}")

        pre_gate_r_param = model.model.latent_gate_r.weight
        print(f"Gate R weight dtype: {pre_gate_r_param.dtype}")
        print(f"Gate R weight mean: {pre_gate_r_param.mean().item():.6e}")
        print(f"Gate R weight std: {pre_gate_r_param.std().item():.6e}")
        print(f"Gate R weight min: {pre_gate_r_param.min().item():.6e}")
        print(f"Gate R weight max: {pre_gate_r_param.max().item():.6e}")

        pre_gate_i_param = model.model.latent_gate_i.weight
        print(f"Gate I weight dtype: {pre_gate_i_param.dtype}")
        print(f"Gate I weight mean: {pre_gate_i_param.mean().item():.6e}")
        print(f"Gate I weight std: {pre_gate_i_param.std().item():.6e}")
        print(f"Gate I weight min: {pre_gate_i_param.min().item():.6e}")
        print(f"Gate I weight max: {pre_gate_i_param.max().item():.6e}")

        # Store for comparison
        pre_lambda = pre_lambda_param.detach().cpu().clone()
        pre_gate_r_weight = pre_gate_r_param.detach().cpu().clone()
        pre_gate_i_weight = pre_gate_i_param.detach().cpu().clone()

    print(f"\nLoading adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(model, adapter_path)

    print("\n--- After Adapter Load (Trained) ---")
    # Access the base model inside PeftModel
    # PeftModel wraps the base model, so we need to access via model.base_model.model
    # Note: HRPOQwen2ForCausalLM has .model attribute which is HRPOQwen2Model
    # The structure with PEFT is usually: PeftModel -> Base (HRPOQwen2ForCausalLM) -> HRPOQwen2Model

    # Depending on how PEFT wraps it, sometimes it's model.base_model.model.model if it wraps the inner model directly
    # But usually PeftModel wraps the ForCausalLM.

    # Let's find where latent_gate_a is
    latent_gate_a = model.base_model.model.model.latent_gate_a
    latent_gate_r = model.base_model.model.model.latent_gate_r
    latent_gate_i = model.base_model.model.model.latent_gate_i

    lambda_param = latent_gate_a.Lambda
    print(f"Lambda dtype: {lambda_param.dtype}")
    print(f"Lambda mean: {lambda_param.mean().item():.6e}")
    print(f"Lambda std: {lambda_param.std().item():.6e}")
    print(f"Lambda min: {lambda_param.min().item():.6e}")
    print(f"Lambda max: {lambda_param.max().item():.6e}")

    gate_r_weight = latent_gate_r.weight
    print(f"Gate R weight dtype: {gate_r_weight.dtype}")
    print(f"Gate R weight mean: {gate_r_weight.mean().item():.6e}")
    print(f"Gate R weight std: {gate_r_weight.std().item():.6e}")
    print(f"Gate R weight min: {gate_r_weight.min().item():.6e}")
    print(f"Gate R weight max: {gate_r_weight.max().item():.6e}")

    gate_i_weight = latent_gate_i.weight
    print(f"Gate I weight dtype: {gate_i_weight.dtype}")
    print(f"Gate I weight mean: {gate_i_weight.mean().item():.6e}")
    print(f"Gate I weight std: {gate_i_weight.std().item():.6e}")
    print(f"Gate I weight min: {gate_i_weight.min().item():.6e}")
    print(f"Gate I weight max: {gate_i_weight.max().item():.6e}")

    # Compare
    diff_lambda = (lambda_param.detach().cpu() - pre_lambda).abs().mean()
    print(f"\nMean absolute difference in Lambda: {diff_lambda.item():.6e}")

    diff_gate_r = (gate_r_weight.detach().cpu() - pre_gate_r_weight).abs().mean()
    print(f"Mean absolute difference in Gate R weights: {diff_gate_r.item():.6e}")

    diff_gate_i = (gate_i_weight.detach().cpu() - pre_gate_i_weight).abs().mean()
    print(f"Mean absolute difference in Gate I weights: {diff_gate_i.item():.6e}")

    if diff_lambda.item() > 0 or diff_gate_r.item() > 0:
        print(
            "\nSUCCESS: Parameters have changed from initialization, confirming they were loaded from adapter."
        )
    else:
        print(
            "\nWARNING: Parameters appear unchanged. They might not have been loaded correctly or trained."
        )


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python inspect_latent_gates.py <base_model_path> <adapter_path>")
        sys.exit(1)

    base_model = sys.argv[1]
    adapter = sys.argv[2]
    inspect_latent_gates(base_model, adapter)
