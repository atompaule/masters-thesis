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
        lambda_param = model.model.latent_gate_a.Lambda
        print(f"Lambda mean: {lambda_param.mean().item():.4f}")
        print(f"Lambda std: {lambda_param.std().item():.4f}")
        print(f"Lambda min: {lambda_param.min().item():.4f}")
        print(f"Lambda max: {lambda_param.max().item():.4f}")

        # Store for comparison
        pre_lambda = lambda_param.detach().cpu().clone()
        pre_gate_r_weight = model.model.latent_gate_r.weight.detach().cpu().clone()

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
    if hasattr(model.base_model.model.model, "latent_gate_a"):
        latent_gate_a = model.base_model.model.model.latent_gate_a
        latent_gate_r = model.base_model.model.model.latent_gate_r
    elif hasattr(model.base_model.model, "latent_gate_a"):
        latent_gate_a = model.base_model.model.latent_gate_a
        latent_gate_r = model.base_model.model.latent_gate_r
    else:
        print("Could not find latent_gate_a in expected location.")
        print(model)
        return

    lambda_param = latent_gate_a.Lambda
    print(f"Lambda mean: {lambda_param.mean().item():.4f}")
    print(f"Lambda std: {lambda_param.std().item():.4f}")
    print(f"Lambda min: {lambda_param.min().item():.4f}")
    print(f"Lambda max: {lambda_param.max().item():.4f}")

    # Compare
    vector_diff_lambda = (lambda_param.detach().cpu() - pre_lambda).abs()
    diff_lambda = vector_diff_lambda.mean()
    print(f"\nMean absolute difference in Lambda: {diff_lambda.item():.6f}")

    diff_gate_r = (latent_gate_r.weight.detach().cpu() - pre_gate_r_weight).abs().mean()
    print(f"Mean absolute difference in Gate R weights: {diff_gate_r.item():.6f}")

    if diff_lambda.item() > 0 or diff_gate_r.item() > 0:
        print(
            "\nSUCCESS: Parameters have changed from initialization, confirming they were loaded from adapter."
        )
    else:
        print(
            "\nWARNING: Parameters appear unchanged. They might not have been loaded correctly or trained."
        )

    # Visualize Lambda distribution
    plt.figure(figsize=(10, 6))
    plt.hist(
        lambda_param.float().detach().cpu().numpy().flatten(),
        bins=50,
        alpha=0.7,
        label="Trained Lambda",
    )
    plt.hist(
        pre_lambda.float().numpy().flatten(),
        bins=50,
        alpha=0.5,
        label="Random Init Lambda",
    )
    plt.title("Distribution of Lambda Parameter")
    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.legend()
    plt.savefig("lambda_distribution.png")
    print("\nSaved lambda distribution plot to lambda_distribution.png")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python inspect_latent_gates.py <base_model_path> <adapter_path>")
        sys.exit(1)

    base_model = sys.argv[1]
    adapter = sys.argv[2]
    inspect_latent_gates(base_model, adapter)
