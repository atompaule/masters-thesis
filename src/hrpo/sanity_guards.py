import os

import torch
from safetensors.torch import load_file


def check_trainable_status(model):
    print("--- Trainable Parameters Audit ---")
    trainable_count = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"✅ Trainable: {name} | shape: {param.shape}")
            trainable_count += 1

    if trainable_count == 0:
        print("💀 Nothing is trainable. You are burning electricity for fun.")
    else:
        print(f"\nTotal trainable tensors: {trainable_count}")


def check_adapter_sanity(adapter_path):
    print(f"--- Autopsy of {adapter_path} ---")

    # Handle both .bin and .safetensors because consistency is a myth
    bin_path = os.path.join(adapter_path, "adapter_model.bin")
    safe_path = os.path.join(adapter_path, "adapter_model.safetensors")

    if os.path.exists(safe_path):
        state_dict = load_file(safe_path)
    elif os.path.exists(bin_path):
        state_dict = torch.load(bin_path, map_location="cpu")
    else:
        print("💀 No adapter file found. Did you even save?")
        state_dict = {}

    for key in state_dict.keys():
        if "latent_gate" in key:
            print(f"Found key: {key}")
