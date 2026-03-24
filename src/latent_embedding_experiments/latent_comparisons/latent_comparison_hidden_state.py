import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer

# =========================================================
# CONFIGURATION
# =========================================================


@dataclass
class Config:
    model_id: str = "meta-llama/Llama-3.2-3B-Instruct"
    temperature: float = 2.0
    k: int = 10
    solver_steps: int = 300
    lr: float = 0.05
    danger_topk: int = 2000
    log_file: str = (
        "src/latent_embedding_experiments/logs/llama_3b_latent_comparison_hidden_state.txt"
    )


CFG = Config()

# =========================================================
# UTILITIES
# =========================================================


def emit(text, f=None):
    print(text)
    if f:
        f.write(text + "\n")


# =========================================================
# INITIALIZATION
# =========================================================

emit(f"Waking the oracle: {CFG.model_id}...")
tokenizer = AutoTokenizer.from_pretrained(CFG.model_id)
model = AutoModelForCausalLM.from_pretrained(
    CFG.model_id, device_map="auto", torch_dtype=torch.bfloat16
)
device = model.device

raw_embeddings = model.get_input_embeddings().weight.detach().to(torch.float32)
norm_dictionary = F.normalize(raw_embeddings, dim=1)

# =========================================================
# THE GEOMETRIC SOLVER
# =========================================================


def fast_geometric_solver(target_norm, target_ids, dict_norm, magnitude, adj_probs):
    k = len(target_ids)
    base = F.normalize(target_norm.sum(dim=0, keepdim=True), dim=1)

    with torch.no_grad():
        sims = base @ dict_norm.T
        _, idxs = torch.topk(sims.squeeze(0), CFG.danger_topk)
        mask = ~torch.isin(idxs, torch.tensor(target_ids, device=device))
        interlopers = dict_norm[idxs[mask]]

    probe = torch.nn.Parameter(base.squeeze(0))
    opt = optim.Adam([probe], lr=CFG.lr)

    for _ in range(CFG.solver_steps):
        opt.zero_grad()
        p = F.normalize(probe.unsqueeze(0), dim=1)

        # 1. The Anchor (Magnetic Pull)
        sims_t = (p @ target_norm.T).squeeze(0)
        pull = -torch.sum(sims_t * adj_probs) * 2.0

        # 2. The Topological Mirror (Relative Spacing)
        # Forcing the geometric intervals to perfectly mimic the probability intervals
        p_centered = adj_probs - adj_probs.mean()
        s_centered = sims_t - sims_t.mean()

        cov = torch.sum(p_centered * s_centered)
        var_p = torch.sqrt(torch.sum(p_centered**2) + 1e-8)
        var_s = torch.sqrt(torch.sum(s_centered**2) + 1e-8)

        corr = cov / (var_p * var_s)

        # We penalize any deviation from a perfect 1.0 correlation
        topology_loss = (1.0 - corr) * 4.0

        # 3. The Repulsion (Interloper Moat)
        sims_i = (p @ interlopers.T).squeeze(0)
        hard, _ = torch.topk(sims_i, k)
        push = F.relu(hard.mean() - sims_t.min() + 0.05) * 1.5

        loss = pull + topology_loss + push
        loss.backward()
        opt.step()

    return F.normalize(probe.unsqueeze(0), dim=1) * magnitude


# =========================================================
# THE EXPERIMENT
# =========================================================

PROMPTS = [
    "The fundamental difference between human consciousness and artificial intelligence is",
    "Deep in the heart of the overgrown ruins, the air tasted like",
    "If you want to truly understand quantum mechanics, you must first",
]

with open(CFG.log_file, "w", encoding="utf-8") as f:
    emit("🌌 THE SUBCONSCIOUS ALIGNMENT LAB 🌌", f)
    emit(
        "Measuring the geometric fidelity between synthetic forges and the true hidden state.\n",
        f,
    )

    for prompt in PROMPTS:
        emit("\n" + "=" * 100, f)
        emit(f"📜 PROMPT: '{prompt}'", f)
        emit("=" * 100, f)

        # 1. Forward Pass (Capturing the Ghost)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, output_hidden_states=True)

            # The True Neural Echo: The final continuous state before the LM head
            true_hidden_state = (
                outputs.hidden_states[-1][0, -1, :].to(torch.float32).unsqueeze(0)
            )
            true_hidden_unit = F.normalize(true_hidden_state, p=2, dim=1)

            # 2. Extracting the Logits
            next_token_logits = outputs.logits[0, -1, :].to(torch.float32)
            probs = F.softmax(next_token_logits, dim=-1)

            top_k_probs, top_k_ids = torch.topk(probs, CFG.k)
            top_k_words = tokenizer.batch_decode(top_k_ids)

            # Thermodynamics
            adj_probs = top_k_probs ** (1.0 / CFG.temperature)
            adj_probs = adj_probs / adj_probs.sum()

            emit("\n[Predicted Horizon]", f)
            for i, (w, p) in enumerate(zip(top_k_words, top_k_probs)):
                cw = w.replace("\n", "\\n").replace("\r", "\\r")
                emit(f"  {i+1}. {cw:<15} | Confidence: {p.item()*100:5.2f}%", f)

            # 3. Forging the Synthetic Vectors
            top_k_raw = raw_embeddings[top_k_ids]
            top_k_norm = norm_dictionary[top_k_ids]
            magnitude = torch.norm(top_k_raw, dim=1).mean().item()

            v_colar = top_k_raw.sum(dim=0, keepdim=True) / math.sqrt(CFG.k)
            v_soft = (top_k_raw * adj_probs.unsqueeze(1)).sum(dim=0, keepdim=True)

            with torch.enable_grad():
                v_solver = fast_geometric_solver(
                    top_k_norm, top_k_ids, norm_dictionary, magnitude, adj_probs
                )

            # Baseline: The single top predicted token
            v_top1 = top_k_raw[0:1]

            # 4. Normalization for Angle Comparison
            unit_colar = F.normalize(v_colar, p=2, dim=1)
            unit_soft = F.normalize(v_soft, p=2, dim=1)
            unit_solver = F.normalize(v_solver, p=2, dim=1)
            unit_top1 = F.normalize(v_top1, p=2, dim=1)

            # 5. The Cosine Friction
            sim_colar = (unit_colar @ true_hidden_unit.T).item()
            sim_soft = (unit_soft @ true_hidden_unit.T).item()
            sim_solver = (unit_solver @ true_hidden_unit.T).item()
            sim_top1 = (unit_top1 @ true_hidden_unit.T).item()

            emit(f"\n[Geometric Alignment to True Hidden State]", f)
            emit(f"  {'Top-1 Token Baseline':<20} : {sim_top1:.4f}", f)
            emit(f"  {'CoLaR Geometry':<20} : {sim_colar:.4f}", f)
            emit(f"  {'Soft Thinking':<20} : {sim_soft:.4f}", f)
            emit(f"  {'Geometric Solver':<20} : {sim_solver:.4f}", f)

    emit("\nAlignment sequence complete.", f)
print(f"\nResults securely logged to → {CFG.log_file}")
