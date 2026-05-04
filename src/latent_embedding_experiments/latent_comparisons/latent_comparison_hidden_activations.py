"""
Hidden activation comparison — soft embedding vs constituent discrete tokens,
across both the reference (base) model and a LoRA-finetuned checkpoint.

For a soft embedding built from k tokens, this script captures the hidden state
at the concept position for every transformer layer and computes, per layer:

  - cosine similarity between soft and each discrete token's hidden state
  - L2 distance between soft and each discrete token's hidden state
  - norm of soft hidden state vs norm of each discrete token's hidden state

Run for both ref and lora models, and additionally compares the two models
directly — showing how LoRA training shifted the hidden states for soft inputs
vs discrete inputs.
"""

import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.latent_embedding_experiments.algorithms.soft_thinking import soft_thinking
from src.latent_embedding_experiments.algorithms.utils import emit

# ── Config ────────────────────────────────────────────────────────────────────


@dataclass
class Config:
    model_id: str = "meta-llama/Llama-3.1-8B-Instruct"
    lora_checkpoint: str = (
        "/work/utsch/masters-thesis/latent_embedding_experiments/checkpoints/exp7_grpo_think/step_00500"
    )
    log_file: str = (
        "src/latent_embedding_experiments/logs/hidden_activation_comparison.txt"
    )


CFG = Config()


# ── Models ────────────────────────────────────────────────────────────────────

print(f"Loading base model: {CFG.model_id}...")
tokenizer = AutoTokenizer.from_pretrained(CFG.model_id)

ref_model = AutoModelForCausalLM.from_pretrained(
    CFG.model_id, device_map="auto", torch_dtype=torch.bfloat16
)
ref_model.eval()
device = next(ref_model.parameters()).device

print(f"Loading LoRA checkpoint: {CFG.lora_checkpoint}...")
lora_model = PeftModel.from_pretrained(
    AutoModelForCausalLM.from_pretrained(
        CFG.model_id, device_map="auto", torch_dtype=torch.bfloat16
    ),
    CFG.lora_checkpoint,
)
lora_model.eval()

# Use ref model's embedding matrix — both models share the same base embeddings
vocab_embs = ref_model.get_input_embeddings().weight.detach().float()

MODELS = {"ref": ref_model, "lora": lora_model}

print(f"Device: {device}")
print(f"Layers: {ref_model.config.num_hidden_layers}")


# ── Hook-based hidden state extractor ────────────────────────────────────────


def get_hidden_states(
    vec: torch.Tensor,  # [1, d] or [d] — the embedding to insert
    prefix: str,
    suffix: str,
    m: torch.nn.Module,
) -> list[torch.Tensor]:
    """
    Run a forward pass with `vec` spliced at the concept position
    (immediately after the prefix). Return a list of [d] hidden state
    tensors, one per transformer layer, at the concept position.
    """
    prefix_ids = tokenizer(prefix, return_tensors="pt").input_ids.to(device)
    suffix_ids = tokenizer(
        suffix, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(device)

    prefix_emb = m.get_input_embeddings()(prefix_ids)
    suffix_emb = m.get_input_embeddings()(suffix_ids)

    if vec.dim() == 1:
        vec_in = vec.unsqueeze(0).unsqueeze(0)  # [d] → [1, 1, d]
    elif vec.dim() == 2:
        vec_in = vec.unsqueeze(0)  # [1, d] → [1, 1, d]
    else:
        vec_in = vec  # already [1, 1, d]
    vec_in = vec_in.to(m.dtype)

    concept_pos = prefix_ids.shape[1]
    inputs_emb = torch.cat([prefix_emb, vec_in, suffix_emb], dim=1)

    layer_hidden: list[torch.Tensor] = []

    def make_hook(pos: int):
        def hook(module, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            layer_hidden.append(hs[0, pos, :].detach().float())

        return hook

    # PeftModel wraps the base model under .base_model.model; plain models expose
    # .model directly — normalise to whichever is present.
    inner = getattr(m, "base_model", m)
    inner = getattr(inner, "model", inner)

    # replace the inner/hooks block:
    def get_transformer_layers(m: torch.nn.Module):
        """Walk down base_model/model wrappers until we reach the layers list."""
        for attr in ("base_model", "model"):
            while hasattr(m, attr) and not hasattr(m, "layers"):
                m = getattr(m, attr)
        return m.layers

    hooks = [
        layer.register_forward_hook(make_hook(concept_pos))
        for layer in get_transformer_layers(m)
    ]
    with torch.no_grad():
        m(inputs_embeds=inputs_emb)
    for h in hooks:
        h.remove()

    return layer_hidden  # list of L tensors, each [d]


# ── Per-layer difference metrics ──────────────────────────────────────────────


def compare_hidden_states(
    soft_hidden: list[torch.Tensor],  # L x [d]
    discrete_hidden: list[torch.Tensor],  # L x [d]
) -> dict[str, list[float]]:
    """
    Per-layer comparison between two sequences of hidden states.

    Returns:
        cosine_sim  : cosine similarity at each layer
        l2_dist     : L2 distance at each layer
        soft_norm   : L2 norm of soft hidden state at each layer
        disc_norm   : L2 norm of discrete hidden state at each layer
        norm_diff   : |soft_norm - disc_norm| at each layer
    """
    cosine_sim, l2_dist, soft_norm, disc_norm, norm_diff = [], [], [], [], []

    for s, d in zip(soft_hidden, discrete_hidden):
        cos = F.cosine_similarity(s.unsqueeze(0), d.unsqueeze(0)).item()
        l2 = (s - d).norm(p=2).item()
        sn = s.norm(p=2).item()
        dn = d.norm(p=2).item()

        cosine_sim.append(cos)
        l2_dist.append(l2)
        soft_norm.append(sn)
        disc_norm.append(dn)
        norm_diff.append(abs(sn - dn))

    return {
        "cosine_sim": cosine_sim,
        "l2_dist": l2_dist,
        "soft_norm": soft_norm,
        "disc_norm": disc_norm,
        "norm_diff": norm_diff,
    }


# ── Reporting ─────────────────────────────────────────────────────────────────


def emit_comparison_table(
    token_names: list[str],
    metrics_per_token: dict[str, dict[str, list[float]]],
    n_layers: int,
    f,
    report_every: int = 4,
):
    """
    Print a per-layer table for cosine similarity and L2 distance,
    one column per discrete token.
    """
    report_layers = set(range(0, n_layers, report_every)) | {n_layers - 1}
    col_w = 9
    pad = "  "

    for metric_key, metric_label in [
        ("cosine_sim", "Cosine similarity  (soft vs discrete, per layer)"),
        ("l2_dist", "L2 distance        (soft vs discrete, per layer)"),
        ("norm_diff", "Norm difference    |‖soft‖ - ‖disc‖|, per layer"),
    ]:
        emit(f"\n{pad}{metric_label}", f)
        header = f"{pad}{'L':>3}  " + "  ".join(f"{n:>{col_w}}" for n in token_names)
        emit(header, f)
        emit(f"{pad}{'-'*3}  {'-'*(len(token_names) * (col_w + 2))}", f)

        for i in range(n_layers):
            if i not in report_layers:
                continue
            cells = "  ".join(
                f"{metrics_per_token[tok][metric_key][i]:>{col_w}.4f}"
                for tok in token_names
            )
            emit(f"{pad}{i:>3}  {cells}", f)

    # Summary: mean cosine similarity per token across all layers
    emit(f"\n{pad}Mean cosine similarity across all layers:", f)
    col = max(len(n) for n in token_names) + 1
    bar_max = 20
    for tok in token_names:
        vals = metrics_per_token[tok]["cosine_sim"]
        mean_cos = sum(vals) / len(vals)
        # final-layer cosine is often the most meaningful
        final_cos = vals[-1]
        bar = "█" * int(mean_cos * bar_max) + "░" * (bar_max - int(mean_cos * bar_max))
        emit(
            f"{pad}  {tok:<{col}}  mean={mean_cos:.4f}  final={final_cos:.4f}  {bar}", f
        )

    # Which token is most similar to the soft embedding, layer by layer
    emit(f"\n{pad}Most similar discrete token at each reported layer:", f)
    emit(f"{pad}{'L':>3}  {'Token':<{col}}  {'CosSim':>7}", f)
    emit(f"{pad}{'-'*3}  {'-'*col}  {'-'*7}", f)
    for i in range(n_layers):
        if i not in report_layers:
            continue
        best_tok = max(
            token_names,
            key=lambda tok: metrics_per_token[tok]["cosine_sim"][i],
        )
        best_val = metrics_per_token[best_tok]["cosine_sim"][i]
        emit(f"{pad}{i:>3}  {best_tok:<{col}}  {best_val:>7.4f}", f)


def emit_norm_profile(
    token_names: list[str],
    metrics_per_token: dict[str, dict[str, list[float]]],
    soft_hidden: list[torch.Tensor],
    n_layers: int,
    f,
    report_every: int = 4,
):
    """Print soft embedding norm alongside each discrete token's norm."""
    report_layers = set(range(0, n_layers, report_every)) | {n_layers - 1}
    col_w = 9
    pad = "  "

    soft_norms = [h.norm(p=2).item() for h in soft_hidden]

    emit(f"\n{pad}Hidden state norms (soft + each discrete token):", f)
    header = f"{pad}{'L':>3}  {'soft':>{col_w}}  " + "  ".join(
        f"{n:>{col_w}}" for n in token_names
    )
    emit(header, f)
    emit(f"{pad}{'-'*3}  {'-'*col_w}  {'-'*(len(token_names)*(col_w+2))}", f)

    for i in range(n_layers):
        if i not in report_layers:
            continue
        disc_cells = "  ".join(
            f"{metrics_per_token[tok]['disc_norm'][i]:>{col_w}.3f}"
            for tok in token_names
        )
        emit(f"{pad}{i:>3}  {soft_norms[i]:>{col_w}.3f}  {disc_cells}", f)


def emit_cross_model_table(
    token_names: list[str],
    ref_hidden: dict[str, list[torch.Tensor]],  # "soft" + each word → L x [d]
    lora_hidden: dict[str, list[torch.Tensor]],
    n_layers: int,
    f,
    report_every: int = 4,
):
    """
    For each input type (soft + each discrete token), show per-layer cosine
    similarity between ref and lora hidden states. This reveals whether LoRA
    shifted soft inputs differently from discrete ones.
    """
    report_layers = set(range(0, n_layers, report_every)) | {n_layers - 1}
    col_w = 9
    pad = "  "
    keys = ["soft"] + token_names

    emit(f"\n{pad}Ref vs LoRA hidden-state cosine similarity (per input type):", f)
    header = f"{pad}{'L':>3}  " + "  ".join(f"{k:>{col_w}}" for k in keys)
    emit(header, f)
    emit(f"{pad}{'-'*3}  {'-'*(len(keys)*(col_w+2))}", f)

    for i in range(n_layers):
        if i not in report_layers:
            continue
        cells = []
        for k in keys:
            r = ref_hidden[k][i]
            l = lora_hidden[k][i]
            cos = F.cosine_similarity(r.unsqueeze(0), l.unsqueeze(0)).item()
            cells.append(f"{cos:>{col_w}.4f}")
        emit(f"{pad}{i:>3}  {'  '.join(cells)}", f)

    # Summary: mean ref-vs-lora cosine per input type
    emit(f"\n{pad}Mean ref-vs-lora cosine similarity (all layers):", f)
    col = max(len(k) for k in keys) + 1
    bar_max = 20
    for k in keys:
        vals = [
            F.cosine_similarity(
                ref_hidden[k][i].unsqueeze(0), lora_hidden[k][i].unsqueeze(0)
            ).item()
            for i in range(n_layers)
        ]
        mean_cos = sum(vals) / len(vals)
        final_cos = vals[-1]
        bar = "█" * int(mean_cos * bar_max) + "░" * (bar_max - int(mean_cos * bar_max))
        emit(f"{pad}  {k:<{col}}  mean={mean_cos:.4f}  final={final_cos:.4f}  {bar}", f)


# ── Experiments ───────────────────────────────────────────────────────────────

experiments = [
    {
        "name": "City — Paris blend",
        "prefix": "Instruction: Vividly describe the concept represented here.\n\nConcept: ",
        "suffix": "\n\nDescription: ",
        "words": [" city", " Paris", " cheese", " wine"],
        "logits": [16.0, 15.5, 15.0, 14.5],
    },
    {
        "name": "Forest — Middle Earth blend",
        "prefix": "Instruction: Vividly describe the concept represented here.\n\nConcept: ",
        "suffix": "\n\nDescription: ",
        "words": [" forest", " wizard", " ring", " elves", " orc"],
        "logits": [16.0, 15.5, 15.0, 14.5, 14.0],
    },
]


os.makedirs(os.path.dirname(CFG.log_file), exist_ok=True)

with open(CFG.log_file, "w", encoding="utf-8") as f:

    emit("HIDDEN ACTIVATION COMPARISON — soft embedding vs discrete tokens", f)
    emit(f"Base model   : {CFG.model_id}", f)
    emit(f"LoRA ckpt    : {CFG.lora_checkpoint}", f)
    emit(f"Layers       : {ref_model.config.num_hidden_layers}", f)

    for exp in experiments:
        emit("\n" + "=" * 100, f)
        emit(f"EXPERIMENT: {exp['name']}", f)
        emit("=" * 100, f)

        # ── Tokenize ──────────────────────────────────────────────────────────
        target_ids = []
        clean_words = []
        for w in exp["words"]:
            ids = tokenizer.encode(w, add_special_tokens=False)
            if len(ids) != 1:
                ids = tokenizer.encode(w.strip(), add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f"'{w}' fragments into multiple tokens: {ids}")
            target_ids.append(ids[0])
            clean_words.append(w.strip())

        logit_values = torch.tensor(exp["logits"], device=device)

        # ── Build soft embedding ───────────────────────────────────────────────
        fake_logits = torch.full((vocab_embs.size(0),), -100.0, device=device)
        for tid, lv in zip(target_ids, logit_values):
            fake_logits[tid] = lv

        soft_vec = soft_thinking(fake_logits, vocab_embs)  # [1, d]

        probs = F.softmax(fake_logits, dim=-1)
        emit(f"\n  Input tokens:", f)
        for w, tid, lv in zip(clean_words, target_ids, logit_values.tolist()):
            p = probs[tid].item()
            emit(f"    '{w}'  id={tid}  logit={lv:.1f}  p={p*100:.2f}%", f)

        # ── Get hidden states for both models ──────────────────────────────────
        # Structure: all_hidden[model_name][input_key] = list of L tensors
        all_hidden: dict[str, dict[str, list[torch.Tensor]]] = {"ref": {}, "lora": {}}

        for model_name, m in MODELS.items():
            print(f"  [{model_name}] soft forward pass...")
            all_hidden[model_name]["soft"] = get_hidden_states(
                soft_vec, exp["prefix"], exp["suffix"], m
            )
            for w, tid in zip(clean_words, target_ids):
                print(f"  [{model_name}] discrete forward pass for '{w}'...")
                disc_vec = vocab_embs[tid].unsqueeze(0).float()
                all_hidden[model_name][w] = get_hidden_states(
                    disc_vec, exp["prefix"], exp["suffix"], m
                )

        n_layers = len(all_hidden["ref"]["soft"])

        # ── Per-model: soft vs discrete comparison ─────────────────────────────
        for model_name in ("ref", "lora"):
            emit(f"\n{'─'*60}", f)
            emit(f"  MODEL: {model_name.upper()}", f)
            emit(f"{'─'*60}", f)

            soft_hidden = all_hidden[model_name]["soft"]
            metrics_per_token: dict[str, dict[str, list[float]]] = {}
            for w in clean_words:
                metrics_per_token[w] = compare_hidden_states(
                    soft_hidden, all_hidden[model_name][w]
                )

            emit_comparison_table(clean_words, metrics_per_token, n_layers, f)
            emit_norm_profile(clean_words, metrics_per_token, soft_hidden, n_layers, f)

        # ── Cross-model: ref vs lora, per input type ───────────────────────────
        emit(f"\n{'─'*60}", f)
        emit(f"  CROSS-MODEL COMPARISON (ref vs lora)", f)
        emit(f"{'─'*60}", f)
        emit_cross_model_table(
            clean_words, all_hidden["ref"], all_hidden["lora"], n_layers, f
        )

    emit("\nDone.", f)

print(f"\nSaved → {CFG.log_file}")
