"""
eval_winogrande_latent.py

Experiment 4.II — Priority 3 runs: LLaMA 3.1 8B IT × WinoGrande
Compares Soft Thinking, Noisy Greedy Discrete (Isotropic Gaussian),
Greedy Discrete, and Default Baseline (temperature sampling).

One combined log + JSON for all 4 approaches.
"""

import argparse
import json
import re

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.latent_embedding_experiments.algorithms.config import CFG
from src.latent_embedding_experiments.algorithms.soft_thinking import soft_thinking
from src.latent_embedding_experiments.algorithms.token_sharpening import (
    noisy_target_sim,
)
from src.latent_embedding_experiments.algorithms.utils import emit

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

MAX_SAMPLES = 10000  # WinoGrande-XL validation split has 1267 items
MAX_NEW_TOKENS = 1024  # model reasons before giving final answer

APPROACH_CONFIGS = [
    {
        "name": "soft_thinking",
        "sampling_temp": 0.0,
        "top_p": 0.95,
    },
    {
        "name": "noisy_discrete",
        "noise_variant": "gaussian",
        "sampling_temp": 0.0,
        "top_p": 0.95,
    },
    {
        "name": "discrete_top1",
        "sampling_temp": 0.0,
        "top_p": 0.95,
    },
    {
        "name": "default_baseline",
        "sampling_temp": 0.6,
        "top_p": 0.90,
    },
]

DEVICE = "cuda"

# ---------------------------------------------------------------------------
# WinoGrande-specific utils
# ---------------------------------------------------------------------------


def build_prompt(sentence: str, option1: str, option2: str) -> str:
    """Format a fill-in-the-blank prompt for WinoGrande."""
    return (
        "Choose the correct option to fill in the blank (_) in the following sentence. "
        "First reason through it under 'reasoning: ', then give your final answer "
        "under 'answer: ' using exactly the format: answer: 1 or answer: 2.\n\n"
        f"Sentence: {sentence}\n"
        f"Option 1: {option1}\n"
        f"Option 2: {option2}\n\n"
        "reasoning:"
    )


def extract_choice(text: str) -> str | None:
    # Require colon to avoid matching reasoning prose
    matches = re.findall(r"answer\s*:\s*([12])\b", text, re.IGNORECASE)
    if matches:
        return matches[-1]  # take last in case model self-corrects

    return None


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def sample_token(
    logits: torch.Tensor,
    temperature: float,
    top_p: float = 1.0,
) -> int:
    """Greedy argmax if temperature == 0, else temperature + nucleus sampling."""
    if temperature == 0.0:
        return int(torch.argmax(logits).item())

    probs = F.softmax(logits / temperature, dim=-1)

    if top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=0)
        mask = (cumulative - sorted_probs) >= top_p
        sorted_probs[mask] = 0.0
        probs = torch.zeros_like(probs).scatter_(0, sorted_idx, sorted_probs)
        probs = probs / probs.sum()

    return int(torch.multinomial(probs, num_samples=1).item())


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def generate_with_approach(
    model,
    tokenizer,
    prompt: str,
    vocab_embs: torch.Tensor,
    vocab_embs_norm: torch.Tensor,
    config: dict,
) -> str:
    """Autoregressive generation for one approach config."""
    approach = config["name"]
    sampling_temp = config["sampling_temp"]
    top_p = config.get("top_p", 1.0)

    device = next(model.parameters()).device
    embed_layer = model.get_input_embeddings()

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    inputs_embeds = embed_layer(input_ids)

    with torch.no_grad():
        outputs = model(inputs_embeds=inputs_embeds, use_cache=True)
    past_key_values = outputs.past_key_values
    logits = outputs.logits[0, -1].float()

    generated_text = ""

    for _ in range(MAX_NEW_TOKENS):
        greedy_id = sample_token(logits, sampling_temp, top_p)

        if greedy_id == tokenizer.eos_token_id:
            break
        if re.search(r"answer\s*[:\-]?\s*[12]\b", generated_text, re.IGNORECASE):
            break

        # ---- Build next embedding ----------------------------------------

        if approach in ("discrete_top1", "default_baseline"):
            next_vec = vocab_embs[greedy_id].unsqueeze(0)
            fed_id = greedy_id

        elif approach == "soft_thinking":
            next_vec = soft_thinking(logits, vocab_embs)
            next_vec_unit = F.normalize(next_vec, p=2, dim=1)
            fed_id = int((next_vec_unit @ vocab_embs_norm.T).squeeze(0).argmax().item())

        elif approach == "noisy_discrete":
            v_soft = soft_thinking(logits, vocab_embs)
            v_soft_unit = F.normalize(v_soft, p=2, dim=1)
            all_cos_sims = (v_soft_unit @ vocab_embs_norm.T).squeeze(0)
            nearest_id = int(all_cos_sims.argmax().item())
            soft_max_sim = all_cos_sims[nearest_id].clamp(-1.0, 1.0).item()
            next_vec = noisy_target_sim(
                vocab_embs[nearest_id],
                target_sim=soft_max_sim,
            ).unsqueeze(0)
            next_vec_unit = F.normalize(next_vec, p=2, dim=1)
            fed_id = int((next_vec_unit @ vocab_embs_norm.T).squeeze(0).argmax().item())

        else:
            raise ValueError(f"Unknown approach: {approach!r}")

        token = tokenizer.decode([fed_id])
        generated_text += token

        if fed_id == tokenizer.eos_token_id:
            break
        if re.search(r"answer\s*[:\-]?\s*[12]\b", generated_text, re.IGNORECASE):
            break

        next_vec = next_vec.to(device=device, dtype=model.dtype)

        with torch.no_grad():
            outputs = model(
                inputs_embeds=next_vec.unsqueeze(0),  # [1, 1, d]
                past_key_values=past_key_values,
                use_cache=True,
            )

        past_key_values = outputs.past_key_values
        logits = outputs.logits[0, -1].float()

    return generated_text


# ---------------------------------------------------------------------------
# Evaluate one approach config over the dataset
# ---------------------------------------------------------------------------


def evaluate_config(
    model,
    tokenizer,
    dataset,
    vocab_embs: torch.Tensor,
    vocab_embs_norm: torch.Tensor,
    config: dict,
    f,
) -> dict:
    """Run pass@1 evaluation for one config. Returns a results dict."""
    emit(f"\n{'#'*70}", f)
    emit(f"CONFIG: {config}", f)
    emit(f"{'#'*70}", f)

    correct = 0
    total = 0
    results: list[dict] = []

    for i, sample in enumerate(dataset):
        if i >= MAX_SAMPLES:
            break

        sentence = sample["sentence"]
        option1 = sample["option1"]
        option2 = sample["option2"]
        gt_answer = sample["answer"]  # "1" or "2"
        prompt = build_prompt(sentence, option1, option2)

        emit(f"\n{'='*60}", f)
        emit(f"[{i+1}] Sentence: {sentence}", f)
        emit(f"Option 1: {option1} | Option 2: {option2}", f)
        emit(f"GT: {gt_answer}", f)

        pred_text = generate_with_approach(
            model,
            tokenizer,
            prompt,
            vocab_embs,
            vocab_embs_norm,
            config,
        )
        pred_answer = extract_choice(pred_text)
        is_correct = pred_answer is not None and pred_answer == gt_answer

        correct += int(is_correct)
        total += 1
        results.append(
            {
                "index": i + 1,
                "gt": gt_answer,
                "pred": pred_answer,
                "correct": is_correct,
            }
        )

        emit(pred_text, f)
        emit(f"Pred: {pred_answer} | {'✅' if is_correct else '❌'}", f)

    acc = correct / total
    emit(f"\napproach={config['name']} → {acc:.4f} ({correct}/{total})", f)

    return {
        "config": config,
        "accuracy": acc,
        "correct": correct,
        "total": total,
        "results": results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id",
        type=str,
        default=CFG.model_id,
        help="HuggingFace model ID or local path (default: model_id)",
    )
    args = parser.parse_args()
    model_id = args.model_id

    model_slug = model_id.split("/")[-1].lower()
    log_file = (
        f"src/latent_embedding_experiments/logs/"
        f"{model_slug}_eval_winogrande_exp4ii_p3_sweep.txt"
    )
    json_file = log_file.replace(".txt", ".json")

    with open(log_file, "w", encoding="utf-8") as f:
        emit(f"Loading model: {model_id}", f)
        emit(f"Experiment 4.II — Priority 3 | {model_id} × WinoGrande-XL", f)
        emit(f"MAX_SAMPLES={MAX_SAMPLES} | MAX_NEW_TOKENS={MAX_NEW_TOKENS}", f)
        emit(f"Approach configs:", f)
        for cfg in APPROACH_CONFIGS:
            emit(f"  {cfg}", f)

        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True,
        )
        model.eval()

        vocab_embs = model.get_input_embeddings().weight.detach().float()
        vocab_embs_norm = F.normalize(vocab_embs, dim=1)

        emit("Loading WinoGrande-XL (validation)...", f)
        dataset = load_dataset(
            "winogrande", "winogrande_xl", split="validation", trust_remote_code=True
        )

        # --- Sweep over all approach configs ---
        all_results: list[dict] = []
        for cfg in APPROACH_CONFIGS:
            result = evaluate_config(
                model,
                tokenizer,
                dataset,
                vocab_embs,
                vocab_embs_norm,
                cfg,
                f,
            )
            all_results.append(result)

        # --- Summary table ---
        emit(f"\n{'='*70}", f)
        emit("SWEEP SUMMARY — Experiment 4.II Priority 3", f)
        emit(f"Model:     {model_id}", f)
        emit(f"Benchmark: winogrande / winogrande_xl / validation", f)
        emit(
            f"\n{'approach':<25} {'samp_temp':<12} {'top_p':<8} {'accuracy':<12} correct/total",
            f,
        )
        emit("-" * 70, f)
        for r in all_results:
            c = r["config"]
            emit(
                f"{c['name']:<25} "
                f"{str(c['sampling_temp']):<12} "
                f"{str(c['top_p']):<8} "
                f"{r['accuracy']:.4f}       "
                f"{r['correct']}/{r['total']}",
                f,
            )
        emit(f"{'='*70}", f)

    # --- Write combined JSON ---
    json_payload = {
        "experiment": "4.II",
        "priority": 3,
        "model": model_id,
        "benchmark": "winogrande/winogrande_xl/validation",
        "max_samples": MAX_SAMPLES,
        "runs": all_results,
    }
    with open(json_file, "w", encoding="utf-8") as jf:
        json.dump(json_payload, jf, indent=2)


if __name__ == "__main__":
    main()
