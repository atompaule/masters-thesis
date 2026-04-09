"""
eval_gsm8k_latent.py

Evaluate a custom embedding approach on GSM8K.
Supports temperature-based multinomial sampling and pass@k evaluation.
"""

import json
import re

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.latent_embedding_experiments.algorithms.config import CFG
from src.latent_embedding_experiments.algorithms.discrete_sharpened import (
    discrete_sharpened,
    discrete_sharpened_dot_rescaled,
)
from src.latent_embedding_experiments.algorithms.soft_thinking import (
    soft_thinking,
    soft_thinking_normalized,
)
from src.latent_embedding_experiments.algorithms.soft_thinking_sharpened import (
    soft_thinking_sharpened_aggregate,
    soft_thinking_sharpened_per_token,
)
from src.latent_embedding_experiments.algorithms.solver import geometric_solver
from src.latent_embedding_experiments.algorithms.utils import emit, select_targets

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

APPROACH = "soft_thinking"

MAX_SAMPLES = 20
MAX_NEW_TOKENS = 1024

N_INTERLOPERS = 10
TARGET_SIM = 0.90

# Sampling config
# TEMPERATURE = 0.0 means greedy (argmax). > 0 enables multinomial sampling.
TEMPERATURE = 0.0
# Number of independent samples per question. pass@k is correct if any is correct.
PASS_K = 1

DEVICE = "cuda"

LOG_FILE = (
    "src/latent_embedding_experiments/logs/"
    f"llama_8b_eval_gsm8k_{APPROACH}_{TARGET_SIM}_t{TEMPERATURE}_pass{PASS_K}.txt"
)
JSON_FILE = LOG_FILE.replace(".txt", ".json")


# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------


def extract_number(text: str):
    """
    Extract the answer number from model output.
    We strip commas immediately to prevent our regexes from choking.
    """
    text_clean = text.replace(",", "")

    answer_match = re.search(
        r"answer\s*:\s*(.*)", text_clean, re.IGNORECASE | re.DOTALL
    )
    if answer_match:
        answer_text = answer_match.group(1)
        boxed = re.search(r"\\boxed\{(-?\d+\.?\d*)\}", answer_text)
        if boxed:
            return boxed.group(1)
        nums = re.findall(r"-?\d+\.?\d*", answer_text)
        if nums:
            return nums[0]

    boxed_all = re.findall(r"\\boxed\{(-?\d+\.?\d*)\}", text_clean)
    if boxed_all:
        return boxed_all[-1]

    matches = re.findall(r"-?\d+\.?\d*", text_clean)
    return matches[-1] if matches else None


def normalize_answer(ans: str) -> str:
    clean_str = ans.strip().replace(",", "")
    try:
        num = float(clean_str)
        if num.is_integer():
            return str(int(num))
        return str(num)
    except ValueError:
        return clean_str


def sample_token(logits: torch.Tensor, temperature: float) -> int:
    """Greedy argmax if temperature == 0, otherwise temperature softmax + multinomial."""
    if temperature == 0.0:
        return int(torch.argmax(logits).item())
    probs = F.softmax(logits / temperature, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


# ---------------------------------------------------------------------------
# Generation with custom embeddings
# ---------------------------------------------------------------------------


def generate_with_approach(
    model,
    tokenizer,
    question: str,
    vocab_embs: torch.Tensor,
    vocab_embs_norm: torch.Tensor,
    temperature: float,
) -> str:
    device = next(model.parameters()).device
    embed_layer = model.get_input_embeddings()

    prompt = (
        f"Answer the following question. "
        f"First reason through it under 'reasoning: ', then give your final numeric answer under 'answer: ' "
        f"in the format: answer: $\\boxed{{N}}$ where N is the number.\n\n"
        f"Question: {question}\n"
        f"reasoning: "
    )
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    inputs_embeds = embed_layer(input_ids)

    with torch.no_grad():
        outputs = model(inputs_embeds=inputs_embeds, use_cache=True)
    past_key_values = outputs.past_key_values
    logits = outputs.logits[0, -1].float()

    generated_text = ""

    for _ in range(MAX_NEW_TOKENS):
        greedy_id = sample_token(logits, temperature)
        token = tokenizer.decode([greedy_id])
        generated_text += token

        if greedy_id == tokenizer.eos_token_id:
            break

        if re.search(r"answer\s*:\s*\$\\boxed\{[^}]+\}", generated_text, re.IGNORECASE):
            break

        # --- target selection (always based on greedy_id for embedding direction) ---
        top1_magnitude = vocab_embs[greedy_id].norm(p=2)
        target_logits, target_ids = select_targets(logits)
        target_probs = F.softmax(logits[target_ids] / CFG.temperature, dim=-1)

        pool_embs = vocab_embs[target_ids]
        target_norms = pool_embs.norm(p=2, dim=1)
        target_magnitude = torch.sum(target_probs * target_norms)

        # --- build next embedding vector ---
        if APPROACH == "discrete_top1":
            next_vec = vocab_embs[greedy_id].unsqueeze(0)

        elif APPROACH == "discrete_cleaned":
            next_vec = discrete_sharpened(
                greedy_id,
                vocab_embs,
                vocab_embs_norm,
                top1_magnitude,
                N_INTERLOPERS,
                TARGET_SIM,
            )

        elif APPROACH == "discrete_cleaned_dot_rescaled":
            next_vec = discrete_sharpened_dot_rescaled(
                greedy_id,
                vocab_embs,
                vocab_embs_norm,
                target_magnitude,
                N_INTERLOPERS,
                TARGET_SIM,
            )

        elif APPROACH == "soft_thinking":
            next_vec = soft_thinking(logits, vocab_embs)

        elif APPROACH == "soft_thinking_normalized":
            next_vec = soft_thinking_normalized(
                logits, vocab_embs_norm, target_magnitude
            )

        elif APPROACH == "clean_soft":
            next_vec = soft_thinking_sharpened_per_token(
                vocab_embs=vocab_embs,
                vocab_embs_norm=vocab_embs_norm,
                target_ids=target_ids,
                target_probs_scaled=target_probs,
                target_magnitude=target_magnitude,
                n_interlopers=N_INTERLOPERS,
                target_sim=TARGET_SIM,
            )

        elif APPROACH == "clean_soft_aggregate":
            v_soft = soft_thinking_normalized(logits, vocab_embs_norm, target_magnitude)
            next_vec = soft_thinking_sharpened_aggregate(
                v_soft=v_soft.squeeze(0),
                vocab_embs=vocab_embs,
                vocab_embs_norm=vocab_embs_norm,
                target_ids=target_ids,
                target_probs_scaled=target_probs,
                target_magnitude=target_magnitude,
                n_interlopers=N_INTERLOPERS,
                target_sim=TARGET_SIM,
            )

        elif APPROACH == "solver":
            next_vec = geometric_solver(logits, vocab_embs)

        else:
            raise ValueError(f"Unknown APPROACH: {APPROACH}")

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
# Main evaluation
# ---------------------------------------------------------------------------


def main():
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        emit(f"Loading model: {CFG.model_id}", f)
        emit(
            f"APPROACH={APPROACH} | TARGET_SIM={TARGET_SIM} | "
            f"N_INTERLOPERS={N_INTERLOPERS} | TEMPERATURE={TEMPERATURE} | PASS_K={PASS_K}",
            f,
        )

        tokenizer = AutoTokenizer.from_pretrained(CFG.model_id)
        model = AutoModelForCausalLM.from_pretrained(
            CFG.model_id, torch_dtype=torch.bfloat16, device_map="auto"
        )
        model.eval()

        vocab_embs = model.get_input_embeddings().weight.detach().float()
        vocab_embs_norm = F.normalize(vocab_embs, dim=1)

        emit("Loading GSM8K...", f)
        dataset = load_dataset("gsm8k", "main", split="test")

        correct = 0
        total = 0
        results: list[dict] = []

        for i, sample in enumerate(dataset):
            if i >= MAX_SAMPLES:
                break

            question = sample["question"]
            gt_answer = extract_number(sample["answer"])

            emit(f"\n{'='*60}", f)
            emit(f"[{i+1}] Q: {question}", f)
            emit(f"GT: {gt_answer}", f)

            # --- Run PASS_K independent samples ---
            any_correct = False
            for k in range(PASS_K):
                pred_text = generate_with_approach(
                    model, tokenizer, question, vocab_embs, vocab_embs_norm, TEMPERATURE
                )
                pred_answer = extract_number(pred_text)

                is_correct = (
                    pred_answer is not None
                    and gt_answer is not None
                    and normalize_answer(pred_answer) == normalize_answer(gt_answer)
                )

                if is_correct:
                    any_correct = True

                emit(f"--- sample {k+1}/{PASS_K} ---", f)
                emit(pred_text, f)
                emit(
                    f"Pred: {pred_answer} | {'✅' if is_correct else '❌'}",
                    f,
                )

                if any_correct:
                    break

            correct += int(any_correct)
            total += 1
            results.append({"index": i + 1, "gt": gt_answer, "pass_at_k": any_correct})

            emit(
                f"--- result (pass@{PASS_K}) ---",
                f,
            )
            emit(
                f"GT: {gt_answer} | {'✅ (any correct)' if any_correct else '❌ (all wrong)'}",
                f,
            )

        acc = correct / total
        emit("\n" + "=" * 50, f)
        emit(f"APPROACH: {APPROACH}", f)
        emit(f"TEMPERATURE: {TEMPERATURE} | PASS_K: {PASS_K}", f)
        emit(f"pass@{PASS_K} accuracy: {acc:.4f} ({correct}/{total})", f)
        emit("=" * 50, f)

    json_payload = {
        "approach": APPROACH,
        "target_sim": TARGET_SIM,
        "n_interlopers": N_INTERLOPERS,
        "temperature": TEMPERATURE,
        "pass_k": PASS_K,
        "accuracy": acc,
        "correct": correct,
        "total": total,
        "results": results,
    }
    with open(JSON_FILE, "w", encoding="utf-8") as jf:
        json.dump(json_payload, jf, indent=2)


if __name__ == "__main__":
    main()