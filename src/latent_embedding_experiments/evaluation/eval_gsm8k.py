"""
eval_gsm8k_latent.py

Evaluate a custom embedding approach on GSM8K.
"""

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

APPROACH = "discrete_top1"

MAX_SAMPLES = 200
MAX_NEW_TOKENS = 1024

N_INTERLOPERS = 10
TARGET_SIM = 0.90

DEVICE = "cuda"

LOG_FILE = (
    "src/latent_embedding_experiments/logs/"
    f"llama_8b_eval_gsm8k_{APPROACH}_{TARGET_SIM}.txt"
)


# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------


def extract_number(text: str):
    """Extract the answer number from model output.

    Priority:
      1. First \\boxed{N} after 'answer:'
      2. First number after 'answer:'
      3. Last \\boxed{N} anywhere in text
      4. Last number in full text
    """
    answer_match = re.search(r"answer\s*:\s*(.*)", text, re.IGNORECASE | re.DOTALL)
    if answer_match:
        answer_text = answer_match.group(1)
        boxed = re.search(r"\\boxed\{(-?\d+\.?\d*)\}", answer_text)
        if boxed:
            return boxed.group(1)
        nums = re.findall(r"-?\d+\.?\d*", answer_text.replace(",", ""))
        if nums:
            return nums[0]

    # Fallback: last boxed anywhere
    boxed_all = re.findall(r"\\boxed\{(-?\d+\.?\d*)\}", text)
    if boxed_all:
        return boxed_all[-1]

    # Final fallback: last number in full text
    matches = re.findall(r"-?\d+\.?\d*", text.replace(",", ""))
    return matches[-1] if matches else None


def normalize_answer(ans: str):
    """
    Cleans the string and distills it down to its purest numeric form,
    stripping away phantom decimals so '75.00' and '75' shake hands.
    """
    # Scrub the dirt (spaces and commas)
    clean_str = ans.strip().replace(",", "")

    try:
        # Weigh it as a true mathematical entity
        num = float(clean_str)

        # If it's a perfectly whole number wearing a decimal disguise (like 75.0),
        # shatter the disguise and return the integer.
        if num.is_integer():
            return str(int(num))

        # Otherwise, let it keep its valid decimals
        return str(num)

    except ValueError:
        # If the model hallucinates total garbage or weird text we can't cast to a float,
        # just return the cleaned string and let it naturally fail the equality check.
        return clean_str


# ---------------------------------------------------------------------------
# Generation with custom embeddings
# ---------------------------------------------------------------------------


def generate_with_approach(model, tokenizer, question, vocab_embs, vocab_embs_norm):
    device = next(model.parameters()).device
    embed_layer = model.get_input_embeddings()

    # ---------------------------
    # Phase 0: initial prompt
    # ---------------------------
    prompt = (
        f"Answer the following question. "
        f"First reason through it under 'reasoning: ', then give your final numeric answer under 'answer: ' "
        f"in the format: answer: $\\boxed{{N}}$ where N is the number.\n\n"
        f"Question: {question}\n"
        f"reasoning: "
    )
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    inputs_embeds = embed_layer(input_ids)

    # Initial forward pass (build KV cache)
    with torch.no_grad():
        outputs = model(inputs_embeds=inputs_embeds, use_cache=True)
    past_key_values = outputs.past_key_values
    logits = outputs.logits[0, -1].float()

    generated_text = ""

    # ---------------------------
    # Single generation loop
    # ---------------------------
    for _ in range(MAX_NEW_TOKENS):
        greedy_id = int(torch.argmax(logits).item())
        token = tokenizer.decode([greedy_id])
        generated_text += token

        # stop at EOS
        if greedy_id == tokenizer.eos_token_id:
            break

        # stop once the answer box is complete
        if re.search(
            r"answer\s*:\s*\$\\boxed\{-?\d+\.?\d*\}", generated_text, re.IGNORECASE
        ):
            break

        # --- target selection ---
        target_logits, target_ids = select_targets(logits)
        target_probs = F.softmax(logits[target_ids] / CFG.temperature, dim=-1)

        pool_embs = vocab_embs[target_ids]
        target_norms = pool_embs.norm(p=2, dim=1)
        target_magnitude = torch.sum(target_probs * target_norms)

        # --- build vector ---
        if APPROACH == "discrete_top1":
            next_vec = vocab_embs[greedy_id].unsqueeze(0)

        elif APPROACH == "discrete_cleaned":
            next_vec = discrete_sharpened(
                greedy_id,
                vocab_embs,
                vocab_embs_norm,
                target_magnitude,
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
                inputs_embeds=next_vec.unsqueeze(0),  # [1,1,d]
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
        tokenizer = AutoTokenizer.from_pretrained(CFG.model_id)
        model = AutoModelForCausalLM.from_pretrained(
            CFG.model_id, torch_dtype=torch.bfloat16, device_map="auto"
        )
        model.eval()

        vocab_embs = model.get_input_embeddings().weight.detach().float()
        vocab_embs_norm = F.normalize(vocab_embs, dim=1)

        emit("Loading GSM8K...")
        dataset = load_dataset("gsm8k", "main", split="test")

        correct = 0
        total = 0

        for i, sample in enumerate(dataset):
            if i >= MAX_SAMPLES:
                break

            question = sample["question"]
            gt_answer = extract_number(sample["answer"])

            pred_text = generate_with_approach(
                model, tokenizer, question, vocab_embs, vocab_embs_norm
            )
            pred_answer = extract_number(pred_text)

            is_correct = (
                pred_answer is not None
                and gt_answer is not None
                and normalize_answer(pred_answer) == normalize_answer(gt_answer)
            )

            correct += int(is_correct)
            total += 1

            emit(f"\n{'='*60}", f)
            emit(f"[{i+1}] Q: {question}", f)
            emit(f"--- response ---\n{pred_text}", f)
            emit(f"--- result ---", f)
            emit(
                f"GT: {gt_answer} | Pred: {pred_answer} | {'✅' if is_correct else '❌'}",
                f,
            )

        acc = correct / total
        emit("\n" + "=" * 50, f)
        emit(f"APPROACH: {APPROACH}", f)
        emit(f"Accuracy: {acc:.4f} ({correct}/{total})", f)
        emit("=" * 50, f)


if __name__ == "__main__":
    main()
