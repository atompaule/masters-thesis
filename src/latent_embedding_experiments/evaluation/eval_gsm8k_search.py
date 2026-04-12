"""
eval_gsm8k_latent.py

Evaluate a custom embedding approach on GSM8K.
Sweeps over (pass_k, target_sim) combinations with a single combined log and json.
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
from src.latent_embedding_experiments.algorithms.token_sharpening import (
    noisy_target_sim,
)
from src.latent_embedding_experiments.algorithms.utils import emit, select_targets

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

APPROACH = "soft_thinking"

MAX_SAMPLES = 1000
MAX_NEW_TOKENS = 1024

N_INTERLOPERS = 10

# Sweep axes
PASS_K_VALUES = [1]
TARGET_SIM_VALUES = [1]

# Sampling config — TEMPERATURE = 0.0 means greedy (argmax)
TEMPERATURE = 0.0

DEVICE = "cuda"

LOG_FILE = (
    "latent_embedding_experiments/logs/" f"llama_8b_eval_gsm8k_{APPROACH}_sweep.txt"
)
JSON_FILE = LOG_FILE.replace(".txt", ".json")


# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------


def extract_number(text: str):
    """Extract the answer number from model output."""
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
    target_sim: float,
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

        if greedy_id == tokenizer.eos_token_id:
            break

        if re.search(r"answer\s*:\s*\$\\boxed\{[^}]+\}", generated_text, re.IGNORECASE):
            break

        top1_magnitude = vocab_embs[greedy_id].norm(p=2)
        target_logits, target_ids = select_targets(logits)
        target_probs = F.softmax(logits[target_ids] / CFG.temperature, dim=-1)

        pool_embs = vocab_embs[target_ids]
        target_norms = pool_embs.norm(p=2, dim=1)
        target_magnitude = torch.sum(target_probs * target_norms)

        if APPROACH == "discrete_top1":
            next_vec = vocab_embs[greedy_id].unsqueeze(0)

        elif APPROACH == "discrete_cleaned":
            next_vec = discrete_sharpened(
                greedy_id,
                vocab_embs,
                vocab_embs_norm,
                top1_magnitude,
                N_INTERLOPERS,
                target_sim,
            )

        elif APPROACH == "discrete_cleaned_dot_rescaled":
            next_vec = discrete_sharpened_dot_rescaled(
                greedy_id,
                vocab_embs,
                vocab_embs_norm,
                target_magnitude,
                N_INTERLOPERS,
                target_sim,
            )

        elif APPROACH == "noisy_discrete":
            v_soft = soft_thinking(logits, vocab_embs)
            v_soft_unit = F.normalize(v_soft, p=2, dim=1)  # [1, d]
            all_cos_sims = (v_soft_unit @ vocab_embs_norm.T).squeeze(0)  # [V]
            nearest_id = all_cos_sims.argmax().item()
            soft_max_sim = all_cos_sims[nearest_id].clamp(-1.0, 1.0).item()
            next_vec = noisy_target_sim(
                vocab_embs[nearest_id],
                target_sim=soft_max_sim,
            ).unsqueeze(
                0
            )  # [1, d]

            # --- Debug: similarity ranking of nearest token ---
            nearest_token = tokenizer.decode([nearest_id]).replace("\n", "\\n")
            greedy_token = tokenizer.decode([greedy_id]).replace("\n", "\\n")
            greedy_sim = all_cos_sims[greedy_id].item()
            # Rank of nearest_id in the soft thinking cosine sim ordering
            nearest_rank = (
                int((all_cos_sims > all_cos_sims[nearest_id]).sum().item()) + 1
            )
            greedy_rank = int((all_cos_sims > greedy_sim).sum().item()) + 1
            diverged = nearest_id != greedy_id
            (
                print(
                    f"[noisy_discrete debug] step | "
                    f"greedy='{greedy_token}' (rank_greedy in ST: {greedy_rank}, sim_greedy in ST: {greedy_sim:.4f}) | "
                    f"fed='{nearest_token}' (rank_fed in ST: {nearest_rank}, sim_fed in ST: {soft_max_sim:.4f}) | "
                    f"{'DIVERGED' if diverged else 'same'}"
                )
                if diverged
                else None
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
                target_sim=target_sim,
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
                target_sim=target_sim,
            )

        elif APPROACH == "solver":
            next_vec = geometric_solver(logits, vocab_embs)

        else:
            raise ValueError(f"Unknown APPROACH: {APPROACH}")

        # --- Resolve the token actually fed forward ---
        # For discrete approaches the fed token is exactly greedy_id.
        # For all others, find the nearest vocab token to next_vec by cosine sim.
        if APPROACH in (
            "discrete_top1",
            "discrete_cleaned",
            "discrete_cleaned_dot_rescaled",
        ):
            fed_id = greedy_id
        else:
            next_vec_unit = F.normalize(next_vec, p=2, dim=1)  # [1, d]
            fed_id = int((next_vec_unit @ vocab_embs_norm.T).squeeze(0).argmax().item())

        token = tokenizer.decode([fed_id])
        generated_text += token

        if fed_id == tokenizer.eos_token_id:
            break

        if re.search(r"answer\s*:\s*\$\\boxed\{[^}]+\}", generated_text, re.IGNORECASE):
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
# Evaluate one (pass_k, target_sim) config over the dataset
# ---------------------------------------------------------------------------


def evaluate_config(
    model,
    tokenizer,
    dataset,
    vocab_embs: torch.Tensor,
    vocab_embs_norm: torch.Tensor,
    pass_k: int,
    target_sim: float,
    f,
) -> dict:
    """Run pass@k evaluation for one config. Returns a results dict."""
    emit(f"\n{'#'*70}", f)
    emit(f"CONFIG: pass_k={pass_k} | target_sim={target_sim}", f)
    emit(f"{'#'*70}", f)

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

        any_correct = False
        for k in range(pass_k):
            pred_text = generate_with_approach(
                model,
                tokenizer,
                question,
                vocab_embs,
                vocab_embs_norm,
                TEMPERATURE,
                target_sim,
            )
            pred_answer = extract_number(pred_text)

            is_correct = (
                pred_answer is not None
                and gt_answer is not None
                and normalize_answer(pred_answer) == normalize_answer(gt_answer)
            )

            if is_correct:
                any_correct = True

            emit(f"--- sample {k+1}/{pass_k} ---", f)
            emit(pred_text, f)
            emit(f"Pred: {pred_answer} | {'✅' if is_correct else '❌'}", f)

            if any_correct:
                break

        correct += int(any_correct)
        total += 1
        results.append({"index": i + 1, "gt": gt_answer, "pass_at_k": any_correct})

        emit(f"--- result (pass@{pass_k}) ---", f)
        emit(
            f"GT: {gt_answer} | {'✅ (any correct)' if any_correct else '❌ (all wrong)'}",
            f,
        )

    acc = correct / total
    emit(
        f"\npass@{pass_k} | target_sim={target_sim} → {acc:.4f} ({correct}/{total})", f
    )

    return {
        "pass_k": pass_k,
        "target_sim": target_sim,
        "accuracy": acc,
        "correct": correct,
        "total": total,
        "results": results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        emit(f"Loading model: {CFG.model_id}", f)
        emit(
            f"APPROACH={APPROACH} | N_INTERLOPERS={N_INTERLOPERS} | "
            f"TEMPERATURE={TEMPERATURE} | MAX_SAMPLES={MAX_SAMPLES}",
            f,
        )
        emit(f"PASS_K sweep: {PASS_K_VALUES}", f)
        emit(f"TARGET_SIM sweep: {TARGET_SIM_VALUES}", f)

        tokenizer = AutoTokenizer.from_pretrained(CFG.model_id)
        model = AutoModelForCausalLM.from_pretrained(
            CFG.model_id, torch_dtype=torch.bfloat16, device_map="auto"
        )
        model.eval()

        vocab_embs = model.get_input_embeddings().weight.detach().float()
        vocab_embs_norm = F.normalize(vocab_embs, dim=1)

        emit("Loading GSM8K...", f)
        dataset = load_dataset("gsm8k", "main", split="test")

        # --- Sweep over all (pass_k, target_sim) combinations ---
        all_configs: list[dict] = []
        for pass_k in PASS_K_VALUES:
            for target_sim in TARGET_SIM_VALUES:
                config_result = evaluate_config(
                    model,
                    tokenizer,
                    dataset,
                    vocab_embs,
                    vocab_embs_norm,
                    pass_k,
                    target_sim,
                    f,
                )
                all_configs.append(config_result)

        # --- Combined summary table ---
        emit(f"\n{'='*70}", f)
        emit("SWEEP SUMMARY", f)
        emit(f"{'pass_k':<10} {'target_sim':<14} {'accuracy':<12} correct/total", f)
        emit("-" * 52, f)
        for cfg in all_configs:
            emit(
                f"{cfg['pass_k']:<10} {cfg['target_sim']:<14} "
                f"{cfg['accuracy']:.4f}       {cfg['correct']}/{cfg['total']}",
                f,
            )
        emit(f"{'='*70}", f)

    # --- Write combined JSON ---
    json_payload = {
        "approach": APPROACH,
        "n_interlopers": N_INTERLOPERS,
        "temperature": TEMPERATURE,
        "max_samples": MAX_SAMPLES,
        "pass_k_values": PASS_K_VALUES,
        "target_sim_values": TARGET_SIM_VALUES,
        "configs": all_configs,
    }
    with open(JSON_FILE, "w", encoding="utf-8") as jf:
        json.dump(json_payload, jf, indent=2)


if __name__ == "__main__":
    main()
