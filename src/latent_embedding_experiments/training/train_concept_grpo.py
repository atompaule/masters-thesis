"""
Experiment 7 — GRPO with dynamic-length concept-token thinking sequences.

The model reasons through a question by producing a variable-length sequence
of soft concept embeddings under 'reasoning: ', then generates a boxed discrete
answer under 'answer: '.

At each think step:
  1. Run a KV-cached forward pass on the current context.
  2. Compute the summed probability mass on answer-trigger tokens.
  3. Sum those probabilities over the last `c` steps (a rolling window).
  4. If that rolling sum exceeds think_stop_threshold AND at least
     min_think_steps have been taken, stop thinking and move to answer mode.
  5. Otherwise sample k tokens at high temperature, build a concept embedding,
     append it to the context, and continue.

The complete sequence fed to the model:

  [prompt tokens] [concept_0] ... [concept_{N-1}] [answer-suffix tokens] [answer tokens]
   real embeddings  continuous blended vectors       real embeddings        real token IDs
"""

import copy
import os
import re
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.latent_embedding_experiments.algorithms.soft_thinking import soft_thinking

# ── Config ────────────────────────────────────────────────────────────────────


@dataclass
class Config:
    # Model
    model_id: str = "meta-llama/Llama-3.2-1B-Instruct"

    # Think sequence
    k: int = 4  # tokens sampled per think step
    sample_temp: float = 0.6  # high temperature → diverse concept sets
    min_think_steps: int = 1  # always take at least this many steps
    max_think_steps: int = 100  # hard cap even if threshold never reached

    # --- The Rolling Wave ---
    trigger_window: int = 3  # 'c': watch the last c steps
    think_stop_threshold: float = 0.8  # sum over 'c' steps must beat this

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_targets: tuple = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )

    # GRPO
    G: int = 4
    kl_coef: float = 0.05
    advantage_eps: float = 1e-8

    # Generation
    max_new_tokens: int = 512
    gen_temp: float = 0.8

    # Training
    lr: float = 5e-5
    batch_size: int = 1
    grad_accum: int = 8
    max_steps: int = 2000
    warmup_steps: int = 100
    max_grad_norm: float = 1.0

    # Logging / checkpointing
    log_every: int = 10
    save_every: int = 250
    log_file: str = "logs/exp7_think_sequences.txt"
    output_dir: str = "checkpoints/exp7_grpo_think"


CFG = Config()


# ── Model + LoRA ──────────────────────────────────────────────────────────────

print(f"Loading {CFG.model_id}...")
tokenizer = AutoTokenizer.from_pretrained(CFG.model_id)
tokenizer.pad_token = tokenizer.eos_token

base = AutoModelForCausalLM.from_pretrained(
    CFG.model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

lora_cfg = LoraConfig(
    r=CFG.lora_r,
    lora_alpha=CFG.lora_alpha,
    lora_dropout=CFG.lora_dropout,
    target_modules=list(CFG.lora_targets),
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(base, lora_cfg)
model.print_trainable_parameters()

ref_model = copy.deepcopy(model).eval()
for p in ref_model.parameters():
    p.requires_grad_(False)

device = next(model.parameters()).device
vocab_embs = model.get_input_embeddings().weight.detach().float()


# ── Answer-trigger tokens ─────────────────────────────────────────────────────
# We sever the word from the colon to ensure we catch the raw, isolated token.

_TRIGGER_STRINGS = ["answer", "Answer", " answer", " Answer", "\nanswer", "\nAnswer"]
_trigger_id_set = set()

for _s in _TRIGGER_STRINGS:
    # We take the *last* token of the encoded string to bypass any strange prefix merging
    _ids = tokenizer.encode(_s, add_special_tokens=False)
    if _ids:
        _trigger_id_set.add(_ids[-1])

ANSWER_TRIGGER_IDS = torch.tensor(
    sorted(_trigger_id_set), dtype=torch.long, device=device
)
print(f"Answer-trigger token IDs : {ANSWER_TRIGGER_IDS.tolist()}")
print(
    f"  decoded                : {[tokenizer.decode([i]) for i in ANSWER_TRIGGER_IDS.tolist()]}"
)


# ── Dataset ───────────────────────────────────────────────────────────────────

dataset = load_dataset("gsm8k", "main", split="train")


def make_prompt(question: str) -> str:
    return (
        "Answer the following question. "
        "First reason through it under 'reasoning: ', then give your final "
        r"numeric answer under 'answer: ' in the format: answer: $\boxed{N}$"
        " where N is the number.\n\n"
        f"Question: {question}\n"
        "reasoning: "
    )


ANSWER_SUFFIX = "\nConclusion: "


# ── Reward ────────────────────────────────────────────────────────────────────


def _last_number(text: str) -> float | None:
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    return float(nums[-1].replace(",", "")) if nums else None


def gsm8k_reward(generated: str, gt_answer: str) -> float:
    m = re.search(r"####\s*([\d,]+)", gt_answer)
    if m is None:
        return 0.0
    gt = float(m.group(1).replace(",", ""))
    boxed = re.search(r"\\boxed\{([\d,.\-]+)\}", generated)
    pred = float(boxed.group(1).replace(",", "")) if boxed else _last_number(generated)
    return 1.0 if pred is not None and abs(pred - gt) < 1e-3 else 0.0


# ── Concept embedding ─────────────────────────────────────────────────────────


def build_concept_embedding(
    logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    probs = F.softmax(logits / CFG.sample_temp, dim=-1)
    sample_ids = torch.multinomial(probs, CFG.k, replacement=False)
    sample_probs = probs[sample_ids]

    fake_logits = torch.full((vocab_embs.size(0),), -100.0, device=device)
    for tid in sample_ids.tolist():
        fake_logits[tid] = logits[tid]

    concept_vec = soft_thinking(fake_logits, vocab_embs.to(device))
    return concept_vec, sample_ids, sample_probs


# ── Think sequence builder ────────────────────────────────────────────────────


@torch.no_grad()
def build_think_sequence(
    prefix_ids: torch.Tensor,
    m: torch.nn.Module | None = None,
) -> tuple[list, list, list, str, object, list, list]:
    if m is None:
        m = model

    prefix_emb = m.get_input_embeddings()(prefix_ids)
    out = m(inputs_embeds=prefix_emb.to(m.dtype), use_cache=True)
    past_kv = out.past_key_values
    logits = out.logits[0, -1, :].float()

    concept_vecs = []
    sample_ids_list = []
    sample_probs_list = []
    argmax_ids_list = []
    trigger_probs = []
    window_sums = []  # <--- Keep track of the rolling pressure
    stop_reason = "max_steps"

    for step in range(CFG.max_think_steps):
        probs = F.softmax(logits, dim=-1)
        trigger_p = probs[ANSWER_TRIGGER_IDS].sum().item()
        trigger_probs.append(trigger_p)

        # Calculate the momentum over the last `c` steps
        window_sum = sum(trigger_probs[-CFG.trigger_window :])
        window_sums.append(window_sum)

        # The dam breaks only if the sustained wave crests the threshold
        if step >= CFG.min_think_steps and window_sum > CFG.think_stop_threshold:
            stop_reason = "threshold"
            break

        argmax_ids_list.append(logits.argmax(dim=-1).item())

        concept_vec, sample_ids, sample_probs = build_concept_embedding(logits)
        concept_vecs.append(concept_vec)
        sample_ids_list.append(sample_ids)
        sample_probs_list.append(sample_probs)

        concept_emb = concept_vec.to(m.dtype).unsqueeze(0)
        out = m(inputs_embeds=concept_emb, past_key_values=past_kv, use_cache=True)
        past_kv = out.past_key_values
        logits = out.logits[0, -1, :].float()

    return (
        concept_vecs,
        sample_ids_list,
        sample_probs_list,
        trigger_probs,
        stop_reason,
        past_kv,
        argmax_ids_list,
        window_sums,  # Pass the swelling tide to the logger
    )


# ── Autoregressive generation from a KV cache ─────────────────────────────────


@torch.no_grad()
def generate_from_past_kv(
    past_kv: object,
    first_logits: torch.Tensor,
    m: torch.nn.Module | None = None,
) -> list[int]:
    if m is None:
        m = model

    gen_ids = []
    logits = first_logits

    for _ in range(CFG.max_new_tokens):
        probs = F.softmax(logits / CFG.gen_temp, dim=-1)
        next_id = torch.multinomial(probs, 1).item()
        gen_ids.append(next_id)
        if next_id == tokenizer.eos_token_id:
            break

        next_emb = m.get_input_embeddings()(torch.tensor([[next_id]], device=device))
        out = m(inputs_embeds=next_emb, past_key_values=past_kv, use_cache=True)
        past_kv = out.past_key_values
        logits = out.logits[0, -1, :].float()

    return gen_ids


# ── Rollout ───────────────────────────────────────────────────────────────────


@torch.no_grad()
def rollout(prompt: str, gt_answer: str) -> dict:
    prefix_ids = tokenizer(
        prompt, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(device)

    suffix_ids = tokenizer(
        ANSWER_SUFFIX, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(device)

    (
        concept_vecs,
        sample_ids_list,
        sample_probs_list,
        trigger_probs,
        stop_reason,
        past_kv_think,
        argmax_ids_list,
        window_sums,
    ) = build_think_sequence(prefix_ids)

    out = model(input_ids=suffix_ids, past_key_values=past_kv_think, use_cache=True)
    past_kv_suffix = out.past_key_values
    first_gen_logits = out.logits[0, -1, :].float()

    gen_ids = generate_from_past_kv(past_kv_suffix, first_gen_logits)
    generated_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    return {
        "prefix_ids": prefix_ids,
        "suffix_ids": suffix_ids,
        "concept_vecs": concept_vecs,
        "sample_ids_list": sample_ids_list,
        "sample_probs_list": sample_probs_list,
        "argmax_ids_list": argmax_ids_list,
        "trigger_probs": trigger_probs,
        "window_sums": window_sums,
        "stop_reason": stop_reason,
        "gen_ids": gen_ids,
        "generated_text": generated_text,
        "reward": gsm8k_reward(generated_text, gt_answer),
    }


# ── Log-prob computation & GRPO Loss ──────────────────────────────────────────


def compute_policy_log_probs(r: dict) -> tuple[torch.Tensor, torch.Tensor]:
    concept_vecs = r["concept_vecs"]
    sample_ids_list = r["sample_ids_list"]
    gen_ids = r["gen_ids"]
    N = len(concept_vecs)
    T = len(gen_ids)

    prefix_emb = model.get_input_embeddings()(r["prefix_ids"])
    suffix_emb = model.get_input_embeddings()(r["suffix_ids"])
    P = prefix_emb.shape[1]
    S = suffix_emb.shape[1]
    d = prefix_emb.shape[-1]

    think_embs = (
        torch.cat(
            [v.detach().to(model.dtype).unsqueeze(0) for v in concept_vecs], dim=1
        )
        if N > 0
        else torch.zeros(1, 0, d, device=device, dtype=model.dtype)
    )

    if T > 0:
        gen_ids_t = torch.tensor(gen_ids, device=device)
        gen_emb = model.get_input_embeddings()(gen_ids_t.unsqueeze(0))
        input_emb = torch.cat(
            [
                prefix_emb.to(model.dtype),
                think_embs,
                suffix_emb.to(model.dtype),
                gen_emb[:, :-1],
            ],
            dim=1,
        )
    else:
        input_emb = torch.cat(
            [
                prefix_emb.to(model.dtype),
                think_embs,
                suffix_emb.to(model.dtype),
            ],
            dim=1,
        )

    logits = model(inputs_embeds=input_emb).logits[0].float()

    sampling_lp = torch.zeros(1, device=device).squeeze()
    for t, sample_ids in enumerate(sample_ids_list):
        logits_t = logits[P - 1 + t]

        # At step t, the model continues. We penalize the probability mass of the trigger
        # AT THIS SPECIFIC STEP, not the window sum, because the local logit is what we backprop through.
        p_trigger = F.softmax(logits_t, dim=-1)[ANSWER_TRIGGER_IDS].sum()
        sampling_lp = sampling_lp + torch.log(1.0 - p_trigger + 1e-8)

        log_p = F.log_softmax(logits_t / CFG.sample_temp, dim=-1)
        sampling_lp = sampling_lp + log_p[sample_ids].sum()

    if r["stop_reason"] == "threshold":
        logits_stop = logits[P - 1 + N]
        p_trigger = F.softmax(logits_stop, dim=-1)[ANSWER_TRIGGER_IDS].sum()
        # When it stops, it's because the final step tipped the rolling scale.
        # We reward that final push of probability.
        sampling_lp = sampling_lp + torch.log(p_trigger + 1e-8)

    gen_lp = torch.zeros(1, device=device).squeeze()
    if T > 0:
        gen_logits = logits[P + N + S - 1 : P + N + S - 1 + T]
        token_lp = F.log_softmax(gen_logits, dim=-1)
        gen_lp = token_lp[torch.arange(T, device=device), gen_ids_t].sum()

    return sampling_lp, gen_lp


@torch.no_grad()
def compute_ref_log_probs(r: dict) -> torch.Tensor:
    gen_ids = r["gen_ids"]
    if not gen_ids:
        return torch.zeros(1, device=device).squeeze()

    N = len(r["concept_vecs"])
    prefix_emb = ref_model.get_input_embeddings()(r["prefix_ids"])
    suffix_emb = ref_model.get_input_embeddings()(r["suffix_ids"])
    P = prefix_emb.shape[1]
    S = suffix_emb.shape[1]
    d = prefix_emb.shape[-1]

    think_embs = (
        torch.cat(
            [v.to(ref_model.dtype).unsqueeze(0) for v in r["concept_vecs"]], dim=1
        )
        if N > 0
        else torch.zeros(1, 0, d, device=device, dtype=ref_model.dtype)
    )

    gen_ids_t = torch.tensor(gen_ids, device=device)
    gen_emb = ref_model.get_input_embeddings()(gen_ids_t.unsqueeze(0))

    input_emb = torch.cat(
        [
            prefix_emb.to(ref_model.dtype),
            think_embs,
            suffix_emb.to(ref_model.dtype),
            gen_emb[:, :-1],
        ],
        dim=1,
    )

    logits = ref_model(inputs_embeds=input_emb).logits[0].float()
    gen_logits = logits[P + N + S - 1 : P + N + S - 1 + len(gen_ids)]
    token_lp = F.log_softmax(gen_logits, dim=-1)
    return token_lp[torch.arange(len(gen_ids), device=device), gen_ids_t].sum()


def grpo_loss(rollouts: list[dict]) -> torch.Tensor:
    rewards = torch.tensor([r["reward"] for r in rollouts], dtype=torch.float32)
    advantages = (rewards - rewards.mean()) / (rewards.std() + CFG.advantage_eps)

    loss = torch.zeros(1, device=device)
    for i, r in enumerate(rollouts):
        sampling_lp, gen_lp = compute_policy_log_probs(r)
        ref_gen_lp = compute_ref_log_probs(r)
        A = advantages[i].to(device)
        loss = loss + (
            -A * (sampling_lp + gen_lp) + CFG.kl_coef * (gen_lp - ref_gen_lp)
        )
    return loss / CFG.G


# ── Optimizer + scheduler ─────────────────────────────────────────────────────

optimizer = AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=CFG.lr,
    weight_decay=0.01,
)
warmup_sched = LinearLR(
    optimizer, start_factor=0.1, end_factor=1.0, total_iters=CFG.warmup_steps
)
cosine_sched = CosineAnnealingLR(
    optimizer, T_max=CFG.max_steps - CFG.warmup_steps, eta_min=CFG.lr * 0.1
)
scheduler = SequentialLR(
    optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[CFG.warmup_steps]
)

os.makedirs(CFG.output_dir, exist_ok=True)
os.makedirs(os.path.dirname(CFG.log_file), exist_ok=True)
seq_log = open(CFG.log_file, "w", encoding="utf-8")


def log(s: str = "") -> None:
    print(s)
    seq_log.write(s + "\n")
    seq_log.flush()


# ── Training loop ─────────────────────────────────────────────────────────────

data_iter = iter(DataLoader(dataset, batch_size=CFG.batch_size, shuffle=True))
optimizer.zero_grad()
running_reward = 0.0
running_loss = 0.0

SEP = "─" * 80

log("Starting GRPO dynamic-think training...")
log(
    f"min_think={CFG.min_think_steps}  max_think={CFG.max_think_steps}  "
    f"threshold={CFG.think_stop_threshold}  window={CFG.trigger_window}  k={CFG.k}"
)

for step in range(1, CFG.max_steps + 1):

    try:
        batch = next(data_iter)
    except StopIteration:
        data_iter = iter(DataLoader(dataset, batch_size=CFG.batch_size, shuffle=True))
        batch = next(data_iter)

    step_loss = 0.0
    step_reward = 0.0
    n_skipped = 0

    for question, answer in zip(batch["question"], batch["answer"]):

        prompt = make_prompt(question)
        rollouts = [rollout(prompt, answer) for _ in range(CFG.G)]
        rewards = [r["reward"] for r in rollouts]
        step_reward += sum(rewards) / len(rewards)

        gt_match = re.search(r"####\s*([\d,]+)", answer)
        gt_str = gt_match.group(1) if gt_match else "?"
        emoji_row = "  ".join("✅" if r == 1.0 else "❌" for r in rewards)

        log(f"\n{SEP}")
        log(f"step {step:5d}  |  GT: {gt_str}  |  {emoji_row}")
        log(f"Q: {question[:200].replace(chr(10), ' ')}")

        for i, r in enumerate(rollouts):
            tag = "✅" if r["reward"] == 1.0 else "❌"
            n = len(r["concept_vecs"])
            log(
                f"\n  [{i}] {tag}  Sequence Length: {n}  (Halted by: {r['stop_reason']})"
            )

            # Decode the sheer, highest-probability instinct
            if n > 0:
                argmax_text = tokenizer.decode(r["argmax_ids_list"])
                clean_text = argmax_text.replace("\n", "↵").replace("\t", "⇥")
                log(f"    🧠 Discrete Path: {clean_text}")
            else:
                log(f"    🧠 Discrete Path: (No think steps taken)")

            # The tension meter now reflects the heavy, rolling wave of the last 'c' steps
            if r["window_sums"]:
                final_wave = r["window_sums"][-1]
                tension = min(10, int((final_wave / CFG.think_stop_threshold) * 10))
                gauge = "■" * tension + "□" * (10 - tension)
                stop_str = (
                    "Wave Crested"
                    if r["stop_reason"] == "threshold"
                    else "Hit max steps"
                )
                log(f"    🏁 stop │ {gauge} {final_wave:.3f} │ ({stop_str})")

            full = r["generated_text"].strip()
            indented = "\n           ".join(full.splitlines()) if full else "(empty)"
            log(f"    🗣️  Answer:\n           {indented}")

        log(SEP)

        if len(set(rewards)) == 1:
            n_skipped += 1
            continue

        loss = grpo_loss(rollouts) / CFG.grad_accum
        loss.backward()
        step_loss += loss.item() * CFG.grad_accum

    running_reward += step_reward / CFG.batch_size
    running_loss += step_loss / CFG.batch_size

    if step % CFG.grad_accum == 0:
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], CFG.max_grad_norm
        )
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    if step % CFG.log_every == 0:
        avg_r = running_reward / CFG.log_every
        avg_l = running_loss / CFG.log_every
        lr_now = scheduler.get_last_lr()[0]
        log(
            f"\n>>> step {step:5d} | loss {avg_l:.4f} | reward {avg_r:.3f}"
            f" | lr {lr_now:.2e} | skipped {n_skipped}/{CFG.batch_size}"
        )
        running_reward = 0.0
        running_loss = 0.0

    if step % CFG.save_every == 0:
        ckpt = os.path.join(CFG.output_dir, f"step_{step:05d}")
        model.save_pretrained(ckpt)
        tokenizer.save_pretrained(ckpt)
        log(f"  -> saved {ckpt}")

seq_log.close()
log("Done.")
