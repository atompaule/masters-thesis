import gc
import os
import time

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch import bfloat16
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.latent_embedding_experiments.algorithms.config import CFG
from src.latent_embedding_experiments.algorithms.soft_thinking import soft_thinking
from src.latent_embedding_experiments.data.gsm8k import (
    dataset,
    gsm8k_reward,
    make_prompt,
)
from src.latent_embedding_experiments.training.utils import (
    format_duration,
    get_memory_gb,
    log,
    log_rollouts,
)

RESUME_FROM = "checkpoints/exp7_grpo_think/step_00500"

APPROACHES = ""
DATE = str(time.time()).split(".")[0]
OUTPUT_DIR = "latent_embedding_experiments/checkpoints/exp7_grpo_think"
LOG_FILE = f"latent_embedding_experiments/logs/rl_continuous_cot_{CFG.model_id.split('/')[-1]}_{DATE}.txt"

LOG_EVERY = 1
SAVE_EVERY = 100

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
seq_log = open(LOG_FILE, "w", encoding="utf-8")

log(seq_log, f"Loading {CFG.model_id}...")
tokenizer = AutoTokenizer.from_pretrained(CFG.model_id)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    CFG.model_id,
    device_map="auto",
    torch_dtype=bfloat16,
)

# how lora works
# W_eff = W_frozen + (alpha / r) * B @ A, whereas A.shape = (r, d_in) and B.shape = (d_out, r)
# out = W_eff @ in = (W_frozen + (alpha / r) * B @ A) * in = W_frozen @ in + (alpha / r) * B @ (A @ in)

if RESUME_FROM:
    model = PeftModel.from_pretrained(base_model, RESUME_FROM, is_trainable=True)
else:
    lora_cfg = LoraConfig(
        r=CFG.rl_config.lora_r,  # number of cols in lora matrices
        lora_alpha=CFG.rl_config.lora_alpha,  # scalar that determines relative influence of the lora matrices on frozen base
        target_modules=list(
            ["c_attn", "c_proj", "c_fc"]
            if (CFG.model_id == "gpt2" or CFG.model_id == "weiser/30M-0.4")
            else CFG.rl_config.lora_targets
        ),  # fine-tune all projections of the model, except for embedding matrices
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(base_model, lora_cfg)

device = model.device
log(seq_log, f"Using {device}")

model_embeddings = model.get_input_embeddings()
vocab_embeds_device = model_embeddings.weight.detach().bfloat16().to(device)

# these are only needed for the gaussian_noise approach, but compute them here for efficiency
rms_norm = vocab_embeds_device.norm(dim=-1).pow(2).mean().sqrt()
sigma = 0.33 * rms_norm

# TODO: should we use warmup steps despite temperature curriculum?
optimizer = AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=CFG.rl_config.learning_rate,
    weight_decay=0.01,
)

start_step = 0
if RESUME_FROM:
    state_path = os.path.join(RESUME_FROM, "training_state.pt")
    if os.path.exists(state_path):
        state = torch.load(state_path, map_location="cpu")
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["rng_state"])
        start_step = state["step"] + 1
        log(seq_log, f"Resumed from step {state['step']} ({RESUME_FROM})")
    else:
        log(seq_log, f"WARNING: No training_state.pt found in {RESUME_FROM}; starting from step 0")


_TRIGGER_STRINGS = [
    "answer",
    "Answer",
    " answer",
    " Answer",
    "\nanswer",
    "\nAnswer",
]
_trigger_id_set = set()

for _s in _TRIGGER_STRINGS:
    # We take the *last* token of the encoded string to bypass any strange prefix merging
    _ids = tokenizer.encode(_s, add_special_tokens=False)
    if _ids:
        _trigger_id_set.add(_ids[-1])

ANSWER_TRIGGER_IDS = torch.tensor(
    sorted(_trigger_id_set), dtype=torch.long, device=device
)

ANSWER_STRING = "\nanswer: "
BOXED_STRING = "$\\boxed{"

prefix_ids_threshold = tokenizer(
    BOXED_STRING, return_tensors="pt", add_special_tokens=False
).input_ids.to(device)
prefix_ids_maxsteps = tokenizer(
    ANSWER_STRING + BOXED_STRING, return_tensors="pt", add_special_tokens=False
).input_ids.to(device)


def get_temp(step: int):
    n_increments = (
        CFG.rl_config.max_update_steps - 1
    ) // CFG.rl_config.temperature_increment_every
    progress = min(
        (step - 1) // CFG.rl_config.temperature_increment_every / n_increments, 1.0
    )
    return CFG.rl_config.temperature_start + progress * (
        CFG.rl_config.temperature_end - CFG.rl_config.temperature_start
    )


def build_concept_embedding(logits: torch.Tensor, temp: float, approach: str):
    concept_vec, sample_ids, sample_probs = None, None, None

    if approach == "token_sampling":
        probs = F.softmax(logits / temp, dim=-1)
        sorted_probs, _ = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        nucleus_mask = (cumsum - sorted_probs) < CFG.top_p
        nucleus_mask[0] = True  # always at least 1
        n = int(nucleus_mask.sum().item())

        # note that i.i.d (replacement=True) would be needed to make log pi a trivial sum, but i violate this intentionally and account for it later in policy_log_prob
        sample_ids = torch.multinomial(probs, n, replacement=False)
        sample_probs = probs[sample_ids]

        fake_logits = torch.full((vocab_embeds_device.size(0),), -1000.0, device=device)
        fake_logits[sample_ids] = logits[sample_ids]

        concept_vec = soft_thinking(fake_logits, vocab_embeds_device).squeeze(0)
    elif approach == "gaussian_noise":
        soft_vec = soft_thinking(logits / temp, vocab_embeds_device).squeeze(0)

        eps = torch.randn_like(soft_vec) * sigma
        concept_vec = soft_vec + eps
    elif approach == "dirichlet_sampling":
        pass

    return concept_vec, sample_ids, sample_probs


@torch.no_grad()
def rollout_single_think_sequence(prompt_ids: torch.Tensor, temp: float, approach: str):
    prompt_embeds = model_embeddings(prompt_ids)

    out = model(inputs_embeds=prompt_embeds.to(model.dtype), use_cache=True)
    past_kv = out.past_key_values
    logits = out.logits[0, -1, :].float()  # logits of last step

    concept_vecs = []
    sample_ids_list = []
    sample_probs_list = []
    argmax_ids_list = []

    trigger_probs = []
    window_sums = []

    stop_reason = "max_steps"

    max_tokens = CFG.rl_config.max_tokens
    for _ in range(max_tokens):
        probs = F.softmax(
            logits, dim=-1
        )  # temp will be applied in build_concept_embedding

        trigger_p = probs[ANSWER_TRIGGER_IDS].sum().item()  # a criterion for stopping
        trigger_probs.append(trigger_p)
        window_sum = sum(trigger_probs[-3:])
        window_sums.append(window_sum)

        if (logits.argmax().item() in _trigger_id_set) or (window_sum > 0.8):
            stop_reason = "threshold"
            break

        argmax_ids_list.append(logits.argmax(dim=-1).item())

        concept_vec, sample_ids, sample_probs = build_concept_embedding(
            logits, temp, approach
        )

        concept_vecs.append(concept_vec)
        sample_ids_list.append(sample_ids)
        sample_probs_list.append(sample_probs)

        concept_embed = concept_vec.to(model.dtype).unsqueeze(0).unsqueeze(0)
        out = model(
            inputs_embeds=concept_embed, past_key_values=past_kv, use_cache=True
        )
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
        window_sums,
        logits,
    )


@torch.no_grad()
def rollout_single_answer_sequence(past_kv_ans: object, logits: torch.Tensor):
    gen_ids = []
    current_text = ""

    max_tokens = CFG.rl_config.max_tokens
    for _ in range(max_tokens):
        # TODO: should we specify a temperature != 1.0 exlusively for answer generation?
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1).item()

        current_text += tokenizer.decode(next_id, skip_special_tokens=True)
        gen_ids.append(next_id)

        if (
            next_id == tokenizer.eos_token_id
            or "}" in current_text
            or "\n\n" in current_text
        ):
            break

        next_embed = model_embeddings(torch.tensor([[next_id]], device=device))
        out = model(
            inputs_embeds=next_embed, past_key_values=past_kv_ans, use_cache=True
        )
        past_kv_ans = out.past_key_values
        logits = out.logits[0, -1, :].float()

    return gen_ids


@torch.no_grad()
def rollout_single(prompt: str, gt_answer: str, temp: float, approach: str):
    prompt_ids = tokenizer(
        prompt, return_tensors="pt", add_special_tokens=False
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
        first_answer_logits,
    ) = rollout_single_think_sequence(prompt_ids, temp, approach)

    if stop_reason == "threshold":
        # answer token already generated, only add prefix
        prefix_ids = prefix_ids_threshold
    else:
        # force added answer token
        prefix_ids = prefix_ids_maxsteps

    out = model(input_ids=prefix_ids, past_key_values=past_kv_think, use_cache=True)
    past_kv_ans = out.past_key_values
    first_answer_logits = out.logits[0, -1, :].float()

    answer_ids = rollout_single_answer_sequence(past_kv_ans, first_answer_logits)
    generated_text = BOXED_STRING + tokenizer.decode(
        answer_ids, skip_special_tokens=True
    )

    return {
        "prompt_ids": prompt_ids,
        "prefix_ids": prefix_ids,
        "concept_vecs": concept_vecs,
        "sample_ids_list": sample_ids_list,
        "sample_probs_list": sample_probs_list,
        "argmax_ids_list": argmax_ids_list,
        "trigger_probs": trigger_probs,
        "window_sums": window_sums,
        "stop_reason": stop_reason,
        "answer_ids": answer_ids,
        "generated_text": generated_text,
    }


def policy_log_prob(rollout, approach, temp):
    prompt_embeds = model_embeddings(rollout["prompt_ids"]).detach()
    prompt_len = prompt_embeds.size(1)

    concept_vecs = torch.stack(rollout["concept_vecs"]).to(device, dtype=model.dtype)
    concept_vecs = concept_vecs.unsqueeze(0)

    full_embeds = torch.cat([prompt_embeds, concept_vecs], dim=1)

    out = model(inputs_embeds=full_embeds, use_cache=False)

    logits = out.logits[0, prompt_len - 1 : -1, :].float()
    probs = F.softmax(logits / temp, dim=-1)

    total_log_prob = 0.0

    if approach == "token_sampling":
        log_probs = F.log_softmax(logits / temp, dim=-1)

        for t in range(len(rollout["concept_vecs"])):
            sample_ids = rollout["sample_ids_list"][t]
            sampled_probs = probs[t, sample_ids]

            cumulative_probs = sampled_probs.cumsum(dim=0)
            remaining = 1.0 - torch.cat(
                [torch.zeros(1, device=device), cumulative_probs[:-1]]
            )
            remaining = remaining.clamp(min=1e-8)

            log_conditionals = log_probs[t, sample_ids] - remaining.log()
            total_log_prob = total_log_prob + log_conditionals.sum()
    elif approach == "gaussian_noise":
        soft_vecs = soft_thinking(logits / temp, vocab_embeds_device)
        total_log_prob += -(
            (concept_vecs.squeeze(0).float() - soft_vecs).pow(2).sum()
        ) / (2 * sigma**2)
    elif approach == "dirichlet_sampling":
        pass

    return total_log_prob


def train(start_step: int = 0):
    approach = "token_sampling"

    data_iter = iter(
        DataLoader(dataset, batch_size=CFG.rl_config.batch_size, shuffle=True)
    )
    optimizer.zero_grad()

    running_reward = 0.0
    running_loss = 0.0

    n_skipped_batches = 0

    t_start = time.perf_counter()
    t_window = t_start

    for step in range(start_step, CFG.rl_config.max_update_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            print("WARNING: Dataset is exhausted; now taking already seen samples")
            data_iter = iter(
                DataLoader(dataset, batch_size=CFG.rl_config.batch_size, shuffle=True)
            )
            batch = next(data_iter)

        temp = get_temp(step)
        n_skipped_questions = 0

        for question, answer in zip(batch["question"], batch["answer"]):
            # rollouts
            prompt = make_prompt(question)

            group_rollouts = [
                rollout_single(prompt, answer, temp, approach)
                for _ in range(CFG.rl_config.group_size)
            ]

            rss_gb, live_gb, driver_gb = get_memory_gb(device)
            log(
                seq_log,
                f" | mem rss {rss_gb:.3f} live {live_gb:.3f} drv {driver_gb:.3f}GB",
            )

            group_rewards = torch.tensor(
                [
                    gsm8k_reward(rollout["generated_text"], answer)
                    for rollout in group_rollouts
                ],
                dtype=torch.float32,
            )
            running_reward += group_rewards.mean().item()

            # log_rollouts(
            #     seq_log,
            #     tokenizer,
            #     group_rollouts,
            #     group_rewards,
            #     question,
            #     answer,
            #     step,
            # )

            if group_rewards.unique().numel() == 1:
                # no step given invariant rewards
                n_skipped_questions += 1
                continue

            # losses
            baseline_rewards = (group_rewards.sum() - group_rewards) / (
                len(group_rewards) - 1
            )
            advantages = group_rewards - baseline_rewards
            for i, rollout in enumerate(group_rollouts):
                log_prob = policy_log_prob(rollout, approach, temp)

                A = advantages[i].to(device)
                loss_i = (-A * log_prob) / (
                    CFG.rl_config.group_size
                    * CFG.rl_config.gradient_accumulation_steps
                    * CFG.rl_config.batch_size
                )
                loss_i.backward()

                running_loss += loss_i.item()

        if n_skipped_questions == CFG.rl_config.batch_size:
            n_skipped_batches += 1

        if (step > 0) and step % CFG.rl_config.gradient_accumulation_steps == 0:
            # update every gradient_accumulation_steps
            # TODO: good idea to clip gradients?
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                CFG.rl_config.max_grad_norm,
            )
            optimizer.step()
            optimizer.zero_grad()

        if (step > 0) and (step % LOG_EVERY == 0):
            now = time.perf_counter()
            window_secs = now - t_window
            elapsed = now - t_start
            secs_per_step = window_secs / LOG_EVERY
            steps_left = CFG.rl_config.max_update_steps - step
            eta_secs = secs_per_step * steps_left

            rss_gb, live_gb, driver_gb = get_memory_gb(device)

            denom = max(LOG_EVERY - n_skipped_batches, 1)
            avg_r = running_reward / (LOG_EVERY * CFG.rl_config.batch_size)
            avg_l = running_loss / denom

            log(
                seq_log,
                f"\n>>> step {step:5d} | loss {avg_l:.4f} | reward {avg_r:.3f}"
                f" | temp {temp:.3f}"
                f" | skipped {n_skipped_questions}/{CFG.rl_config.batch_size}"
                f" | mem rss {rss_gb:.3f} live {live_gb:.3f} drv {driver_gb:.3f}GB"
                f" | {secs_per_step:.1f}s/step"
                f" | elapsed {format_duration(elapsed)}"
                f" | ETA {format_duration(eta_secs)}",
            )

            running_reward = 0.0
            running_loss = 0.0
            n_skipped_batches = 0
            t_window = now

        if (step > 0) and (step % SAVE_EVERY == 0):
            ckpt = os.path.join(OUTPUT_DIR, f"step_{step:05d}")
            model.save_pretrained(ckpt)
            tokenizer.save_pretrained(ckpt)
            torch.save(
                {
                    "step": step,
                    "optimizer": optimizer.state_dict(),
                    "rng_state": torch.get_rng_state(),
                },
                os.path.join(ckpt, "training_state.pt"),
            )
            log(seq_log, f"  -> saved {ckpt}")


if __name__ == "__main__":
    train(start_step=start_step)
