import math
import random
from dataclasses import dataclass

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.latent_embedding_experiments.algorithms.solver import geometric_solver

# =========================================================
# CONFIG
# =========================================================


@dataclass
class Config:
    model_id: str = "meta-llama/Llama-3.1-8B-Instruct"
    temperature: float = 0.6  # Cooled down to match the Soft Thinking paper
    k: int = 5
    menu_size: int = 20  # The size of our new interrogation lineup
    display_k: int = 10
    log_file: str = (
        "src/latent_embedding_experiments/logs/llama_8b_latent_comparisons_menu_understanding.txt"
    )

    # --- automated menu-comparison experiments ---
    num_trials: int = 100
    seed: int = 42
    max_attempts_per_trial: int = 50


CFG = Config()

CONCEPT_CANDIDATES = [
    "mountain",
    "river",
    "forest",
    "ocean",
    "desert",
    "valley",
    "island",
    "cave",
    "storm",
    "rain",
    "snow",
    "wind",
    "thunder",
    "mist",
    "shadow",
    "sun",
    "moon",
    "flame",
    "ice",
    "stone",
    "metal",
    "glass",
    "gold",
    "steel",
    "cold",
    "warm",
    "dark",
    "bright",
    "silent",
    "loud",
    "bitter",
    "sweet",
    "smoke",
    "dust",
    "ash",
    "glow",
    "neon",
    "velvet",
    "crystal",
    "ancient",
    "modern",
    "royal",
    "sacred",
    "urban",
    "wild",
    "ruined",
    "gentle",
    "violent",
    "lonely",
    "peaceful",
    "grim",
    "strange",
    "magic",
    "hope",
    "grief",
    "anger",
    "fear",
    "joy",
    "envy",
    "pride",
    "love",
    "mercy",
    "chaos",
    "order",
    "freedom",
    "memory",
    "dream",
    "tower",
    "garden",
    "temple",
    "palace",
    "castle",
    "bridge",
    "cathedral",
    "machine",
    "sword",
    "crown",
    "mirror",
    "lantern",
]

# =========================================================
# UTIL
# =========================================================


def emit(text, f=None):
    print(text)
    if f:
        f.write(text + "\n")


# =========================================================
# MODEL
# =========================================================

print(f"Awakening model: {CFG.model_id}")
tokenizer = AutoTokenizer.from_pretrained(CFG.model_id)
model = AutoModelForCausalLM.from_pretrained(
    CFG.model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
model.eval()

device = model.device

raw_embeddings = model.get_input_embeddings().weight.detach().to(torch.float32)
norm_dictionary = F.normalize(raw_embeddings, dim=1)

# =========================================================
# VECTOR FORGING
# =========================================================


def build_vectors(target_ids, logits):
    raw = raw_embeddings[target_ids]

    probs_soft = F.softmax(logits / CFG.temperature, dim=0)
    solver_pool_probs = F.softmax(logits, dim=0)

    v_colar = raw.sum(dim=0, keepdim=True) / math.sqrt(len(target_ids))
    v_soft = (raw * probs_soft.unsqueeze(1)).sum(dim=0, keepdim=True)

    magnitude = v_soft.norm(dim=1).item()

    with torch.enable_grad():
        v_solver = geometric_solver(
            target_norm=F.normalize(raw, dim=1),
            target_ids=target_ids,
            dict_norm=norm_dictionary,
            magnitude=magnitude,
            pool_probs=solver_pool_probs,
            temperature=CFG.temperature,
        )

    return v_colar, v_soft, v_solver


# =========================================================
# TOKEN POOL SAMPLING
# =========================================================


def build_curated_concept_pool(words, tokenizer):
    valid = []
    rejected = []

    for word in words:
        found = False
        for form in [f" {word}", word]:
            ids = tokenizer.encode(form, add_special_tokens=False)
            if len(ids) == 1:
                tid = ids[0]
                decoded = tokenizer.decode([tid]).strip()
                valid.append(
                    {
                        "id": tid,
                        "surface": decoded,
                        "source_word": word,
                    }
                )
                found = True
                break
        if not found:
            rejected.append(word)

    deduped = []
    seen_ids = set()
    for item in valid:
        if item["id"] not in seen_ids:
            deduped.append(item)
            seen_ids.add(item["id"])

    return deduped, rejected


# =========================================================
# MENU PROMPTS + EVAL
# =========================================================


def build_menu_prompt_text(menu_words):
    menu_str = ", ".join(menu_words)
    prefix = (
        "Instruction: A hidden conceptual representation has been injected directly into your mind.\n"
        "It is a complex blend of exactly one primary ingredient and several minor ingredients.\n"
        "Here is a lineup of candidate words:\n"
        f"[{menu_str}]\n"
        "Analyze the injected representation and identify the single strongest ingredient used to create it.\n"
        "Answer with exactly one word from the list above.\n"
    )
    suffix = "\nStrongest Ingredient:"
    return prefix, suffix


def score_menu_question(vec, target_ids, true_dominant_id, rng):
    # 1. Find the natural neighbors of this specific vector
    vec_unit = F.normalize(vec, dim=1)
    sims = vec_unit @ norm_dictionary.T

    # Grab plenty of neighbors so we have enough decoys after filtering out true ingredients
    _, neighbor_idxs = torch.topk(sims, CFG.menu_size * 2, dim=1)
    neighbor_list = neighbor_idxs[0].tolist()

    # 2. Build the menu (True ingredients + Top Decoys)
    menu_ids = list(target_ids)
    for nid in neighbor_list:
        if len(menu_ids) >= CFG.menu_size:
            break
        if nid not in menu_ids:
            menu_ids.append(nid)

    # 3. Shuffle to destroy ordering bias
    rng.shuffle(menu_ids)
    menu_words = [tokenizer.decode([mid]).strip() for mid in menu_ids]

    prefix, suffix = build_menu_prompt_text(menu_words)

    prefix_ids = tokenizer(
        prefix, return_tensors="pt", add_special_tokens=True
    ).input_ids.to(device)
    suffix_ids = tokenizer(
        suffix, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(device)

    prefix_emb = model.get_input_embeddings()(prefix_ids)
    suffix_emb = model.get_input_embeddings()(suffix_ids)

    vec = vec.unsqueeze(0).to(model.dtype)
    inputs = torch.cat([prefix_emb, vec, suffix_emb], dim=1)

    with torch.no_grad():
        out = model(inputs_embeds=inputs)

    # 4. Measure the model's instinct (Logits for the menu options)
    logits = out.logits[0, -1].float()
    probs = F.softmax(logits, dim=-1)

    # Calculate how much probability mass it assigns to the true dominant vs the rest of the menu
    dominant_prob = probs[true_dominant_id].item()

    menu_probs = {mid: probs[mid].item() for mid in menu_ids}
    chosen_id = max(menu_probs, key=menu_probs.get)

    return {
        "menu_words": menu_words,
        "dominant_prob": dominant_prob,
        "chosen_id": chosen_id,
        "chosen_token": tokenizer.decode([chosen_id]).strip(),
        "is_correct": chosen_id == true_dominant_id,
    }


# =========================================================
# TRIAL SAMPLING
# =========================================================


def sample_trial(candidate_pool, rng):
    for _ in range(CFG.max_attempts_per_trial):
        sample = rng.sample(candidate_pool, CFG.k)

        target_ids = [item["id"] for item in sample]
        words = [item["surface"] for item in sample]

        # Give them random raw "logits" to simulate a pre-softmax distribution
        logits = torch.tensor(
            [rng.uniform(0.0, 4.0) for _ in range(CFG.k)],
            device=device,
            dtype=torch.float32,
        )

        # We renormalize these just like the Soft Thinking paper
        probs = F.softmax(logits / CFG.temperature, dim=0)

        # Identify the true dominant ingredient
        dominant_idx = torch.argmax(probs).item()

        return {
            "target_ids": target_ids,
            "words": words,
            "logits": logits,
            "probs": probs,
            "dominant_id": target_ids[dominant_idx],
            "dominant_token": words[dominant_idx],
        }
    return None


# =========================================================
# MAIN EXPERIMENT
# =========================================================


def run_menu_comparison_experiments():
    rng = random.Random(CFG.seed)
    torch.manual_seed(CFG.seed)

    candidate_pool, rejected = build_curated_concept_pool(CONCEPT_CANDIDATES, tokenizer)

    records = []

    with open(CFG.log_file, "w", encoding="utf-8") as f:
        emit("🌌 LATENT HALL OF MIRRORS INTERROGATION 🌌", f)
        emit(f"Trials requested: {CFG.num_trials}", f)
        emit(f"Menu Size: {CFG.menu_size} options per trial\n", f)

        completed = 0
        attempts = 0
        max_total_attempts = CFG.num_trials * 5

        while completed < CFG.num_trials and attempts < max_total_attempts:
            attempts += 1
            trial = sample_trial(candidate_pool, rng)
            if trial is None:
                continue

            target_ids = trial["target_ids"]
            probs = trial["probs"]

            _, v_soft, v_solver = build_vectors(target_ids, trial["logits"])
            methods = {
                "Soft": v_soft,
                "Solver": v_solver,
            }

            emit("=" * 110, f)
            emit(
                f"TRIAL {completed + 1} | True Ingredients: {', '.join(trial['words'])}",
                f,
            )
            emit(
                f"Dominant target: {trial['dominant_token']} ({probs[torch.argmax(probs)].item()*100:.1f}% mass)",
                f,
            )
            emit("-" * 110, f)

            for method_name, vec in methods.items():
                result = score_menu_question(
                    vec=vec,
                    target_ids=target_ids,
                    true_dominant_id=trial["dominant_id"],
                    rng=rng,
                )

                correct = result["is_correct"]
                marker = "✅" if correct else "❌"

                emit(
                    f"{method_name:>6} | Chose: '{result['chosen_token']:<10}' {marker} | "
                    f"P(Truth) = {result['dominant_prob']*100:.2f}%",
                    f,
                )

                records.append(
                    {
                        "trial": completed + 1,
                        "method": method_name,
                        "dominant_token": trial["dominant_token"],
                        "chosen_token": result["chosen_token"],
                        "correct": int(correct),
                        "dominant_prob": result["dominant_prob"],
                    }
                )

            completed += 1

        df = pd.DataFrame(records)

        emit("\n" + "#" * 110, f)
        emit("SUMMARY OF RECOGNITION (Out of 20 Options)", f)
        emit("#" * 110, f)

        if not df.empty:
            summary = (
                df.groupby("method")
                .agg(
                    accuracy=("correct", "mean"),
                    avg_prob_on_truth=("dominant_prob", "mean"),
                )
                .sort_values("accuracy", ascending=False)
            )
            emit(summary.to_string(float_format=lambda x: f"{x:.4f}"), f)
        else:
            emit("No successful trials were completed.", f)

    print(f"\nSaved → {CFG.log_file}")
    return df


if __name__ == "__main__":
    df_results = run_menu_comparison_experiments()
